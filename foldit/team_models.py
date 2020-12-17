import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import plot_partial_dependence, permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, ShuffleSplit, ParameterGrid, GroupShuffleSplit
from sklearn.feature_selection import RFECV
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import csv
import json
import networkx as nx
import argparse
from datetime import datetime
from itertools import groupby, combinations, chain
from collections import Counter
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
import sys
sys.path.append("../")

from plot_util import make_boxplot


def get_first_time(team):
    xs = [team]
    front = team['children'][:]
    while len(front) > 0:
        cur = front.pop()
        xs.append(cur)
        front.extend(cur["children"])
    return min(x['start'] for x in xs if x['start'])
    

def get_last_time(team):
    xs = [team]
    front = team['children'][:]
    while len(front) > 0:
        cur = front.pop()
        xs.append(cur)
        front.extend(cur["children"])
    return max(x['end'] for x in xs if x['end'])


def get_team_depth(team):
    if len(team['children']) == 0:
        return 1
    return 1 + max([get_team_depth(c) for c in team['children']])


def get_team_breadths(team):
    max_depth = get_team_depth(team)
    depth = 0
    front = [(0, team)]
    breadths = []
    while depth < max_depth:
        breadths.append(len([d for d, t in front if d == depth]))
        front = sum([[(d + 1, c) for c in t['children']] for d, t in front], [])
        depth += 1
    return breadths


def get_team_breadth(team):
    return max(get_team_breadths(team))


def get_uids_tuple(team):
    uids = [team['uid']]
    front = team['children'][:]
    while len(front) > 0:
        cur = front.pop()
        uids.append(cur['uid'].split("evol")[0].split("_")[0])
        front.extend(cur['children'])
    return tuple(set(uids))


def get_linked_teams(teams):
    linked_teams = {tag: sorted(ts, key=lambda t: t['energy'], reverse=True) for tag, ts in
                    groupby(sorted(teams, key=lambda t: (t['uid'], t['pid'])), lambda t: (t['uid'], t['pid']))}
    tid = 0
    for tag, ts in list(linked_teams.items()):
        new_t = deepcopy(ts[0])
        new_t["parent"] = None
        cur = new_t
        for i, t in enumerate(ts[1:], 1):
            # screen the rare case where the same evolution gets matched with multiple indistinguishable shares
            if cur["children"] != t["children"]:
                cur["children"].append(deepcopy(t))
                cur["children"][-1]["parent"] = cur
                cur = cur["children"][-1]
                cur['uid'] = cur['uid'] + "_" + str(i)
        new_t["first_time"] = get_first_time(new_t)
        new_t["last_time"] = get_last_time(new_t)
        new_t["members"] = get_uids_tuple(new_t)
        new_t["tid"] = tid
        tid += 1
        linked_teams[tag] = new_t
    return linked_teams    


def populate_graph(team, G):
    G.add_node(team['uid'], color="blue" if 'e' in team['uid'] else "red")
    for child in team['children']:
        G.add_edge(team['uid'], child['uid'])
        populate_graph(child, G)


def get_collab_nodes(team):
    nodes = [team]
    idx = 0
    while idx < len(nodes):
        nodes.extend(nodes[idx]["children"])
        idx += 1
    return nodes


def is_iterative(team):
    return len([node for node in get_collab_nodes(team) if "evol" not in node["uid"]]) > 1


def count_branches(team):
    return len([node for node in get_collab_nodes(team) if len(node["children"]) > 1])


def make_team_ontology(teams):
    ont = {}
    # one shared solo solution with one evolver
    ont["minimal"] = [t for t in teams if get_team_depth(t) == 2 and get_team_breadth(t) == 1]
    ont["line"] = [t for t in teams if get_team_depth(t) > 2 and get_team_breadth(t) == 1]
    ont["one_branch"] = [t for t in teams if count_branches(t) == 1 and not is_iterative(t)]
    ont["one_branch_itr"] = [t for t in teams if count_branches(t) == 1 and is_iterative(t)]
    ont["n_branch"] = [t for t in teams if count_branches(t) > 1 and count_branches(t) < get_team_depth(t) and not is_iterative(t)]
    ont["n_branch_itr"] = [t for t in teams if count_branches(t) > 1 and count_branches(t) < get_team_depth(t) and is_iterative(t)]
    ont["rich"] = [t for t in teams if count_branches(t) >= get_team_depth(t) and not is_iterative(t)]
    ont["rich_itr"] = [t for t in teams if count_branches(t) >= get_team_depth(t) and is_iterative(t)]
    assert len(teams) == len(sum(ont.values(), []))
    return ont


def get_team_uids(team):
    uids = [team['uid']]
    front = team['children'][:]
    while len(front) > 0:
        cur = front.pop()
        uids.append(cur['uid'].split("_")[0])
        front.extend(cur['children'])
    return uids


def get_team_perf(team, metas):
    return max(metas[(uid, team['pid'])]['perf'] if (uid, team['pid']) in metas else 0 for uid in get_team_uids(team))


def get_collab_timings(team, puz_s, puz_e):
    timings = []
    front = team['children'][:]
    while len(front) > 0:
        cur = front.pop()
        if cur["source"]:
            timings.append(cur["start"])
        front.extend(cur['children'])
    return [(s - puz_s) / (puz_e - puz_s) for s in timings]


def get_previous_collabs(team, all_teams):
    prior_teams = [t for t in all_teams if t["last_time"] < team["first_time"]]
    subsets = chain.from_iterable(combinations(team["members"], n) for n in range(2, len(team["members"]) + 1))
    return {subset: len([t for t in prior_teams if set(subset).issubset(set(t["members"]))]) for subset in subsets}


def get_prior_score(team, all_teams):
    prior = get_previous_collabs(team, all_teams)
    score = 0
    for subset, n in prior.items():
        score += len(subset) * (len(subset) - 1) / 2 * n
    return score / len(team["members"])


def score_param(param, model, X, y, cv):
    # feature selection under these params
    selector = RFECV(model(**param), step=1, cv=cv)
    selector.fit(X, y)
    X_sel = selector.transform(X)
    # score for these params is CV score fitting on X_sel
    return np.mean(cross_val_score(model(**param), X_sel, y, cv=cv))


def diagnose_patterns(team):
    solo_pts = []
    evol_pts = []
    for node in get_collab_nodes(team):
        if "evol" in node['uid']:
            evol_pts.extend([pt for pt, c in node["pattern_count"].items() if c > 0])
        else:
            solo_pts.extend([pt for pt, c in node["pattern_count"].items() if c > 0])
    return set(solo_pts).difference(set(evol_pts)), set(evol_pts).difference(set(solo_pts)), set(solo_pts).intersection(set(evol_pts))


def total_actions(team):
    total = 0
    for node in get_collab_nodes(team):
        for pt, c in node["pattern_count"].items():
            total += c
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='team_models.py')
    parser.add_argument("--render-teams", action="store_true")
    parser.add_argument("--fit-models", action="store_true")
    parser.add_argument("--ontology-report", action="store_true")
    parser.add_argument("--feature-file")
    args = parser.parse_args()

    seed = 13*17*31
    target_pids = ["2003111", "2003125", "2003177", "2003195", "2003206", "2003236", "2003240", "2003270", "2003281", "2003287",
                   "2003303", "2003313", "2003322", "2003340", "2003374", "2003383", "2003388", "2003416", "2003433", "2003465",
                   "2003483", "2003535", "2003578", "2003583", "2003621", "2003639", "2003642", "2003668", "2003698", "2003730",
                   "2003734", "2003762", "2003780", "2003784", "2003791", "2003814", "2003823", "2003825", "2003846", "2003854",
                   "2003856", "2003876", "2003881", "2003893", "2003900", "2003903", "2003912", "2003918"]

    print("loading data... ", end="")
    data_dir = "/mathcs1/awb/data_files"
    with open(data_dir + "/user_teams_v3.json") as fp:
        all_teams = [json.loads(line) for line in fp]
    teams = [t for t in all_teams]# if t['pid'] in target_pids]

    with open(data_dir + "/user_metadata_v4.csv") as fp:
        user_metas = {(r['uid'], r['pid']): r for r in csv.DictReader(fp)}
        for v in user_metas.values():
            v['time'] = int(v['time'])
            v['relevant_time'] = int(float(v['relevant_time']))
            v['best_energy_time'] = int(v['best_energy_time'])
            v['action_count_all'] = int(v['action_count_all'])
            v['action_count_relevant'] = int(v['action_count_relevant'])
            v['action_count_best'] = int(v['action_count_best'])
            v['best_energy'] = float(v['best_energy'])
            v['perf'] = float(v['perf'])
            v['solo_perf'] = float(v['solo_perf']) if v['solo_perf'] != "" else np.nan

    with open(data_dir + "/puz_metadata_v4.csv") as fp:
        puz_infos = {r['pid']: {'start':     int(r['start']),
                                'end':       int(r['end']),
                                'baseline':  float(r['baseline']),
                                'best':      float(r['best']),
                                'best_solo': float(r['best_solo'])
                               } for r in csv.DictReader(fp)}

    with open(data_dir + "/user_patterns_v1.csv") as fp:
        reader = csv.DictReader(fp)
        pattern_count_lookup = {}
        for r in reader:
            pattern_count_lookup[(r['uid'], r['pid'])] = {pt: float(c) for pt, c in r.items() if pt != 'uid' and pt != 'pid'}
    print("done")

    train_idx, test_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(teams, groups=[t['pid'] for t in teams]))
    test_teams = [teams[i] for i in test_idx]
    teams = [teams[i] for i in train_idx]
    print("split data into {} training teams and {} test teams".format(len(teams), len(test_teams)))

    linked_teams = get_linked_teams(teams)
    all_linked_teams = get_linked_teams(all_teams)
    print("linked teams by soloist, producing {} linked teams ({} train, {} test)".format(len(all_linked_teams), len(linked_teams),
                                                                                          len(all_linked_teams) - len(linked_teams)))
    teams_by_pid = {pid: list(ts) for pid, ts in groupby(sorted(all_linked_teams.values(), key=lambda x: x['pid']), lambda x: x['pid'])}

    if args.render_teams:
        print("rendering teams")
        node_labels = {}
        for team in all_linked_teams.values():
            node_labels[team['uid']] = team['uid']
            front = team['children'][:]
            while len(front) > 0:
                cur = front.pop()
                node_labels[cur['uid']] = "_".join(cur['uid'].split("evol"))
                front.extend(cur['children'])

        graphs = []
        for tag, t in all_linked_teams.items():
            if get_team_depth(t) > 2 and get_team_breadth(t) > 1:
                G = nx.DiGraph()
                populate_graph(t, G)
                G.name = "_".join(tag)
                graphs.append(G)

        dim = int(len(graphs)**0.5) + 1
        fig = plt.figure(figsize=(3*dim,3*dim))
        for i, G in enumerate(sorted(graphs, key=lambda g: len(g.nodes)), 1):
            print("{} out of {}\r".format(i, len(graphs)), end="")
            ax = fig.add_subplot(dim, dim, i)
            ax.set_title(G.name)
            pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
            nx.draw_networkx(G, pos, ax=ax, node_size=50, labels={n: v for n, v in node_labels.items() if n in G.node},
                             node_color=[d["color"] for n, d in G.nodes(data=True)], font_size=6)
        print()
        fig.tight_layout()
        fig.savefig("team_model_output/complex_linked_teams.png")
        print("done")

    if args.ontology_report:
        ontology = make_team_ontology(all_linked_teams.values())
        ontology_lookup = {(t['uid'], t['pid']): teamtype for teamtype, ts in ontology.items() for t in ts}
        teamtype_labels = {"minimal": "Minimal", "line": "Line", "one_branch": "One Branch", "one_branch_itr": "One Branch Iterative",
                           "n_branch": "N Branch", "n_branch_itr": "N Branch Iterative", "rich": "Rich", "rich_itr": "Rich Iterative"}
        make_boxplot([[get_team_perf(t) for t in ts] for ts in ontology.values()], [teamtype_labels[teamtype] for teamtype in ontology],
                     "performance", "team_model_output/ontology_perfs.png", ylims=(0.5, 1.02))
        make_boxplot([[len([t for t in ts if ontology_lookup[(t['uid'], t['pid'])] == teamtype]) / len(ts) for ts in teams_by_pid.values()] for teamtype in ontology],
                     [teamtype_labels[teamtype] for teamtype in ontology], "proportion of teams", "team_model_output/ontology_variation.png", ylims=(-0.05, 1))

    solo_pts, evol_pts, both_pts = zip(*[diagnose_patterns(team) for team in all_linked_teams.values()])
    solo_count = Counter(sum([list(x) for x in solo_pts], []))
    evol_count = Counter(sum([list(x) for x in evol_pts], []))
    both_count = Counter(sum([list(x) for x in both_pts], []))

    if args.feature_file and os.path.exists(args.feature_file):
        print("loading features from {}".format(args.feature_file))
        features_tagged = pd.read_csv(args.feature_file, index_col=0)
    else:
        print("computing features for {} teams".format(len(all_linked_teams)))
        acc = []
        for i, ((uid, pid), team) in enumerate(all_linked_teams.items()):
            print("team {}\r".format(i), end="")
            d = {"uid": uid, "pid": pid}
            d["actions"] = total_actions(team)
            d["perf"] = get_team_perf(team, user_metas)
            d["size"] = len(get_uids_tuple(team))
            d["depth"] = get_team_depth(team)
            d["breadth"] = get_team_breadth(team)
            timings = get_collab_timings(team, puz_infos[team['pid']]['start'], puz_infos[team['pid']]['end'])
            d["first_collab"] = min(timings)
            d["num_evolves"] = len(timings)
            d["prior_collab"] = get_prior_score(team, all_linked_teams.values())
            puz_info = puz_infos[team["pid"]]
            d["starting_perf"] = min(0, team['energy'] - puz_info["baseline"]) / (puz_info["best"] - puz_info["baseline"])
            # d["improve"] = d["perf"] - d["starting_perf"] # <- definitely seems like a bug...
            d["improve_norm"] = (d["perf"] - d["starting_perf"]) / (1 - d["starting_perf"])
            # for each soloist iteration with timing info, count the number of direct evolves of the previous share that began before the next share, then sum
            d["feedback"] = sum(len([c for c in node["parent"]["children"] if c != node and c["start"] < node["end"]])
                                for node in get_collab_nodes(team) if "parent" in node and node["parent"] and node["end"])
            # for each other team on this puzzle, count how many uids overlap with this team, then sum
            d["connections"] = sum(len(set(t['members']).intersection(set(team['members']))) for t in teams_by_pid[team['pid']] if t != team)
            acc.append(d)
        features_tagged = pd.DataFrame(data=acc)
        print("done")
        if args.feature_file:
            print("saving features to {}".format(args.feature_file))
            features_tagged.to_csv(args.feature_file)

    if args.fit_models:
        models = {#"ridge": Ridge,
                  "ensemble": GradientBoostingRegressor}
        model_params = {"ridge": {"random_state": [seed], "alpha": [0.1, 0.5, 1, 5, 10], "normalize": [True, False]},
                        "ensemble": {"random_state": [seed], "learning_rate": [0.01, 0.02, 0.05, 0.1], "subsample": [0.3, 0.5, 0.7],
                                     "n_estimators": [1000], "n_iter_no_change": [200]}}
        # std_base = deepcopy(model_params["ensemble"])
        # std_base["loss"] = ["ls", "lad"]
        # huber_base = deepcopy(model_params["ensemble"])
        # huber_base["loss"] = ["huber"]
        # huber_base["alpha"] = [0.9, 0.95, 0.99]
        # model_params["ensemble"] = [std_base, huber_base]
        model_params["ensemble"]["loss"] = ["huber"]
        model_params["ensemble"]["alpha"] = [0.85, 0.9, 0.95, 0.99]

        print("scoring models")
        outcomes = ["perf", "improve_norm"]
        features = features_tagged.drop(["uid", "pid"], axis=1)
        model_scores = {}
        with Pool() as pool:
            cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=seed)
            for outcome in outcomes:
                print("finding best model for", outcome)
                scores = {}
                X = features.drop(outcomes, axis=1).values
                y = features[outcome].values
                for lab, model in models.items():
                    results = []
                    for param in ParameterGrid(model_params[lab]):
                        results.append((pool.apply_async(score_param, (param, model, X, y, cv)), param))
                    print("{}: {} scores sent to pool, collecting results".format(lab, len(results)))
                    scores[lab] = [(x.get(), param) for x, param in results]
                model_scores[outcome] = scores

        for outcome, scores in model_scores.items():
            print("plotting results for", outcome)
            X = features.drop(outcomes, axis=1).values
            y = features[outcome].values
            lab, (score, param) = max([(lab, max(results, key=lambda x: x[0])) for lab, results in scores.items()], key=lambda x: x[1][0])
            selector = RFECV(model(**param), step=1, cv=cv).fit(X, y)
            X_sel = features.drop(outcomes, axis=1).iloc[:, selector.get_support()]
            best_model = models[lab](**param).fit(X_sel, features[outcome])

            # labels = np.array(["Breadth", "Connections with Other Teams", "Depth", "Feedback", "Start of Collaboration",
            #                   "Number of Evolutions", "Measure of Prior Collaboration", "Team Size", "Quality of Initial Soloist Solution"])
            starting_perf_idx = X_sel.columns.get_loc("starting_perf")
            fig, ax = plt.subplots(figsize=(20, 5 * ((X_sel.shape[1] - 1) // 3 + 1)))
            plot_partial_dependence(best_model, X_sel, [i for i in range(X_sel.shape[1]) if i != starting_perf_idx], feature_names=X_sel.columns,
                                    grid_resolution=50, percentiles=(0.05, 0.95), line_kw={"linewidth": 5}, ax=ax, n_jobs=-1)
            fig.tight_layout()
            fig.savefig("team_model_output/team_structure_PDP_{}.png".format(outcome))
            plt.close(fig)

            # Plot feature importance
            feature_importance = best_model.feature_importances_
            # make importances relative to max importance
            feature_importance = 100.0 * (feature_importance / feature_importance.max())
            sorted_idx = np.argsort(feature_importance)
            pos = np.arange(sorted_idx.shape[0]) + .5
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.barh(pos, feature_importance[sorted_idx], align='center')
            ax.set_yticks(pos)
            ax.set_yticklabels(X_sel.columns[sorted_idx])
            ax.set_xlabel('Relative Importance')
            ax.set_title('Variable Importance')
            fig.savefig("team_model_output/feature_importance_{}.png".format(outcome))
            plt.close(fig)

            result = permutation_importance(best_model, X_sel, y, n_repeats=10, random_state=seed, n_jobs=-1)
            sorted_idx = result.importances_mean.argsort()

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.boxplot(result.importances[sorted_idx].T,
                       vert=False, labels=X_sel.columns[sorted_idx])
            ax.set_title("Permutation Importances (train set)")
            ax.set_xlabel("Relative Importance");
            fig.tight_layout()
            fig.savefig("team_model_output/feature_importance_{}_permutation.png".format(outcome))
            plt.close(fig)
