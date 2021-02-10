import argparse
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, ShuffleSplit, ParameterGrid
from sklearn.feature_selection import RFECV
import pickle
import numpy as np
import pandas as pd
import json
import os
import string
import sys
import csv
from itertools import product, groupby
from functools import partial
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple

sys.path.append("../")
from foldit_data import load_extend_data, make_series
from util import PatternLookup, SubClusters, SubclusterSeries, SubSeriesLookup
from check_models import load_sub_lookup
from pattern_extraction import combine_user_series, get_predicted_lookups, get_pattern_lookups, score_param, load_TICC_output, get_pattern_label

IGNORE_COLUMNS = ['energies', 'evol_lines', 'first_pdb', 'frontier_pdbs', 'frontier_tmscores', 'lines',
                  'pid', 'timestamps', 'uid', 'upload_rate', 'upload_ratio', 'deltas', 'relevant_sids']


def compute_pattern_actions(r: pd.Series, k: int, cid: int, sub_k: int,
                            cluster_lookup: Dict[int, np.ndarray],
                            subcluster_series: SubclusterSeries,
                            puz_idx_lookup:  Dict[Tuple[str, str], Tuple[int, int]]) -> pd.Series:
    if sub_k == 0:
        puz_cs = cluster_lookup[k][slice(*puz_idx_lookup[(r.uid, r.pid)])]
        return pd.Series(sum(r.actions[puz_cs == cid]), index=["pattern_{}_actions".format(cid)])
    else:
        puz_cs = subcluster_series.series[slice(*puz_idx_lookup[(r.uid, r.pid)])]
        return pd.Series([sum(r.actions[puz_cs == scid]) for scid in subcluster_series.labels],
                         index=["pattern_{}_actions".format(l) for l in subcluster_series.labels])

def score_candidate(X, y, cv):
    selector = RFECV(GradientBoostingRegressor(loss="huber"), step=1, cv=cv)
    selector.fit(X, y)
    X_sel = selector.transform(X)
    # score for this candidate is CV score fitting on X_sel
    return np.mean(cross_val_score(GradientBoostingRegressor(loss="huber"), X_sel, y, cv=cv))


def generate_candidates(k, sub_ks, pattern_lookups, cids, cur_sub_k_idx):
    return [c for c in product(*[list(product([cid], sub_ks[cid][:cur_sub_k_idx[cid] + 1]))
                                 for cid in cids])
            if all(sub_k in pattern_lookups[k][cid] for cid, sub_k in c)]


def find_best_predictive_model(model_dir: str, user: str, data: pd.DataFrame,
                               puz_idx_lookup:  Dict[Tuple[str, str], Tuple[int, int]],
                               pattern_lookups: Dict[int, PatternLookup],
                               cluster_lookup: Dict[int, np.ndarray],
                               subseries_lookups: Dict[int, Dict[int, SubSeriesLookup]],
                               subclusters: SubClusters,
                               action_counts) -> Tuple:
    """
    BRUTE FORCE APPROACH, memory needs are too high
    1. for each candidate (choice of k and sub-ks)
        1a. compute the action count for each pattern/subpattern
            (build data structure like existing lookups for whole suite of action counts)
            (gradually assemble, computing new counts as needed)
        1b. score model using those action count features -> candidate score
    2. return the best-scoring candidate

    NEW APPROACH
    1. compute base model scores (no subpatterns)
    2. add all cids to active list
    3. for each sub_k
        3a. generate all possible new candidates involving active cids and current sub_k
        3b. score new candidates
        3c. average scores for candidates evolving each active cids and current sub_k, compare to average for previous
            sub_k
        3d. set active list to those cids for which current average exceeds previous
    """


    # action_counts: (k, (cid, sub_k)) -> pd.DataFrame (single column for base pattern, column per subpattern otherwise)
    subcluster_series_lookup = {} # (k, (cid, sub_k)) -> SubclusterSeries

    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=304)
    scores_lookup = {}
    for k in pattern_lookups[user]:
        scores = {}  # candidate tuple -> CV score
        cids = {p.cid for p in pattern_lookups[user][k]["base"]}
        sub_ks = {cid: sorted(pattern_lookups[user][k][cid].keys()) for cid in cids}
        active_cids = list(cids)
        cur_sub_k_idx = {cid: (1 if len(sub_ks[cid]) > 1 else 0) for cid in cids}
        with Pool(25) as pool:
            round = 0
            while (len(active_cids) > 0):
                round += 1
                pool_count = 0
                pooled = []
                candidates = generate_candidates(k, sub_ks, pattern_lookups[user], cids, cur_sub_k_idx)
                print("k =", k, "round", round)
                print("active_cid =", active_cids, ":", len([c for c in candidates if c not in scores]), "candidates")
                for candidate in candidates:
                    if candidate not in scores:
                        print("candidate", candidate)
                        for (cid, sub_k) in candidate:

                            if sub_k != 0 and (k, (cid, sub_k)) not in subcluster_series_lookup:
                                all_subclusters = cluster_lookup[user][k].astype(np.str)
                                labels = ["{}{}".format(cid, string.ascii_uppercase[x]) for x in range(sub_k)]
                                cs = subclusters[user][k][cid][sub_k]
                                for (_, _, start_idx), (s, e) in subseries_lookups[user][k][cid].idx_lookup.items():
                                    all_subclusters[start_idx: start_idx + (min(e, len(cs)) - s)] = [labels[c] for c in cs[s:e]]
                                subcluster_series_lookup[(k, (cid, sub_k))] = SubclusterSeries(labels, all_subclusters)

                            if (k, (cid, sub_k)) not in action_counts:
                                f = partial(compute_pattern_actions, k=k, cid=cid, sub_k=sub_k, cluster_lookup=cluster_lookup[user],
                                            subcluster_series=subcluster_series_lookup.get((k, (cid, sub_k)), None),
                                            puz_idx_lookup=puz_idx_lookup)
                                action_counts[(k, (cid, sub_k))] = data.apply(f, axis=1)

                        features = pd.concat([data.drop(IGNORE_COLUMNS + ["time", "actions"], axis=1)] +
                                             [action_counts[(k, (cid, sub_k))] for (cid, sub_k) in candidate],
                                             axis=1)
                        X = features.drop(["perf"], axis=1).values
                        y = features["perf"].values.ravel()
                        pooled.append((candidate, pool.apply_async(score_candidate, (X, y, cv))))
                        pool_count += 1

                print("scoring k =", k, "round", round, "candidates...")
                i = 0
                for candidate, x in pooled:
                    i += 1
                    print("{} out of {}\r".format(i, pool_count), end="")
                    scores[candidate] = x.get()
                print()
                for cid in active_cids[:]:
                    prev_subk = sub_ks[cid][cur_sub_k_idx[cid] - 1]
                    cur_subk = sub_ks[cid][cur_sub_k_idx[cid]]
                    prev_scores = [score for c, score in scores.items() if (cid, prev_subk) in c]
                    cur_scores = [score for c, score in scores.items() if (cid, cur_subk) in c]
                    if np.mean(prev_scores) >= np.mean(cur_scores):
                        active_cids.remove(cid)
                        cur_sub_k_idx[cid] -= 1
                cur_sub_k_idx = {cid: (idx + 1 if cid in active_cids else idx) for cid, idx in cur_sub_k_idx.items()}
                active_cids = [cid for cid in active_cids if len(sub_ks[cid]) > cur_sub_k_idx[cid]]

            print("done\n\n\n")
            scores_lookup[k] = scores
            with open(f"{model_dir}/eval/{user}_k{k}_scores.pickle", "wb") as fp:
                pickle.dump(scores, fp)

    best_k = best_candidate = None
    best_score = 0
    for k, scores in scores_lookup.items():
        for candidate, score in scores.items():
            if score > best_score:
                best_score = score
                best_k = k
                best_candidate = candidate
    return best_k, best_candidate, scores_lookup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='pattern_model_evaluation.py')
    parser.add_argument("model_dir")
    parser.add_argument("config")
    args = parser.parse_args()
    assert os.path.exists(args.model_dir)
    assert os.path.exists(args.config)
    with open(args.config) as fp:
        config = json.load(fp)


    print("loading raw data", end="...")
    # pids = ["2003433", "2003642", "2003195", "2003313", "2003287", "2002475", "2002294", "2002196", "2002141", "2002110"]
    soln_lookup = {}
    parent_lookup = {}
    child_lookup = {}
    data, puz_metas = load_extend_data(config["pids"] + config["test_pids"], soln_lookup, parent_lookup, child_lookup, False, 600)

    with open("../data/foldit/user_metadata_v4.csv") as fp:
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
    user_meta_lookup = {uid: list(metas) for uid, metas in groupby(sorted(user_metas.values(), key=lambda m: m['uid']), lambda m: m['uid'])}

    with open("../data/foldit/puz_metadata_v4.csv") as fp:
        puz_infos = {r['pid']: {'start':     int(r['start']),
                                'end':       int(r['end']),
                                'baseline':  float(r['baseline']),
                                'best':      float(r['best']),
                                'best_solo': float(r['best_solo'])
                               } for r in csv.DictReader(fp)}
    print("done")

    print("evaluating model at", args.model_dir)
    print("loading model data", end="...")
    noise = np.loadtxt(args.model_dir + "/noise_values.txt")
    puz_idx_lookup, series_lookup, _ = make_series(data, noise=noise)
    if "user" not in config:
        idx_lookup, all_series = combine_user_series(series_lookup, noise)
        puz_idx_lookup = {(uid, pid): (s + idx_lookup[uid][0], e + idx_lookup[uid][0])
                          for (uid, pid), (s, e) in puz_idx_lookup.items()}
        series_lookup["all"] = all_series

    krange = config["krange"]
    users = [d for d in os.listdir(args.model_dir) if os.path.isdir(f"{args.model_dir}/{d}")] if "user" in config else ["all"]

    _, mrf_lookup, model_lookup, _ = load_TICC_output(args.model_dir, users, krange)
    dummy_subseries_lookup = {user: {int(d.strip("k")): [int(c.strip("cid")) for c in os.listdir(f"{args.model_dir}/{user}/subpatterns/{d}")]
                                     for d in os.listdir(f"{args.model_dir}/{user}/subpatterns") if d.startswith("k")} for user in users}
    sub_lookup = load_sub_lookup(args.model_dir, users, dummy_subseries_lookup, config["sub_krange"])

    if os.path.exists(args.model_dir + "/eval/cluster_lookup.pickle"):
        with open(args.model_dir + "/eval/cluster_lookup.pickle", "rb") as fp:
            cluster_lookup = pickle.load(fp)
        with open(args.model_dir + "/eval/subseries_lookup.pickle", "rb") as fp:
            subseries_lookups = pickle.load(fp)
        with open(args.model_dir + "/eval/sub_clusters.pickle", "rb") as fp:
            subclusters = pickle.load(fp)
        with open(args.model_dir + "/eval/pattern_lookup.pickle", "rb") as fp:
            pattern_lookups = pickle.load(fp)
    else:
        # predict patterns on full data for all candidate models
        print("generating patterns on full data", end="...")
        cluster_lookup    = {}
        subseries_lookups = {}
        subclusters       = {}
        pattern_lookups   = {}
        with ProcessPoolExecutor(len(krange)) as pool:
            for user in users:
                pooled = []
                for k in krange:
                    pooled.append((k, pool.submit(get_predicted_lookups, series_lookup[user], k, model_lookup[user],
                                                  sub_lookup.models[user], mrf_lookup[user], puz_idx_lookup, noise)))
                cl = {}
                sl = {}
                scs = {}
                for k, task in pooled:
                    clusters, subseries, subcs = task.result()
                    cl[k] = clusters
                    sl[k] = subseries
                    scs[k] = subcs
                cluster_lookup[user] = cl;
                subseries_lookups[user] = sl;
                subclusters[user] = scs;            
                pattern_lookups[user] = get_pattern_lookups(krange, subclusters[user], sub_lookup.mrfs[user], subseries_lookups[user],
                                                            cluster_lookup[user], mrf_lookup[user], puz_idx_lookup)
        os.makedirs(args.model_dir + "/eval", exist_ok=True)
        with open(args.model_dir + "/eval/cluster_lookup.pickle", "wb") as fp:
            pickle.dump(cluster_lookup, fp)
        with open(args.model_dir + "/eval/subseries_lookup.pickle", "wb") as fp:
            pickle.dump(subseries_lookups, fp)
        with open(args.model_dir + "/eval/sub_clusters.pickle", "wb") as fp:
            pickle.dump(subclusters, fp)
        with open(args.model_dir + "/eval/pattern_lookup.pickle", "wb") as fp:
            pickle.dump(pattern_lookups, fp)
    print("done")
    # select model
    print("generating action count series", end="...")
    rows = []
    for _, r in data.iterrows():
        if r.relevant_sids is None or (r.uid, r.pid) not in puz_idx_lookup:
            continue
        deltas = sorted([d for d in r.deltas if d.sid in r.relevant_sids], key=lambda x: x.timestamp)
        actions = np.array([sum(d.action_diff.values()) for d in deltas])
        if actions.sum() == 0:
            # logging.debug("SKIPPING {} {}, no actions recorded".format(r.uid, r.pid))
            continue
        rows.append({'uid': r.uid, 'pid': r.pid, "actions": actions})
    data = data.merge(pd.DataFrame(data=rows), on=["pid", "uid"])
    assert len(rows) == len(data)  # check that data consists of only rows with an actions column

    data["experience"] = data.apply(lambda r: len([x for x in user_meta_lookup[r.uid]
                                                   if puz_infos[x['pid']]["end"] < puz_infos[r.pid]["start"]]), axis=1)
    data["median_prior_perf"] = data.apply(lambda r: np.median([float(x['perf']) for x in user_meta_lookup[r.uid]
                                                                if puz_infos[x['pid']]["end"] < puz_infos[r.pid]["start"]]), axis=1)
    data.median_prior_perf.fillna(data.median_prior_perf.median(), inplace=True)
    print("done")

    scores_lookup = {}
    for user in users:
        print("finding most predictive model for", user)
        action_counts = {}
        best_k, best_candidate, scores = find_best_predictive_model(args.model_dir, user, data, puz_idx_lookup, pattern_lookups,
                                                                    cluster_lookup, subseries_lookups, subclusters,
                                                                    action_counts)
        scores_lookup[user] = scores
        with open(f"{args.model_dir}/eval/{user}_best_model.txt", 'w') as fp:
            fp.write(str((best_k, best_candidate)) + "\n")
            
        with open(f"{args.model_dir}/eval/{user}_action_counts.pickle", "wb") as fp:
            pickle.dump(action_counts, fp)

        print("selected model:", best_k, best_candidate)
    
        ps = sum([[(get_pattern_label(p, cid, sub_k), p) for p in pattern_lookups[best_k][cid][sub_k]] for cid, sub_k in best_candidate], [])
        ps_uid_pid = {tag: sorted(xs) for tag, xs in groupby(sorted(ps, key=lambda p: (p[1].uid, p[1].pid)), lambda p: (p[1].uid, p[1].pid))}
        pattern_use_lookup = {tag: {pt for pt, _ in xs} for tag, xs in ps_uid_pid.items()}
        pts = {pt for pt, p in ps}
    
        # collect pattern features using selected model
        results = pd.concat([data] + [action_counts[(best_k, (cid, sub_k))] for (cid, sub_k) in best_candidate], axis=1)
    
        pattern_features = ["pattern_{}".format(pt) for pt in pts]
    
        acc = []
        for (uid, pid), use in pattern_use_lookup.items():
            r = {"uid": uid, "pid": pid}
            for pt in pts:
                r["pattern_"+ pt+"_use"] = 1 if pt in use else 0
            acc.append(r)
        results = results.merge(pd.DataFrame(data=acc), on=["uid", "pid"])
        #results["distinct_patterns"] = results.apply(lambda r: len(pattern_use_lookup[(r.uid, r.pid)]), axis=1)
        #results["action_count_all"] = results.apply(lambda r: user_metas[(r.uid, r.pid)]["action_count_all"], axis=1)
        # results["action_count_relevant"] = results.apply(lambda r: user_metas[(r.uid, r.pid)]["action_count_relevant"], axis=1)
        #results["action_count_best"] = results.apply(lambda r: user_metas[(r.uid, r.pid)]["action_count_best"], axis=1)
        #results["best_energy_time"] = results.apply(lambda r: user_metas[(r.uid, r.pid)]["best_energy_time"], axis=1)
        #results["action_rate_all"] = results.apply(lambda r: r.action_count_all / r.time, axis=1)
        #results["action_rate_relevant"] = results.apply(lambda r: r.action_count_relevant / r.relevant_time, axis=1)
    
    
        # find best model, compare to baseline
    
        features = results.drop(IGNORE_COLUMNS, axis=1)
    
        baseline_features = ["action_count_relevant", "median_prior_perf", "experience"]
    
        seed = 13*17*31
        models = {#"ridge": Ridge,
                  "ensemble": GradientBoostingRegressor}
        model_params = {"ridge": {"random_state": [seed], "alpha": [0.1, 0.5, 1, 5, 10], "normalize": [True, False]},
                        "ensemble": {"random_state": [seed], "learning_rate": [0.01, 0.02, 0.05, 0.1],
                                     "n_estimators": [100, 500, 1000], "n_iter_no_change": [100]}}
        # std_base = deepcopy(model_params["ensemble"])
        # std_base["loss"] = ["ls", "lad"]
        # huber_base = deepcopy(model_params["ensemble"])
        # huber_base["loss"] = ["huber"]
        # huber_base["alpha"] = [0.9, 0.95, 0.99]
        # model_params["ensemble"] = [std_base, huber_base]
        model_params["ensemble"]["loss"] = ["huber"]
        model_params["ensemble"]["alpha"] = [0.85, 0.9, 0.95, 0.99]
    
        with Pool(25, maxtasksperchild=4) as pool:
            cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=seed)
            print("fitting baseline")
            scores = {}
            X = features[baseline_features].values
            y = features["perf"].values.ravel()
            for lab, model in models.items():
                evals = []
                for param in ParameterGrid(model_params[lab]):
                    evals.append((pool.apply_async(score_param, (param, model, X, y, cv)), param))
                print("{}: {} scores sent to pool, collecting results".format(lab, len(evals)))
                scores[lab] = [(x.get(), param) for x, param in evals]
            baseline_scores = scores
            print("baseline")
            print(max(scores["ensemble"], key=lambda x: x[0][0]))
    
            model_scores = {}
            for ftype in ["actions", "use"]:
                print("fitting pattern {} models".format(ftype))
                candidate_features = ["median_prior_perf", "experience"] + ["{}_{}".format(f, ftype) for f in pattern_features]
                if ftype == "use":
                    candidate_features.append("action_count_relevant")
                scores = {}
                X = features[candidate_features].values
                y = features["perf"].values.ravel()
                for lab, model in models.items():
                    evals = []
                    for param in ParameterGrid(model_params[lab]):
                        evals.append((pool.apply_async(score_param, (param, model, X, y, cv)), param))
                    print("{}: {} scores sent to pool, collecting results".format(lab, len(evals)))
                    scores[lab] = [(x.get(), param) for x, param in evals]
                model_scores[ftype] = scores
                print("best {} model".format(ftype))
                print(max(scores["ensemble"], key=lambda x: x[0][0]))
    
        with open(f"{args.model_dir}/eval/{user}_baseline_scores.pickle", 'wb') as fp:
            pickle.dump(baseline_scores, fp)
        with open(f"{args.model_dir}/eval/{user}_model_scores.pickle", 'wb') as fp:
            pickle.dump(model_scores, fp)
