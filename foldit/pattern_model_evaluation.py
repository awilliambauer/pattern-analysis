import argparse
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GroupShuffleSplit, ParameterGrid, GroupKFold
from sklearn.feature_selection import RFECV
import pickle
import numpy as np
import pandas as pd
import json
import os
import string
import sys
import csv
import ast
import time
from itertools import product, groupby
from functools import partial
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

sys.path.append("../")
from foldit_data import load_extend_data, make_series
from util import PatternLookup, SubClusters, SubclusterSeries, SubSeriesLookup
from check_models import load_sub_lookup
from pattern_extraction import combine_user_series, get_predicted_lookups, get_pattern_lookups, score_param, load_TICC_output, get_pattern_label

IGNORE_COLUMNS = ['energies', 'evol_lines', 'first_pdb', 'frontier_pdbs', 'frontier_tmscores', 'lines',
                  'pid', 'timestamps', 'uid', 'upload_rate', 'upload_ratio', 'deltas', 'relevant_sids']
IGNORE_SAMPLE_COLUMNS = ["uid", "pid", "uid_ext", "actions", "stop_idx"]


def perf(energy: float, pid: str, puz_metas: dict) -> float:
    return (min(0, energy - puz_metas[pid].energy_baseline) / (
        min(puz_metas[pid].pfront) - puz_metas[pid].energy_baseline))


def samples_from_row(row: pd.Series, resolution: int, puz_metas: dict) -> list:
    """
    Take a row with the data from one user on one puzzle and a resolution
    Return a list of lists where each element is a subset of the original data up to some point
    Subsets are generated at an interval of `resolution`
    First "subset" is the original data point
    Returned samples have fields: uid, pid, sample id, list of action counts, index where the sample stops, perf
    """
    samples = []
    samples.append([row.uid, row.pid, "", row.actions,
                   row.actions.size, perf(row.energies.min(), row.pid, puz_metas)])
    for i, stop in enumerate(range(resolution, row.actions.size, resolution), 1):
        samples.append([row.uid, row.pid, f"{i}", row.actions[:stop], stop,
                       perf(row.energies[:stop].min(), row.pid, puz_metas)])
    return samples


def compute_pattern_actions(r: pd.Series, k: int, cid: int, sub_k: int,
                            cluster_lookup: Dict[int, np.ndarray],
                            subcluster_series: SubclusterSeries,
                            puz_idx_lookup:  Dict[Tuple[str, str], Tuple[int, int]]) -> pd.Series:
    start = puz_idx_lookup[(r.uid, r.pid)][0]
    if sub_k == 0:
        puz_cs = cluster_lookup[k][slice(start, start + r.stop_idx)]
        return pd.Series(sum(r.actions[puz_cs == cid]), index=["pattern_{}_actions".format(cid)])
    else:
        puz_cs = subcluster_series.series[slice(start, start + r.stop_idx)]
        return pd.Series([sum(r.actions[puz_cs == scid]) for scid in subcluster_series.labels],
                         index=["pattern_{}_actions".format(l) for l in subcluster_series.labels])


def generate_sampled_users(data: pd.DataFrame, resolution: int, puz_metas: dict) -> pd.DataFrame:
    return pd.DataFrame([new_row for _, r in data.iterrows() for new_row in samples_from_row(r, resolution, puz_metas)],
                        columns=["uid", "pid", "uid_ext", "actions", "stop_idx", "perf"])

def score_candidate(X, y, cv, groups):
    gbr = GradientBoostingRegressor(loss="huber", alpha=0.95, n_estimators=1500,
                                    n_iter_no_change=100, max_depth=5, subsample=0.5)
    return np.mean(cross_val_score(gbr, X, y, groups=groups, cv=cv))


def generate_candidates(k, sub_ks, pattern_lookups, cids, cur_sub_k_idx):
    return [c for c in product(*[list(product([cid], sub_ks[cid][:cur_sub_k_idx[cid] + 1]))
                                 for cid in cids])
            if all(sub_k in pattern_lookups[k][cid] for cid, sub_k in c)]


def find_best_predictive_model(user: str, data: pd.DataFrame,
                               puz_idx_lookup:  Dict[Tuple[str, str], Tuple[int, int]],
                               pattern_lookups: Dict[str, Dict[int, PatternLookup]],
                               cluster_lookup: Dict[str, Dict[int, np.ndarray]],
                               subseries_lookups: Dict[str, Dict[int, Dict[int, SubSeriesLookup]]],
                               subclusters: Dict[str, SubClusters]) -> Tuple:
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

    cv = GroupKFold(10)
    scores_lookup = {}
    for k in pattern_lookups[user]:
        subcluster_series_lookup = {} # (k, (cid, sub_k)) -> SubclusterSeries
        action_counts = {} # (k, (cid, sub_k)) -> pd.DataFrame (single column for base pattern, column per subpattern otherwise)
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
                        # print("candidate", candidate)
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

                        features = pd.concat([data.drop(IGNORE_SAMPLE_COLUMNS, axis=1)] +
                                             [action_counts[(k, (cid, sub_k))] for (cid, sub_k) in candidate],
                                             axis=1)
                        X = features.drop(["perf"], axis=1).values
                        y = features["perf"].values.ravel()
                        pooled.append((candidate, pool.apply_async(score_candidate, (X, y, cv, data.uid))))
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
                        cur_sub_k_idx[cid] = max(cur_sub_k_idx[cid] - 1, 0)  # need to avoid downgrading a pattern with no subpatterns from index 0 to -1
                cur_sub_k_idx = {cid: (idx + 1 if cid in active_cids else idx) for cid, idx in cur_sub_k_idx.items()}
                active_cids = [cid for cid in active_cids if len(sub_ks[cid]) > cur_sub_k_idx[cid]]

            print("done\n\n\n")
            scores_lookup[k] = scores

    best_k = best_candidate = None
    best_score = 0
    for k, scores in scores_lookup.items():
        for candidate, score in scores.items():
            if score > best_score:
                best_score = score
                best_k = k
                best_candidate = candidate
    return best_k, best_candidate, scores_lookup


def load_raw_data(config):
    start_time = time.perf_counter()

    print("loading raw data", end="...")
    # pids = ["2003433", "2003642", "2003195", "2003313", "2003287", "2002475", "2002294", "2002196", "2002141", "2002110"]
    soln_lookup = {}
    parent_lookup = {}
    child_lookup = {}
    data, puz_metas = load_extend_data(config["train_pids"] + config["test_pids"],
                                       soln_lookup, parent_lookup, child_lookup, False, 600)

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
            v['solo_perf'] = float(
                v['solo_perf']) if v['solo_perf'] != "" else np.nan
    user_meta_lookup = {uid: list(metas) for uid, metas in groupby(
        sorted(user_metas.values(), key=lambda m: m['uid']), lambda m: m['uid'])}

    with open("../data/foldit/puz_metadata_v4.csv") as fp:
        puz_infos = {r['pid']: {'start':     int(r['start']),
                                'end':       int(r['end']),
                                'baseline':  float(r['baseline']),
                                'best':      float(r['best']),
                                'best_solo': float(r['best_solo'])
                                } for r in csv.DictReader(fp)}
    print(f"done (took {time.perf_counter() - start_time:0.2f} seconds)")
    return data, puz_metas, user_meta_lookup, puz_infos


def load_model_data(config, model_dir: str, data: pd.DataFrame):
    """
    Create time series from data
    Load TICC models
    Predict patterns on full data if pickled data files not available
        (and save pickled results)
    Compute list of action counts for each data point
    """
    print("loading model data", end="...")
    start_time = time.perf_counter()
    noise = np.loadtxt(f"{model_dir}/noise_values.txt")
    puz_idx_lookup, series_lookup, _ = make_series(data, noise=noise)
    if "user" not in config:
        idx_lookup, all_series = combine_user_series(series_lookup, noise)
        puz_idx_lookup = {(uid, pid): (s + idx_lookup[uid][0], e + idx_lookup[uid][0])
                          for (uid, pid), (s, e) in puz_idx_lookup.items()}
        series_lookup["all"] = all_series

    krange = config["krange"]
    users = [d for d in os.listdir(model_dir) if
             os.path.isdir(f"{model_dir}/{d}") and
             os.path.exists(f"{model_dir}/{d}/subpatterns")] if "user" in config else ["all"]

    _, mrf_lookup, model_lookup, _ = load_TICC_output(
        model_dir, users, krange)
    dummy_subseries_lookup = {user: {int(d.strip("k")): [int(c.strip("cid")) for c in os.listdir(f"{model_dir}/{user}/subpatterns/{d}")]
                                     for d in os.listdir(f"{model_dir}/{user}/subpatterns") if d.startswith("k")} for user in users}
    sub_lookup = load_sub_lookup(model_dir, users, dummy_subseries_lookup, config["sub_krange"])

    if os.path.exists(f"{model_dir}/eval/cluster_lookup.pickle"):
        with open(f"{model_dir}/eval/cluster_lookup.pickle", "rb") as fp:
            cluster_lookup = pickle.load(fp)
        with open(f"{model_dir}/eval/subseries_lookup.pickle", "rb") as fp:
            subseries_lookups = pickle.load(fp)
        with open(f"{model_dir}/eval/sub_clusters.pickle", "rb") as fp:
            subclusters = pickle.load(fp)
        with open(f"{model_dir}/eval/pattern_lookup.pickle", "rb") as fp:
            pattern_lookups = pickle.load(fp)
    else:
        # predict patterns on full data for all candidate models
        # print("\ngenerating patterns on full data", end="...")
        print("\ngenerating patterns on full data")
        gen_start_time = time.perf_counter()
        cluster_lookup = {}
        subseries_lookups = {}
        subclusters = {}
        pattern_lookups = {}
        with ProcessPoolExecutor(len(krange)) as pool:
            for user in users:
                pooled = []
                for k in krange:
                    print(f"submitting {user}:{k}")
                    pooled.append((k, pool.submit(get_predicted_lookups, series_lookup[user], k, model_lookup[user],
                                                  sub_lookup.models[user], mrf_lookup[user], puz_idx_lookup, noise, user)))
                cl = {}
                sl = {}
                scs = {}
                for k, task in pooled:
                    clusters, subseries, subcs = task.result()
                    cl[k] = clusters
                    sl[k] = subseries
                    scs[k] = subcs
                cluster_lookup[user] = cl
                subseries_lookups[user] = sl
                subclusters[user] = scs
                pattern_lookups[user] = get_pattern_lookups(krange, subclusters[user], sub_lookup.mrfs[user], subseries_lookups[user],
                                                            cluster_lookup[user], mrf_lookup[user], puz_idx_lookup, user)
        os.makedirs(f"{model_dir}/eval", exist_ok=True)
        with open(f"{model_dir}/eval/cluster_lookup.pickle", "wb") as fp:
            pickle.dump(cluster_lookup, fp)
        with open(f"{model_dir}/eval/subseries_lookup.pickle", "wb") as fp:
            pickle.dump(subseries_lookups, fp)
        with open(f"{model_dir}/eval/sub_clusters.pickle", "wb") as fp:
            pickle.dump(subclusters, fp)
        with open(f"{model_dir}/eval/pattern_lookup.pickle", "wb") as fp:
            pickle.dump(pattern_lookups, fp)
        print(f"done (took {time.perf_counter() - gen_start_time:0.2f} seconds)")

    # compute list of action counts for each data point
    rows = []
    for _, r in data.iterrows():
        if r.relevant_sids is None or (r.uid, r.pid) not in puz_idx_lookup:
            continue
        deltas = sorted(
            [d for d in r.deltas if d.sid in r.relevant_sids], key=lambda x: x.timestamp)
        actions = np.array([sum(d.action_diff.values()) for d in deltas])
        if actions.sum() == 0:
            # logging.debug("SKIPPING {} {}, no actions recorded".format(r.uid, r.pid))
            continue
        rows.append({'uid': r.uid, 'pid': r.pid, "actions": actions})
    action_data = data.merge(pd.DataFrame(data=rows), on=["pid", "uid"])

    print(f"done (took {time.perf_counter() - start_time:0.2f} seconds)")
    return action_data, puz_idx_lookup, users, cluster_lookup, subseries_lookups, subclusters, pattern_lookups


def historical_data(data: pd.DataFrame, user_meta_lookup: dict[str, list[Dict[str, str]]],
                    puz_infos: dict[str, dict[str, float]], history_size: int) -> pd.DataFrame:
    data["experience"] = data.apply(lambda r: min(len([x for x in user_meta_lookup[r.uid]
                                                   if puz_infos[x['pid']]["end"] < puz_infos[r.pid]["start"]]), history_size), axis=1)
    # get the users most recent `history_size` puzzles
    data["median_prior_perf"] = data.apply(lambda r: np.median([float(x['perf']) for x in
                                                                sorted(user_meta_lookup[r.uid], reverse=True,
                                                                       key=lambda m: puz_infos[m["pid"]]["start"])[:history_size]]), axis=1)
    data.median_prior_perf.fillna(data.median_prior_perf.median(), inplace=True)
    return data


def model_selection(config, model_dir: str, user: str, resolution: int, data, puz_idx_lookup, pattern_lookups, cluster_lookup, subseries_lookups, subclusters):
    start_time = time.perf_counter()
    print(f"finding most predictive model for {user}{resolution}", end="...")
    best_k, best_candidate, scores = find_best_predictive_model(user, data, puz_idx_lookup, pattern_lookups,
                                                                cluster_lookup, subseries_lookups, subclusters)
    
    for k in config["krange"]:
        with open(f"{model_dir}/eval/{user}{resolution}_k{k}_scores.pickle", "wb") as fp:
            pickle.dump(scores[k], fp)
    
    with open(f"{model_dir}/eval/{user}{resolution}_best_model.txt", 'w') as fp:
        fp.write(str((best_k, best_candidate)) + "\n")

    print(f"done (took {time.perf_counter() - start_time:0.2f} seconds)")
    return best_k, best_candidate, scores


def evaluate(model_dir: str, user: str, resolution: str, tag: str,
             features: pd.DataFrame, baseline_features: List[str],
             pattern_features: List[str], historical_features: List[str],
             pool: Pool):
    seed = 13*17*31
    models = {"gbr": GradientBoostingRegressor}
            #   "hist": HistGradientBoostingRegressor} # no HistGradientBoost for now, as it does not support feature importance
                                                    # this breaks RFECV in addition to being less useful
    model_params = {"gbr": {"random_state": [seed], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7],
                            "n_estimators": [1500], "n_iter_no_change": [100], "subsample": [0.5, 1.0]}}
    model_params["gbr"] = [model_params["gbr"], model_params["gbr"].copy()]

    model_params["gbr"][0]["loss"] = ["huber"]
    model_params["gbr"][0]["alpha"] = [0.95, 0.99]
    model_params["gbr"][1]["loss"] = ["ls"]

    cv = GroupKFold(10)
    print("fitting baseline")
    start_time = time.perf_counter()
    scores = {}
    X = features[baseline_features + historical_features].values
    y = features["perf"].values.ravel()
    for model_label, model in models.items():
        evals = []
        for param in ParameterGrid(model_params[model_label]):
            evals.append((pool.apply_async(score_param, (param, model, X, y, cv, features.uid)), param))
        print("{}: {} scores sent to pool, collecting results".format(model_label, len(evals)))
        scores[model_label] = [(x.get(), param) for x, param in evals]
    baseline_scores = scores
    print("baseline")
    # [0] to get return value of score_param, [0] to get the CV accuracy
    for model_label in models:
        print(f"{model_label}: {max(scores[model_label], key=lambda x: x[0][0])}")
    print(f"(took {time.perf_counter() - start_time:0.2f} seconds)")
    start_time = time.perf_counter()

    model_scores = {}
    for ftype in ["actions", "use"]:
        print("fitting pattern {} models".format(ftype))
        candidate_features = ["{}_{}".format(f, ftype) for f in pattern_features]
        if ftype == "use":
            candidate_features.append("action_count_relevant")
        scores = {}
        X = features[candidate_features + historical_features].values
        y = features["perf"].values.ravel()
        for model_label, model in models.items():
            evals = []
            for param in ParameterGrid(model_params[model_label]):
                evals.append((pool.apply_async(score_param, (param, model, X, y, cv, features.uid)), param))
            print("{}: {} scores sent to pool, collecting results".format(model_label, len(evals)))
            scores[model_label] = [(x.get(), param) for x, param in evals]
        model_scores[ftype] = scores
        print("best {} model".format(ftype))
        # [0] to get return value of score_param, [0] to get the CV accuracy
        for model_label in models:
            print(f"{model_label}: {max(scores[model_label], key=lambda x: x[0][0])}")
        print(f"(took {time.perf_counter() - start_time:0.2f} seconds)")
        start_time = time.perf_counter()

    with open(f"{model_dir}/eval/{user}{resolution}_baseline_scores{tag}.pickle", 'wb') as fp:
        pickle.dump(baseline_scores, fp)
    with open(f"{model_dir}/eval/{user}{resolution}_model_scores{tag}.pickle", 'wb') as fp:
        pickle.dump(model_scores, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='pattern_model_evaluation.py')
    parser.add_argument("model_dir")
    parser.add_argument("config")
    args = parser.parse_args()
    assert os.path.exists(args.model_dir)
    assert os.path.exists(args.config)
    with open(args.config) as fp:
        config = json.load(fp)
    
    print("evaluating model at", args.model_dir)

    data, puz_metas, user_meta_lookup, puz_infos = load_raw_data(config)
    data, puz_idx_lookup, users, cluster_lookup, subseries_lookups, subclusters, pattern_lookups = load_model_data(config, args.model_dir, data)
    
    datasets = {"": data}
    if "resolution" in config:
        for resolution in config["resolution"]:
            datasets[str(resolution)] = generate_sampled_users(data, resolution, puz_metas)

    for resolution, dataset in datasets.items():
        for user in users:
            if os.path.exists(f"{args.model_dir}/eval/{user}{resolution}_best_model.txt"):
                with open(f"{args.model_dir}/eval/{user}{resolution}_best_model.txt") as fp:
                    best_k, best_candidate = ast.literal_eval(fp.read())
            else:
                best_k, best_candidate, scores = model_selection(config, args.model_dir, user, resolution,
                                                                 dataset[dataset.apply(lambda r: r.pid in config["train_pids"], axis=1)],
                                                                 puz_idx_lookup, pattern_lookups, cluster_lookup, subseries_lookups,
                                                                 subclusters)
            print(f"selected model: {best_k} {best_candidate}")

            # compute action counts for selected model on full data
            subcluster_series_lookup = {}
            action_counts = {}
            for (cid, sub_k) in best_candidate:
                if sub_k != 0:
                    all_subclusters = cluster_lookup[user][best_k].astype(np.str)
                    labels = ["{}{}".format(cid, string.ascii_uppercase[x]) for x in range(sub_k)]
                    cs = subclusters[user][best_k][cid][sub_k]
                    for (_, _, start_idx), (s, e) in subseries_lookups[user][best_k][cid].idx_lookup.items():
                        all_subclusters[start_idx: start_idx + (min(e, len(cs)) - s)] = [labels[c] for c in cs[s:e]]
                    subcluster_series_lookup[(best_k, (cid, sub_k))] = SubclusterSeries(labels, all_subclusters)

                f = partial(compute_pattern_actions, k=best_k, cid=cid, sub_k=sub_k,
                            cluster_lookup=cluster_lookup[user],
                            subcluster_series=subcluster_series_lookup.get((best_k, (cid, sub_k)), None),
                            puz_idx_lookup=puz_idx_lookup)
                action_counts[(best_k, (cid, sub_k))] = dataset.apply(f, axis=1)

            ps = sum([[(get_pattern_label(p, cid, sub_k), p) for p in pattern_lookups[user][best_k][cid][sub_k]] for cid, sub_k in best_candidate], [])
            ps_uid_pid = {tag: sorted(xs) for tag, xs in groupby(sorted(ps, key=lambda p: (p[1].uid, p[1].pid)), lambda p: (p[1].uid, p[1].pid))}
            pattern_use_lookup = {tag: {pt for pt, _ in xs} for tag, xs in ps_uid_pid.items()}
            pts = {pt for pt, p in ps}

            # collect pattern features using selected model
            results = pd.concat([dataset] + [action_counts[(best_k, (cid, sub_k))] for (cid, sub_k) in best_candidate], axis=1)

            pattern_features = ["pattern_{}".format(pt) for pt in pts]

            acc = []
            for (uid, pid), use in pattern_use_lookup.items():
                r = {"uid": uid, "pid": pid}
                for pt in pts:
                    r["pattern_"+ pt+"_use"] = 1 if pt in use else 0
                acc.append(r)
            results = results.merge(pd.DataFrame(data=acc), on=["uid", "pid"])
            results["action_count_relevant"] = results.apply(lambda r: r.actions.sum(), axis=1)


            # find best model, compare to baseline

            tuning_pids = config["train_pids"] + config["test_pids"][:len(config["test_pids"]) // 3]
            features = results[results.apply(lambda r: r.pid in tuning_pids, axis=1)]
            baseline_features = ["action_count_relevant"]
            historical_features = ["median_prior_perf"]

            with Pool(25, maxtasksperchild=4) as pool:
                print("No-history models")
                evaluate(args.model_dir, user, resolution, "", features, baseline_features, pattern_features, [], pool)

                for history_size in range(1, 16):
                    print(f"History = {history_size} models")
                    historical_data(results, user_meta_lookup, puz_infos, history_size)
                    features = results[results.apply(lambda r: r.pid in tuning_pids, axis=1)]
                    evaluate(args.model_dir, user, resolution, f"_hist{history_size}",
                             features, baseline_features, pattern_features, historical_features, pool)

                # with all available history
                print(f"All history models")
                historical_features = ["experience", "median_prior_perf"]
                historical_data(results, user_meta_lookup, puz_infos, 1000)
                features = results[results.apply(lambda r: r.pid in tuning_pids, axis=1)]
                evaluate(args.model_dir, user, resolution, f"_histAll",
                         features, baseline_features, pattern_features, historical_features, pool)
