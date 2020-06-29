from pattern_extraction import *
from check_models import load_sub_lookup, predict_from_saved_model
from util import get_pattern_label
import argparse
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, ShuffleSplit, ParameterGrid, GroupShuffleSplit
from sklearn.feature_selection import RFECV
import pygam
import pickle
import numpy as np
import pandas as pd
import logging
import json
import sys
import os
import string
import re
import matplotlib
from multiprocessing import Pool
from statsmodels import robust
from collections import Counter
from copy import deepcopy
from operator import itemgetter
matplotlib.use("Agg")


def compute_pattern_times(k: int, subs: tuple, data: pd.DataFrame, cluster_lookup: Dict[int, np.ndarray],
                          mrf_lookup: Dict[int, Dict[int, np.ndarray]], puz_idx_lookup: dict) -> pd.DataFrame:
    cluster_times = []
    print("computing pattern times")
    for _, r in data.iterrows():
        if r.relevant_sids is None or (r.uid, r.pid) not in puz_idx_lookup:
            continue
        deltas = sorted([d for d in r.deltas if d.sid in r.relevant_sids], key=lambda x: x.timestamp)
        ts = np.array([d.timestamp for d in deltas])
        actions = np.array([sum(d.action_diff.values()) for d in deltas])
        if actions.sum() == 0:
            print("SKIPPING {} {}, no actions recorded".format(r.uid, r.pid))
            continue
        result_dict = {'uid': r.uid, 'pid': r.pid}
        valid = True
        puz_cs = cluster_lookup[k][slice(*puz_idx_lookup[(r.uid, r.pid)])]
        if len(ts) != len(puz_cs):
            print("SKIPPING {} {}, k={}, mismatch between number of timestamps and cluster data".format(r.uid, r.pid, k))
            valid = False
            continue
        for ci, sub_k in subs:
            if sub_k == 0:
                #result_dict["pattern_{}_time".format(ci)] = time_played(ts[puz_cs == ci])
                #result_dict["pattern_{}_ratio".format(ci)] = result_dict["pattern_{}_time".format(ci)] / r.relevant_time
                result_dict["pattern_{}_action".format(ci)] = sum(actions[puz_cs == ci])
                #result_dict["pattern_{}_action_ratio".format(ci)] = result_dict["pattern_{}_action".format(ci)] / actions.sum()
        if valid:
            cluster_times.append(result_dict)
    return data.merge(pd.DataFrame(data=cluster_times), on=['pid', 'uid'])


def compute_subpattern_times(k: int, subs: tuple, data: pd.DataFrame, cluster_lookup: dict, subclusters: dict,
                             subseries_lookup: dict, puz_idx_lookup: dict) -> pd.DataFrame:
    results = {}
    print("generating timestamps")
    for _, r in data.iterrows():
        if r.relevant_sids is None or (r.uid, r.pid) not in puz_idx_lookup:
            continue
        deltas = sorted([d for d in r.deltas if d.sid in r.relevant_sids], key=lambda x: x.timestamp)
        ts = np.array([d.timestamp for d in deltas])
        actions = np.array([sum(d.action_diff.values()) for d in deltas])
        if actions.sum() == 0:
            print("SKIPPING {} {}, no actions recorded".format(r.uid, r.pid))
            continue
        results[(r.uid, r.pid)] = {"times": {'uid': r.uid, 'pid': r.pid}, "ts": ts, "actions": actions, "valid": True}
    print("computing subpattern times")
    all_clusters = cluster_lookup[k]
    for cid, sub_k in subs:
        if sub_k == 0:
            continue
        all_subclusters = all_clusters.astype(np.str)
        labels = ["{}{}".format(cid, string.ascii_uppercase[x]) for x in range(sub_k)]
        cs = subclusters[k][cid][sub_k]
        for (_, _, start_idx), (s, e) in subseries_lookup[k][cid]['idx_lookup'].items():
            all_subclusters[start_idx: start_idx + (min(e, len(cs)) - s)] = [labels[c] for c in cs[s:e]]
        for uid, pid in results:
            puz_cs = all_subclusters[slice(*puz_idx_lookup[(uid, pid)])]
            ts = results[(uid, pid)]["ts"]
            actions = results[(uid, pid)]["actions"]
            if len(ts) != len(puz_cs):
                results[(uid, pid)]["valid"] = False
                continue
            for scid in labels:
                #results[(uid, pid)]["times"]["subpattern_{}_time".format(scid)] = time_played(ts[puz_cs == scid])
                #results[(uid, pid)]["times"]["subpattern_{}_ratio".format(scid)] = results[(uid, pid)]["times"]["subpattern_{}_time".format(scid)] / r.relevant_time
                results[(uid, pid)]["times"]["pattern_{}_action".format(scid)] = sum(actions[puz_cs == scid])
                #results[(uid, pid)]["times"]["subpattern_{}_action_ratio".format(scid)] = results[(uid, pid)]["times"]["subpattern_{}_action".format(scid)] / actions.sum()
    subcluster_times = [v["times"] for v in results.values() if v["valid"]]
    return data.merge(pd.DataFrame(data=subcluster_times), on=['pid', 'uid'])


def get_predicted_lookups(all_series, krange, model_lookup, sub_models, mrf_lookup, puz_idx_lookup, noise):
    cluster_lookup = {}
    sub_clusters = {}
    subseries_lookup = {}
    for k in krange:
        print("predicting k =", k)
        cluster_lookup[k] = predict_from_saved_model(all_series, model_lookup[k])
        patterns = get_patterns(mrf_lookup[k], cluster_lookup[k], puz_idx_lookup)
        subseries_lookup[k] = make_subseries_lookup(k, patterns, mrf_lookup[k], all_series, noise)
        sub_cs = {}        
        for cid in sub_models[k]:
            if cid in subseries_lookup[k]:
                sub_cs[cid] = {}
                for sub_k in sub_models[k][cid]:
                    print("    cid =", cid, "({})".format(sub_k))
                    sub_cs[cid][sub_k] = predict_from_saved_model(subseries_lookup[k][cid]["series"],
                                                                  sub_models[k][cid][sub_k])
        sub_clusters[k] = sub_cs

    return cluster_lookup, subseries_lookup, sub_clusters


def make_selection_lookups(all_series, pattern_lookup, subseries_lookup, sub_clusters, sub_mrfs):
    dispersions = {}
    modes = {}
    print("computing selection criteria")
    for k in pattern_lookup:
        print("k =", k)
        for cid in {p.cid for p in pattern_lookup[k]["base"]}:
            print("    cid", cid)
            ss = [all_series[p.start_idx:p.end_idx] for p in pattern_lookup[k]["base"] if p.cid == cid]
            ubiqs = np.array([[len([x for x in s[:, i] if x > 0]) / len(s) for i in range(s.shape[1])] for s in ss])

            dispersions[(k, cid, 0)] = np.mean(robust.mad(ubiqs, axis=0))
            modes[(k, cid, 0)] = stats.mode(np.round(ubiqs, 1)).mode

            if cid not in sub_clusters[k]:
                continue

            idx_lookup = subseries_lookup[k][cid]["idx_lookup"]
            ser = subseries_lookup[k][cid]["series"]
            for sub_k in sub_clusters[k][cid]:
                if not any(is_null_cluster(mrf) for mrf in sub_mrfs[k][cid][sub_k].values()) or len(idx_lookup) <= sub_k:
                    continue
                ps = pattern_lookup[k][cid][sub_k]
                for sub_cid in {p.cid for p in ps}:
                    ss = [ser[p.start_idx:p.end_idx] for p in ps if p.cid == sub_cid]
                    ubiqs = np.array([[len([x for x in s[:, i] if x > 0]) / len(s) for i in range(s.shape[1])] for s in ss])
                    dispersions[(k, cid, sub_k, sub_cid)] = np.mean(robust.mad(ubiqs, axis=0))
                    modes[(k, cid, sub_k, sub_cid)] = stats.mode(np.round(ubiqs, 1)).mode

    dispersion_lookup = {tag: [x[1] for x in sorted(xs)] for tag, xs in groupby(sorted(dispersions.items()), lambda x: x[0][:3])}
    mode_lookup = {tag: [x[1] for x in sorted(xs)] for tag, xs in groupby(sorted(modes.items()), lambda x: x[0][:3])}

    return dispersion_lookup, mode_lookup


def dispersion_score(k, candidate, dispersion_lookup, pattern_lookup):
    sub_weights = [[len([p for p in pattern_lookup[k][cid][sub_k] if p.cid == sub_cid]) for sub_cid in 
                    sorted({p.cid for p in pattern_lookup[k][cid][sub_k]})] for cid, sub_k in candidate]
    return np.average([np.average(dispersion_lookup[(k, cid, sub_k)], weights=ws) for ws, (cid, sub_k) in zip(sub_weights, candidate)],
                      weights=[len(pattern_lookup[k][cid][sub_k]) for cid, sub_k in candidate])


def mode_score(k, candidate, mode_lookup):
    ms = sum((mode_lookup[(k, cid, sub_k)] for cid, sub_k in candidate), [])
    return np.mean([min([abs(x - m).sum() for x in ms if x is not m]) for m in ms])



def find_best_dispersion_model(all_series, pattern_lookup, subseries_lookup, sub_clusters):
    """
    for each pattern p
        compute signal ubiquity for each instance of p (for each signal i, compute fraction of instance's duration where i is active)
        for each signal i, compute the median absolute deviation among ubiquities 
        (i.e., to what degree do different instances of p vary in terms of how much they involve i)
        take the mean (i.e, averaged across all signals, how much do instances of p vary), call this DISPERSION
        also compute the modal ubiquity for each signal
    ^^^ this is done in make_selection_lookups()
    for each pattern identified by the initial extraction, we have to choose a recursive model (or no recursive model)
    we make each of these choices to minimize the mean dispersion of the resulting patterns
    hence, for each value of k used to perform an initial extraction, we have a candidate model where the choices of recursive models minimize mean dispsersion
    each of these candidate models are given an overall dispersion score by performing a weighted average of the dispsersions of all patterns in the model
        weighted according to the number of instances of that pattern--we care more that a very common pattern is cohesive than a very rare one
    if we were to use only this dispersion score to select the final model,
    we would have a strong bias toward models that extract many patterns that occur just once or a handful of times, as these patterns will naturally be very cohesive under our definition
    to mitigate this bias, we take into account the distinctiveness of a model's patterns in the final selection step
    we measure distinctiveness as the mean nearest-neighbor distance in terms of modal ubiquity across all patterns
    we rank candidate models by dispersion (smaller is better) and nearest-neighbor distance (larger is better)
    we sum the two ranks for each candidate and the best candidate is that which has the smallest sum
    """
    dispersion_lookup, mode_lookup = make_selection_lookups(all_series, pattern_lookup, subseries_lookup, sub_clusters)
    disp_scores = {}
    for k in pattern_lookup:
        print("k =", k)
        candidate = []
        for cid in {p.cid for p in pattern_lookup[k]["base"]}:
            xs = [(cid, 0)]
            if cid not in sub_clusters[k]:
                candidate.append(xs[0])
                continue
            idx_lookup = subseries_lookup[k][cid]["idx_lookup"]
            for sub_k in sub_clusters[k][cid]:
                if (k, cid, sub_k) not in dispersion_lookup:
                    print("SKIPPING", k, cid, sub_k)
                    continue
                xs.append((cid, sub_k))
            candidate.append(min(xs, key=lambda x: np.mean(dispersion_lookup[(k, x[0], x[1])])))
        disp_scores[(k, tuple(candidate))] = dispersion_score(k, candidate, dispersion_lookup, pattern_lookup)
    disp_ranks = {kc: i for i, (kc, s) in enumerate(sorted(disp_scores.items(), key=itemgetter(1)))}
    mode_scores = {(k, c): mode_score(k, c, mode_lookup) for k, c in disp_scores}
    mode_ranks = {kc: i for i, (kc, s) in enumerate(sorted(mode_scores.items(), key=itemgetter(1), reverse=True))}

    # sort the keys to return the smaller k in case of ties
    return min(sorted(disp_ranks.keys()), key=lambda kc: disp_ranks[kc] + mode_ranks[kc])


def score_param(param, model, X, y, cv):
    # feature selection under these params
    selector = RFECV(model(**param), step=1, cv=cv)
    selector.fit(X, y)
    X_sel = selector.transform(X)
    # score for these params is CV score fitting on X_sel
    return np.mean(cross_val_score(model(**param), X_sel, y, cv=cv)), selector.get_support()


def load_eval_model(model_dir):
    with open(model_dir + "/eval/cluster_lookup.pickle", "rb") as fp:
        cluster_lookup = pickle.load(fp)
    with open(model_dir + "/eval/subseries_lookup.pickle", "rb") as fp:
        subseries_lookup = pickle.load(fp)
    with open(model_dir + "/eval/sub_clusters.pickle", "rb") as fp:
        sub_clusters = pickle.load(fp)
    with open(model_dir + "/eval/pattern_lookup.pickle", "rb") as fp:
        pattern_lookup = pickle.load(fp)
    best_k, best_subs = eval(open(model_dir + "/eval/best_model.txt").read())
    return cluster_lookup, subseries_lookup, sub_clusters, pattern_lookup, best_k, best_subs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='pattern_model_evaluation.py')
    parser.add_argument("model_dirs", nargs="+")
    args = parser.parse_args()
    assert all(os.path.exists(model_dir) for model_dir in args.model_dirs)

    print("loading data", end="...")
    pids = ["2003433", "2003642", "2003195", "2003313", "2003287", "2002475", "2002294", "2002196", "2002141", "2002110"]
    soln_lookup = {}
    parent_lookup = {}
    child_lookup = {}
    data, puz_metas = load_extend_data(pids, soln_lookup, parent_lookup, child_lookup, False, 600)

    with open("data/user_metadata_v4.csv") as fp:
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

    with open("data/puz_metadata_v4.csv") as fp:
        puz_infos = {r['pid']: {'start':     int(r['start']),
                                'end':       int(r['end']),
                                'baseline':  float(r['baseline']),
                                'best':      float(r['best']),
                                'best_solo': float(r['best_solo'])
                               } for r in csv.DictReader(fp)}
    print("done")

    for model_dir in args.model_dirs:
        print("evaluating model at", model_dir)
        noise = np.loadtxt(model_dir + "/noise_values.txt")
        puz_idx_lookup, series_lookup, _ = make_series(data, noise=noise)
        idx_lookup, all_series = combine_user_series(series_lookup, noise)
        puz_idx_lookup = {(uid, pid): (s + idx_lookup[uid][0], e + idx_lookup[uid][0])
                          for (uid, pid), (s, e) in puz_idx_lookup.items()}

        with open(model_dir + "/config.json") as fp:
            config = json.load(fp)
        krange = config["krange"]
        _, mrf_lookup, model_lookup, _ = load_TICC_output(model_dir, ["all"], krange)
        dummy_subseries_lookup = {int(d.strip("k")): [int(c.strip("cid")) for c in os.listdir(model_dir + "/all/subpatterns/" + d)]
                                  for d in os.listdir(model_dir + "/all/subpatterns") if d.startswith("k")}
        sub_lookup = load_sub_lookup(model_dir + "/all", dummy_subseries_lookup, [3, 6, 9, 12])

        # predict patterns on full data for all candidate models
        if os.path.exists(model_dir + "/eval/cluster_lookup.pickle"):
            with open(model_dir + "/eval/cluster_lookup.pickle", "rb") as fp:
                cluster_lookup = pickle.load(fp)
            with open(model_dir + "/eval/subseries_lookup.pickle", "rb") as fp:
                subseries_lookup = pickle.load(fp)
            with open(model_dir + "/eval/sub_clusters.pickle", "rb") as fp:
                sub_clusters = pickle.load(fp)
            with open(model_dir + "/eval/pattern_lookup.pickle", "rb") as fp:
                pattern_lookup = pickle.load(fp)
        else:
            cluster_lookup, subseries_lookup, sub_clusters = get_predicted_lookups(all_series, krange, model_lookup["all"],
                                                                                   sub_lookup["models"], mrf_lookup["all"],
                                                                                   puz_idx_lookup, noise)
            pattern_lookup = get_patterns_lookup(krange, sub_clusters, sub_lookup["mrfs"], subseries_lookup, cluster_lookup,
                                                 mrf_lookup["all"], puz_idx_lookup)
            os.makedirs(model_dir + "/eval", exist_ok=True)
            with open(model_dir + "/eval/cluster_lookup.pickle", "wb") as fp:
                pickle.dump(cluster_lookup, fp)
            with open(model_dir + "/eval/subseries_lookup.pickle", "wb") as fp:
                pickle.dump(subseries_lookup, fp)
            with open(model_dir + "/eval/sub_clusters.pickle", "wb") as fp:
                pickle.dump(sub_clusters, fp)
            with open(model_dir + "/eval/pattern_lookup.pickle", "wb") as fp:
                pickle.dump(pattern_lookup, fp)

        # select model by minimum dispersion for predicted patterns
        best_k, best_subs = find_best_dispersion_model(all_series, pattern_lookup, subseries_lookup, sub_clusters, sub_lookup["mrfs"])
        with open(model_dir + "/eval/best_model.txt", 'w') as fp:
            fp.write(str((best_k, best_subs)) + "\n")
        print("selected model:", best_k, best_subs)

        ps = sum([[(get_pattern_label(p, cid, sub_k), p) for p in pattern_lookup[best_k][cid][sub_k]] for cid, sub_k in best_subs], [])
        ps_uid_pid = {tag: sorted(xs) for tag, xs in groupby(sorted(ps, key=lambda p: (p[1].uid, p[1].pid)), lambda p: (p[1].uid, p[1].pid))}
        pattern_use_lookup = {tag: {pt for pt, _ in xs} for tag, xs in ps_uid_pid.items()}
        pts = {pt for pt, p in ps}

        # compute pattern features using selected model
        results = compute_pattern_times(best_k, best_subs, data, cluster_lookup, mrf_lookup["all"], puz_idx_lookup)
        results = compute_subpattern_times(best_k, best_subs, results, cluster_lookup, sub_clusters, subseries_lookup,
                                           puz_idx_lookup)

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
        results["action_count_relevant"] = results.apply(lambda r: user_metas[(r.uid, r.pid)]["action_count_relevant"], axis=1)
        #results["action_count_best"] = results.apply(lambda r: user_metas[(r.uid, r.pid)]["action_count_best"], axis=1)
        #results["best_energy_time"] = results.apply(lambda r: user_metas[(r.uid, r.pid)]["best_energy_time"], axis=1)
        #results["action_rate_all"] = results.apply(lambda r: r.action_count_all / r.time, axis=1)
        #results["action_rate_relevant"] = results.apply(lambda r: r.action_count_relevant / r.relevant_time, axis=1)
        results["experience"] = results.apply(lambda r: len([x for x in user_meta_lookup[r.uid]
                                                            if puz_infos[x['pid']]["end"] < puz_infos[r.pid]["start"]]), axis=1)
        results["median_prior_perf"] = results.apply(lambda r: np.median([float(x['perf']) for x in user_meta_lookup[r.uid]
                                                                         if puz_infos[x['pid']]["end"] < puz_infos[r.pid]["start"]]), axis=1)
        results.median_prior_perf.fillna(results.median_prior_perf.median(), inplace=True)

        # find best model, compare to baseline
        ignore_columns = ['energies', 'evol_lines', 'first_pdb', 'frontier_pdbs', 'frontier_tmscores', 'lines', 'pid',
                          'timestamps', 'uid', 'upload_rate', 'upload_ratio', 'deltas', 'relevant_sids']
        features = results.drop(ignore_columns, axis=1)

        baseline_features = ["action_count_relevant", "median_prior_perf", "experience"]

        seed = 13*17*31
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

        with Pool(50, maxtasksperchild=4) as pool:
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
            for ftype in ["action", "use"]:
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

        with open(model_dir + "/eval/baseline_scores.pickle", 'wb') as fp:
            pickle.dump(baseline_scores, fp)
        with open(model_dir + "/eval/model_scores.pickle", 'wb') as fp:
            pickle.dump(model_scores, fp)
