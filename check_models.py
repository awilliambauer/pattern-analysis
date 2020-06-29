from pattern_extraction import *
import argparse
from sklearn import svm, linear_model, ensemble
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
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
from ticc.TICC_solver import TICC
from typing import Dict, Tuple, List
from itertools import combinations, groupby, chain
from util import category_lookup
import matplotlib
matplotlib.use("Agg")
SUBPATTERN_KRANGE = [5, 10]


def predict_from_saved_model(test_data: np.ndarray, saved_model: dict) -> np.ndarray:
    test_ticc = TICC(window_size=saved_model["window_size"], number_of_clusters=saved_model["number_of_clusters"], num_proc=1)
    test_ticc.num_blocks = saved_model["num_blocks"]
    test_ticc.switch_penalty = saved_model["switch_penalty"]
    test_ticc.trained_model = saved_model["trained_model"]
    cs = test_ticc.predict_clusters(test_data)
    test_ticc.pool.close()
    return cs


def compute_cluster_times(data: pd.DataFrame, cluster_lookup: Dict[int, np.ndarray],
                          mrf_lookup: Dict[int, Dict[int, np.ndarray]], puz_idx_lookup: dict) -> pd.DataFrame:
    cluster_times = []
    logging.debug("computing cluster times")
    for _, r in data.iterrows():
        if r.relevant_sids is None or (r.uid, r.pid) not in puz_idx_lookup:
            continue
        deltas = sorted([d for d in r.deltas if d.sid in r.relevant_sids], key=lambda x: x.timestamp)
        ts = np.array([d.timestamp for d in deltas])
        actions = np.array([sum(d.action_diff.values()) for d in deltas])
        if actions.sum() == 0:
            logging.debug("SKIPPING {} {}, no actions recorded".format(r.uid, r.pid))
            continue
        result_dict = {'uid': r.uid, 'pid': r.pid}
        valid = True
        for k in cluster_lookup:
            puz_cs = cluster_lookup[k][slice(*puz_idx_lookup[(r.uid, r.pid)])]
            if len(ts) != len(puz_cs):
                logging.debug("SKIPPING {} {}, k={}, mismatch between number of timestamps and cluster data".format(r.uid, r.pid, k))
                valid = False
                continue
            for ci in range(k):
                if not is_null_cluster(mrf_lookup[k][ci]):
                    result_dict["cluster_{}_time_k{}".format(ci, k)] = time_played(ts[puz_cs == ci])
                    result_dict["cluster_{}_ratio_k{}".format(ci, k)] = result_dict["cluster_{}_time_k{}".format(ci, k)] / r.relevant_time
                    result_dict["cluster_{}_action_k{}".format(ci, k)] = sum(actions[puz_cs == ci])
                    result_dict["cluster_{}_action_ratio_k{}".format(ci, k)] = result_dict["cluster_{}_action_k{}".format(ci, k)] / actions.sum()
        if valid:
            cluster_times.append(result_dict)
    return data.merge(pd.DataFrame(data=cluster_times), on=['pid', 'uid'])


def compute_subcluster_times(data: pd.DataFrame, cluster_lookup: dict, subclusters: dict,
                             subseries_lookup: dict, puz_idx_lookup: dict) -> pd.DataFrame:
    results = {}
    logging.debug("generating timestamps")
    for _, r in data.iterrows():
        if r.relevant_sids is None or (r.uid, r.pid) not in puz_idx_lookup:
            continue
        deltas = sorted([d for d in r.deltas if d.sid in r.relevant_sids], key=lambda x: x.timestamp)
        ts = np.array([d.timestamp for d in deltas])
        actions = np.array([sum(d.action_diff.values()) for d in deltas])
        if actions.sum() == 0:
            logging.debug("SKIPPING {} {}, no actions recorded".format(r.uid, r.pid))
            continue
        results[(r.uid, r.pid)] = {"times": {'uid': r.uid, 'pid': r.pid}, "ts": ts, "actions": actions, "valid": True}
    logging.debug("computing subcluster times")
    for k in cluster_lookup:
        all_clusters = cluster_lookup[k]
        for cid in subclusters[k]:
            for k2 in SUBPATTERN_KRANGE:
                if k2 not in subclusters[k][cid]:
                    continue
                all_subclusters = all_clusters.astype(np.str)
                labels = ["{}{}".format(cid, string.ascii_uppercase[x]) for x in range(k2)]
                cs = subclusters[k][cid][k2]
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
                        results[(uid, pid)]["times"]["sk{}_subcluster_{}_time_k{}".format(k2, scid, k)] = time_played(ts[puz_cs == scid])
                        results[(uid, pid)]["times"]["sk{}_subcluster_{}_ratio_k{}".format(k2, scid, k)] = results[(uid, pid)]["times"]["sk{}_subcluster_{}_time_k{}".format(k2, scid, k)] / r.relevant_time
                        results[(uid, pid)]["times"]["sk{}_subcluster_{}_action_k{}".format(k2, scid, k)] = sum(actions[puz_cs == scid])
                        results[(uid, pid)]["times"]["sk{}_subcluster_{}_action_ratio_k{}".format(k2, scid, k)] = results[(uid, pid)]["times"]["sk{}_subcluster_{}_action_k{}".format(k2, scid, k)] / actions.sum()
    subcluster_times = [v["times"] for v in results.values() if v["valid"]]
    return data.merge(pd.DataFrame(data=subcluster_times), on=['pid', 'uid'])


def load_sub_lookup(datapath: str, subseries_lookup: dict, sub_krange=[5, 10]) -> Dict[str, Dict[int, dict]]:
    sub_lookup = {"clusters": {}, "mrfs": {}, "models": {}, "bics": {}}
    for k in subseries_lookup:
        dp = "{}/subpatterns/k{}".format(datapath, k)
        cs, mrfs, ms, bs = load_TICC_output(dp, ["cid{}".format(cid) for cid in subseries_lookup[k]], sub_krange)
        sub_lookup["clusters"][k] = {int(k.replace("cid", "")): v for k, v in cs.items()}
        sub_lookup["mrfs"][k] = {int(k.replace("cid", "")): v for k, v in mrfs.items()}
        sub_lookup["models"][k] = {int(k.replace("cid", "")): v for k, v in ms.items()}
        sub_lookup["bics"][k] = {int(k.replace("cid", "")): v for k, v in bs.items()}
    return sub_lookup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='check_models.py')
    parser.add_argument("datapath")
    parser.add_argument("--new-ticc", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--no-test", action='store_true')
    args = parser.parse_args()
    seed =13*17*31

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.exists(args.datapath):
        logging.error("datapath {} does not exist".format(args.datapath))
        sys.exit(1)

    with open(args.datapath + "/config.json") as fp:
        config = json.load(fp)
    pids = config["pids"]
    test_pids = config["test_pids"]
    krange = config["krange"]

    soln_lookup = {}
    parent_lookup = {}
    child_lookup = {}
    data, puz_metas = load_extend_data(pids, soln_lookup, parent_lookup, child_lookup, config["evolver"], 600)
    test_data, test_metas = load_extend_data(test_pids, soln_lookup, parent_lookup, child_lookup, config["evolver"], 600)

    if args.new_ticc:
        logging.debug("Constructing time series")
        puz_idx_lookup, series_lookup, noise = make_series(data)
        num_features = next(x for x in series_lookup.values()).shape[1]
        idx_lookup, all_series = combine_user_series(series_lookup, noise)
        puz_idx_lookup = {(uid, pid): (s + idx_lookup[uid[:uid.index('e')] if 'e' in uid else uid][0],
                                       e + idx_lookup[uid[:uid.index('e')] if 'e' in uid else uid][0])
                          for (uid, pid), (s, e) in puz_idx_lookup.items()}
        np.savetxt(args.datapath + "/noise_values.txt", noise)
        np.savetxt(args.datapath + "/all_series.txt", all_series)
        with open(args.datapath + "/puz_idx_lookup.pickle", 'wb') as fp:
            pickle.dump(puz_idx_lookup, fp)
        with open(args.datapath + "/idx_lookup.pickle", 'wb') as fp:
            pickle.dump(idx_lookup, fp)

        run_TICC({"all": all_series}, args.datapath, krange)
        cluster_lookup, mrf_lookup, model_lookup, bic_lookup = load_TICC_output(args.datapath, ["all"], krange)
        subseries_lookup = {}
        for k in krange:
            patterns = get_patterns(mrf_lookup["all"][k], cluster_lookup["all"][k], puz_idx_lookup)
            subseries_lookup[k] = make_subseries_lookup(k, patterns, mrf_lookup["all"][k], all_series, noise)
        run_sub_TICC(subseries_lookup, args.datapath, "all", SUBPATTERN_KRANGE)
    else:
        noise = np.loadtxt(args.datapath + "/noise_values.txt")
        all_series = np.loadtxt(args.datapath + "/all_series.txt")

        with open(args.datapath + "/idx_lookup.pickle", 'rb') as fp:
            idx_lookup = pickle.load(fp)
        with open(args.datapath + "/puz_idx_lookup.pickle", 'rb') as fp:
            puz_idx_lookup = pickle.load(fp)

        cluster_lookup, mrf_lookup, model_lookup, bic_lookup = load_TICC_output(args.datapath, ["all"], krange)
        with open(args.datapath + "/all/subpatterns/subseries_lookup.pickle", 'rb') as fp:
            subseries_lookup = pickle.load(fp)

    sub_lookup = load_sub_lookup(args.datapath + "/all", subseries_lookup)

    if args.no_test:
        sys.exit(0)

    test_puz_idx_lookup, test_series_lookup, _ = make_series(test_data, noise)
    test_idx_lookup, test_all_series = combine_user_series(test_series_lookup, noise)
    test_puz_idx_lookup = {(uid, pid): (s + test_idx_lookup[uid[:uid.index('e')] if 'e' in uid else uid][0],
                                        e + test_idx_lookup[uid[:uid.index('e')] if 'e' in uid else uid][0])
                           for (uid, pid), (s, e) in test_puz_idx_lookup.items()}

    logging.debug("Predicting clusters on test data")
    cluster_lookup["test"] = {}
    test_subseries_lookup = {}
    test_subcluster_lookup = {}
    for k in krange:
        cluster_lookup["test"][k] = predict_from_saved_model(test_all_series, model_lookup["all"][k])
        test_patterns = get_patterns(mrf_lookup["all"][k], cluster_lookup["test"][k], test_puz_idx_lookup)
        test_subseries_lookup[k] = make_subseries_lookup(k, test_patterns, mrf_lookup["all"][k], test_all_series, noise)
        test_subcluster_lookup[k] = {}
        for cid in range(k):
            if cid not in test_subseries_lookup[k]:
                continue
            test_subcluster_lookup[k][cid] = {}
            for k2 in SUBPATTERN_KRANGE:
                test_subcluster_lookup[k][cid][k2] = predict_from_saved_model(test_subseries_lookup[k][cid]["series"],
                                                                              sub_lookup["models"][k][cid][k2])

    logging.debug("Computing cluster times")
    results = compute_cluster_times(data, cluster_lookup["all"], mrf_lookup["all"], puz_idx_lookup)
    results = compute_subcluster_times(results, cluster_lookup["all"], sub_lookup["clusters"], subseries_lookup, puz_idx_lookup)
    test_results = compute_cluster_times(test_data, cluster_lookup["test"], mrf_lookup["all"], test_puz_idx_lookup)
    test_results = compute_subcluster_times(test_results, cluster_lookup["test"], test_subcluster_lookup, test_subseries_lookup,
                                            test_puz_idx_lookup)

    baseline_features = ["relevant_time", "time", "action_count_all", "action_count_relevant", "action_count_best", "action_rate_all",
                         "action_rate_relevant", "median_prior_perf", "experience", "best_energy_time"]
    
    with open("data/user_metadata_v2.csv") as fp:
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
    user_meta_lookup = {uid: list(metas) for uid, metas in groupby(sorted(user_metas.values(), key=lambda m: m['uid']), lambda m: m['uid'])}

    with open("data/puzzle_categories_latest.csv") as fp:
        puz_cat = {r['nid']: r['categories'].split(',') for r in csv.DictReader(fp)}
    
    results["action_count_all"] = results.apply(lambda r: user_metas[(r.uid, r.pid)]["action_count_all"], axis=1)
    results["action_count_relevant"] = results.apply(lambda r: user_metas[(r.uid, r.pid)]["action_count_relevant"], axis=1)
    results["action_count_best"] = results.apply(lambda r: user_metas[(r.uid, r.pid)]["action_count_best"], axis=1)
    results["best_energy_time"] = results.apply(lambda r: user_metas[(r.uid, r.pid)]["best_energy_time"], axis=1)
    results["action_rate_all"] = results.apply(lambda r: r.action_count_all / r.time, axis=1)
    results["action_rate_relevant"] = results.apply(lambda r: r.action_count_relevant / r.relevant_time, axis=1)
    results["experience"] = results.apply(lambda r: len([x for x in user_meta_lookup[r.uid] if x['pid'] < r.pid and category_lookup["beginner"] not in puz_cat[x['pid']]]), axis=1)
    results["median_prior_perf"] = results.apply(lambda r: np.median([float(x['perf']) for x in user_meta_lookup[r.uid] if x['pid'] < r.pid and category_lookup["beginner"] not in puz_cat[x['pid']]]), axis=1)
    results.median_prior_perf.fillna(results.median_prior_perf.median(), inplace=True)
    
    test_results["action_count_all"] = test_results.apply(lambda r: user_metas[(r.uid, r.pid)]["action_count_all"], axis=1)
    test_results["action_count_relevant"] = test_results.apply(lambda r: user_metas[(r.uid, r.pid)]["action_count_relevant"], axis=1)
    test_results["action_count_best"] = test_results.apply(lambda r: user_metas[(r.uid, r.pid)]["action_count_best"], axis=1)
    test_results["best_energy_time"] = test_results.apply(lambda r: user_metas[(r.uid, r.pid)]["best_energy_time"], axis=1)
    test_results["action_rate_all"] = test_results.apply(lambda r: r.action_count_all / r.time, axis=1)
    test_results["action_rate_relevant"] = test_results.apply(lambda r: r.action_count_relevant / r.relevant_time, axis=1)
    test_results["experience"] = test_results.apply(lambda r: len([x for x in user_meta_lookup[r.uid] if x['pid'] < r.pid and category_lookup["beginner"] not in puz_cat[x['pid']]]), axis=1)
    test_results["median_prior_perf"] = test_results.apply(lambda r: np.median([float(x['perf']) for x in user_meta_lookup[r.uid] if x['pid'] < r.pid and category_lookup["beginner"] not in puz_cat[x['pid']]]), axis=1)
    test_results.median_prior_perf.fillna(test_results.median_prior_perf.median(), inplace=True)

    models = {#"ridge": linear_model.Ridge,
              "ensemble": ensemble.GradientBoostingRegressor,}
              #"gam": pygam.LinearGAM}
    model_params = {"ridge": {"random_state": seed},
                    "ensemble": {"random_state": seed, "learning_rate": 0.1, "subsample": 0.5,
                                 "loss": "huber", "n_estimators": 1000, "n_iter_no_change": 100,
                                 "alpha": 0.95},
                    "gam": {}}

    print("BASELINE MODELS")
    for label, model in models.items():
        fsets = [fset for fset in chain(*[combinations(baseline_features, n) for n in range(1, len(baseline_features) + 1)])]
        ms = [model(**model_params[label]).fit(results[list(fset)], results.perf) for fset in fsets]
        base_fset, base_m = min(zip(fsets, ms), key=lambda x: mean_squared_error(test_results.perf, x[1].predict(test_results[list(x[0])])))
        print(label)
        print("explained variance", explained_variance_score(test_results.perf,
                                                             base_m.predict(test_results[list(base_fset)])))
        print("RMSE", mean_squared_error(test_results.perf,
                                         base_m.predict(test_results[list(base_fset)]))**0.5)
        print(base_fset)
        print()
    print()


    print("MODELS OF PATTERN TIMES".format(k))
    for ftype in ["time", "ratio", "action", "action_ratio"]:
        print(ftype)
        for label, model in models.items():
            print(label)
            fsets = []
            for k in krange:
                print("k={}".format(k), end=", ")
                sys.stdout.flush()
                patterns_best_fsets = []
                for cid in range(k):
                    if "cluster_{}_{}_k{}".format(cid, ftype, k) not in results.columns:
                        continue
                    std_fsets = [["cluster_{}_{}_k{}".format(n, ftype, k) for n in range(k) if "cluster_{}_{}_k{}".format(n, ftype, k)
                                  in results.columns]]
                    std_fsets.append([x for x in std_fsets[0] if "_{}_".format(cid) not in x])
                    for k2 in SUBPATTERN_KRANGE:
                        std_fsets.append(std_fsets[1] + ["sk{}_subcluster_{}{}_{}_k{}".format(k2, cid, l, ftype, k) for l
                                                         in string.ascii_uppercase[:k2] if "sk{}_subcluster_{}{}_{}_k{}".format(k2, cid, l, ftype, k) in test_results.columns])
                    for fset in std_fsets:
                        fset.extend(base_fset)
                    ms = [model(**model_params[label]).fit(results[fset], results.perf) for fset in std_fsets]
                    best_fset = min(zip(ms, std_fsets),
                                    key=lambda x: mean_squared_error(test_results.perf, x[0].predict(test_results[x[1]])))[1]
                    patterns_best_fsets.extend([f for f in best_fset if re.search(r"cluster_{}[_A-Z]".format(cid), f)])
                features = patterns_best_fsets  # [f for f in results.columns if "_k{}".format(k) in f]
                best_fset = features
                best_fset.extend(base_fset)
                best_m = model(**model_params[label]).fit(results[best_fset], results.perf)
                cur_MSE = mean_squared_error(test_results.perf, best_m.predict(test_results[features]))
                new_MSE = 0
                while new_MSE < cur_MSE and len(best_fset) > 1:
                    new_MSE = cur_MSE = mean_squared_error(test_results.perf, best_m.predict(test_results[best_fset]))
                    for fset in combinations(best_fset, len(best_fset) - 1):
                        fset = list(fset)
                        m = model(**model_params[label]).fit(results[fset], results.perf)
                        MSE = mean_squared_error(test_results.perf, m.predict(test_results[fset]))
                        if MSE < new_MSE:
                            best_fset = fset
                            best_m = m
                            new_MSE = MSE
                if len(features) == 1:  # edge case where patterns_best_fsets ends up with just a single feature
                    m = model(**model_params[label]).fit(results[best_fset], results.perf)
                    # ensure new_MSE has correct value, not init value of 0 when while loop is never entered
                    new_MSE = mean_squared_error(test_results.perf, m.predict(test_results[best_fset]))
                fsets.append((new_MSE, best_fset))
            print()
            best_fset = min(fsets)[1]
            best_m = model(**model_params[label]).fit(results[best_fset], results.perf)
            print(best_fset)
            if hasattr(best_m, "feature_importances_"):
                print(best_m.feature_importances_)
            print("explained variance", explained_variance_score(test_results.perf, best_m.predict(test_results[best_fset])))
            print("RMSE", mean_squared_error(test_results.perf, best_m.predict(test_results[best_fset]))**0.5)
            print()
