"""
common functions for pattern extraction
"""
import argparse
import copy
import sys
from functools import reduce, partial
from operator import itemgetter
from typing import List, Tuple, Dict, Optional, Callable, Any, Iterable, Union, Set, Hashable
import typing

from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from statsmodels import robust

from ticc.TICC_solver import TICC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import json
import os
import pickle
import logging
import string
from collections import Counter, OrderedDict
from itertools import groupby
from util import *
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import matplotlib

matplotlib.use("Agg")


def combine_user_series(series_lookup: Dict[Hashable, np.ndarray], noise: np.ndarray,
                        default_noise_duration=1000) -> Tuple[Dict[Hashable, Tuple[int, int]], np.ndarray]:
    """
    concatenate all the series in series lookup together
    :param series_lookup: dictionary mapping uid to time series
    :param noise: noise values to tile between users
    :param default_noise_duration:
    :return: dictionary mapping uid to a tuple of the start and end indices in all_series for that user and the
             resulting concatenate series
    """
    uids = sorted(series_lookup.keys())
    all_series = series_lookup[uids[0]]
    idx_lookup = {}
    idx_lookup[uids[0]] = (0, len(all_series))
    for uid in uids[1:]:
        ser = series_lookup[uid]
        idx_lookup[uid] = (len(all_series) + default_noise_duration,
                           len(all_series) + default_noise_duration + len(ser))
        all_series = np.concatenate((all_series, np.tile(noise, (default_noise_duration, 1)), ser))
    # correct off-by-one error for the range that includes the last point
    idx_lookup[uids[-1]] = (idx_lookup[uids[-1]][0], idx_lookup[uids[-1]][1] + 1)
    return idx_lookup, all_series


def fit_TICC(series: np.ndarray, k: int, window_size=1, num_proc=4,
             ticc_instance=None) -> Tuple[np.ndarray, Dict[int, np.ndarray], float, dict]:
    """

    :param series:
    :param k:
    :param window_size:
    :param num_proc:
    :param ticc_instance:
    :return:
    """
    if ticc_instance is None:
        ticc = TICC(window_size=window_size, number_of_clusters=k, compute_BIC=True, num_proc=num_proc)
    else:
        ticc = ticc_instance
    clusters, mrfs, bic = ticc.fit(series)
    ticc_props = {"trained_model": ticc.trained_model,
                  "window_size": ticc.window_size,
                  "num_blocks": ticc.num_blocks,
                  "number_of_clusters": ticc.number_of_clusters,
                  "switch_penalty": ticc.switch_penalty}
    if ticc_instance is None:  # clean up ticc solver if this isn't a reused instance
        ticc.pool.close()
    return clusters, mrfs, bic, ticc_props


def run_TICC(series_lookup: Dict[str, np.ndarray], datapath: str, krange: Iterable[int],
             save_model=True, skip_series_fn=None, window_size=1, num_proc=4) -> None:
    """

    :param series_lookup: uid -> series
    :param datapath:
    :param krange:
    :param save_model:
    :param skip_series_fn: predicate of a series and k indicating whether to skip it
    :param window_size:
    :param num_proc:
    :return:
    """
    if not os.path.exists(datapath):
        raise ValueError("datapath {} does not exist".format(datapath))
    with ProcessPoolExecutor(min(len(series_lookup), os.cpu_count() // num_proc)) as pool:
        for k in krange:
            logging.debug(f"\n{k} clusters")
            tasks = []
            for uid, series in series_lookup.items():
                if skip_series_fn and skip_series_fn(series, k):
                    logging.info("SKIPPED {} for k = {}".format(uid, k))
                else:
                    tasks.append((uid, pool.submit(fit_TICC, series, k, window_size, num_proc)))
            for uid, task in tasks:
                logging.debug(uid)
                clusters, mrfs, bic, ticc_props = task.result()

                os.makedirs(f"{datapath}/{uid}", exist_ok=True)
                # noinspection PyTypeChecker
                np.savetxt(f"{datapath}/{uid}/clusters_k{k}.txt", clusters)
                with open(f"{datapath}/{uid}/mrfs_k{k}.pickle", 'wb') as pkl:
                    pickle.dump(mrfs, pkl)
                if save_model:
                    with open(f"{datapath}/{uid}/ticc_model_k{k}.pickle", 'wb') as pkl:
                        pickle.dump(ticc_props, pkl)
                # if the user doesn't remove bics.csv in between multiple runs, there will be duplicate entries
                with open(f"{datapath}/{uid}/bics.csv", 'a') as bic_out:
                    bic_out.write(f"{k},{bic}\n")


def make_subseries_lookup(k: int, patterns: List[PatternInstance], mrfs: Dict[int, np.ndarray],
                          all_series: np.ndarray, noise: np.ndarray) -> Dict[int, SubSeriesLookup]:
    """
    returns a dictionary from k (clusters in top-level TICC model) to dictionaries from cluster id (cid) to dictionaries
    with entries for
    patterns: (uid, pid, start_index) -> action time series for that pattern
    series: the series constructed by concatenating the action time series for all patterns with this cid (with separating noise)
    idx_lookup: (uid, pid, start_index) -> index in series where this pattern begins
    """
    lookup: Dict[int, SubSeriesLookup] = {}
    for cid in range(k):
        if is_null_cluster(mrfs[cid]):
            continue
        # logging.debug(f"assembling series for pattern {cid}")
        subpatterns = {(p.uid, p.pid, p.start_idx): all_series[p.start_idx:p.end_idx] for p in patterns if
                        p.cid == cid}
        if len(subpatterns) == 0:
            logging.debug(f"skipping pattern {cid} because there are no instances of it")
            continue
        sub_idx_lookup, all_subseries = combine_user_series(subpatterns, noise, default_noise_duration=100)
        # if len(all_subseries) - 100 * (len(sslu["patterns"]) - 1) < 20:
        #     logging.debug("skipping pattern {} because it only appears for {} timesteps".format(cid, len(all_subseries) - 100 * (len(sslu["patterns"]) - 1)))
        #     continue
        lookup[cid] = SubSeriesLookup(subpatterns, all_subseries, sub_idx_lookup)
    return lookup


def run_sub_TICC(subseries_lookups: dict, datapath: str, uid: str, sub_krange: list, save_model=True,
                 skip_series_fn=None, window_size=1, num_proc=4):
    """

    :param subseries_lookups:
    :param datapath:
    :param uid:
    :param sub_krange:
    :param save_model:
    :param skip_series_fn:
    :param window_size:
    :param num_proc:
    :return:
    """
    results_dir = f"{datapath}/{uid}/subpatterns"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/subseries_lookups.pickle", 'wb') as fp:
        pickle.dump(subseries_lookups, fp)

    for k in subseries_lookups:
        os.makedirs(f"{results_dir}/k{k}", exist_ok=True)

    with ProcessPoolExecutor(min(len(subseries_lookups), os.cpu_count() // num_proc)) as pool:
        pool.map(partial(run_TICC, krange=sub_krange, save_model=save_model, skip_series_fn=skip_series_fn,
                         window_size=window_size, num_proc=num_proc),
                 [{f"cid{cid}": lookup.series for cid, lookup in subseries_lookups[k].items()} for k in subseries_lookups],
                 [f"{results_dir}/k{k}" for k in subseries_lookups])
    # for k in subseries_lookups:
    #     logging.debug(f"running TICC to get subpatterns for k={k}")
    #     os.makedirs(results_dir + "/k{}".format(k), exist_ok=True)
    #     run_TICC({f"cid{cid}": lookup["series"] for cid, lookup in subseries_lookups[k].items()},
    #              f"{results_dir}/k{k}", sub_krange, save_model, skip_series_fn, window_size, num_proc)


def is_null_cluster(mrf: np.ndarray) -> bool:
    """
    predicate for whether a cluster is null/invalid based on the mrf
    :param mrf:
    :return:
    """
    return np.triu(mrf, 1).sum() == 0


def load_TICC_output(datapath: str, uids: List[str], krange: Iterable[int]) -> Tuple[
    Dict[str, Dict[int, np.ndarray]],
    Dict[str, Dict[int, Dict[int, np.ndarray]]],
    Dict[str, Dict[int, Dict]],
    Dict[str, Dict[int, float]]]:
    """
    load in output written to files by run_TICC
    :param datapath:
    :param uids:
    :param krange:
    :return: cluster_lookup mapping uid to dictionary from k to cluster labels
             mrf_lookup mapping uid to dictionary from k to mrf dictionary (cluster label to mrf)
             model_lookup mapping uid to dictionary from k to dictionary of ticc model parameters
             bic_lookup mapping uid to dictionary from k to bic
    """
    cluster_lookup = {}
    mrf_lookup = {}
    bic_lookup = {}
    model_lookup = {}
    for uid in uids:
        if os.path.exists(f"{datapath}/{uid}") and any(
                os.path.exists(f"{datapath}/{uid}/clusters_k{k}.txt") for k in krange):
            cd = {}
            md = {}
            modd = {}
            for k in krange:
                if os.path.exists(f"{datapath}/{uid}/clusters_k{k}.txt"):
                    cd[k] = np.loadtxt(f"{datapath}/{uid}/clusters_k{k}.txt", np.int32)
                    with open(f"{datapath}/{uid}/mrfs_k{k}.pickle", 'rb') as fp:
                        md[k] = pickle.load(fp)
                    with open(f"{datapath}/{uid}/ticc_model_k{k}.pickle", 'rb') as fp:
                        modd[k] = pickle.load(fp)
            cluster_lookup[uid] = cd
            mrf_lookup[uid] = md
            model_lookup[uid] = modd
            with open(f"{datapath}/{uid}/bics.csv") as fp:
                bic_in = csv.DictReader(fp, fieldnames=["k", "bic"])
                bics = {}
                # guard against duplicate entries by having the most recent one take precedence
                for row in bic_in:
                    bics[int(row["k"])] = float(row["bic"])
                bic_lookup[uid] = bics
        else:
            # logging.warning("TICC output for uid {} not found".format(uid))
            raise FileNotFoundError(f"TICC output for uid {uid} not found")
    return cluster_lookup, mrf_lookup, model_lookup, bic_lookup


def load_sub_lookup(datapath: str, subseries_lookup: dict, sub_krange: Iterable[int]) -> SubLookup:
    """
    loads the output of run_sub_TICC
    :param datapath:
    :param subseries_lookup:
    :param sub_krange:
    :return:
    """
    clusters = {} # Dict[int, Dict[int, Dict[int, np.ndarray]]] (k to cid to sub_k to cluster labels)
    mrfs = {}     # Dict[int, Dict[int, Dict[int, Dict[int, np.ndarray]]]] (k to cid to sub_k to mrf dictionary (cluster label to mrf))
    models = {}   # Dict[int, Dict[int, Dict[int, Dict]]] (k to cid to sub_k to dict of ticc model parameters)
    bics = {}     # Dict[int, Dict[int, Dict[int, float]]] (k to cid to sub_k to bic)
    for k in subseries_lookup:
        dp = "{}/subpatterns/k{}".format(datapath, k)
        cs, ms, mods, bs = load_TICC_output(dp, ["cid{}".format(cid) for cid in subseries_lookup[k]], sub_krange)
        clusters[k] = {int(k.replace("cid", "")): v for k, v in cs.items()}
        mrfs[k] = {int(k.replace("cid", "")): v for k, v in ms.items()}
        models[k] = {int(k.replace("cid", "")): v for k, v in mods.items()}
        bics[k] = {int(k.replace("cid", "")): v for k, v in bs.items()}
    return SubLookup(clusters, mrfs, models, bics)


def handles_noise(series: np.ndarray, null_cids: List[int], clusters: np.ndarray, noise: np.ndarray) -> bool:
    """
    check whether a given pattern model correctly assigns noise (and only noise) to null clusters
    :param series: one user's series
    :param null_cids: the cluster ids for all null clusters
    :param clusters: the cluster assignments for the user's series
    :param noise: the noise value used in the series
    :return: true if the cluster assignment handles noise correctly
    """
    return len(null_cids) > 0 and \
           all((clusters[i] in null_cids if (series[i] == noise).all() else clusters[i] not in null_cids)
               for i in range(len(series)))


def select_TICC_model(cluster_lookup: Dict[str, Dict[int, np.ndarray]],
                      mrf_lookup: Dict[str, Dict[int, Dict[int, np.ndarray]]],
                      series_lookup: Dict[str, np.ndarray], noise: np.ndarray,
                      selection_metric: Callable[[Dict[int, np.ndarray]], float], minimize=True) -> Dict[str, int]:
    """
    :param cluster_lookup: dictionary from uid to a dictionary from k to cluster labels
    :param mrf_lookup: dictionary from uid to a dictionary from k to a dictionary from cluster label to mrf
    :param series_lookup: dictionary from uid to time series
    :param noise: static noise values used in constructing time series
    :param selection_metric: function take takes a dictionary from cluster label to mrf and returns a quality score
    :param minimize: if True, optimal k is that which minimizes selection_metric, otherwise optimal k is that which maximizes
    :return: dictionary from uid to optimal k
    """
    best_k = {}
    for uid in series_lookup:
        logging.debug("model selection for {}".format(uid))
        cd = cluster_lookup[uid]
        md = mrf_lookup[uid]
        ser = series_lookup[uid]
        no_dups = {k for k in cd if
                   not any(any((m == md[k][ci]).all() for cj, m in md[k].items() if cj > ci) for ci in range(k))}
        logging.debug("k with no dups: {}".format(no_dups))
        null_clusters = {k: [ci for ci in md[k] if is_null_cluster(md[k][ci])] for k in md}
        handles_noise = {k for k in cd if len(null_clusters[k]) > 0 and all(
            cd[k][i] in null_clusters[k] for i in range(len(ser)) if (ser[i] == noise).all())}
        logging.debug("k that handle noise: {}".format(handles_noise))
        valid_k = no_dups.intersection(handles_noise)
        if len(valid_k) > 0:
            scores = {k: selection_metric(md[k]) for k in valid_k}
            logging.debug("scores: {}".format(scores))
            if minimize:
                best_k[uid] = min(scores.items(), key=lambda x: x[1])[0]
            else:
                best_k[uid] = max(scores.items(), key=lambda x: x[1])[0]
        else:
            logging.info(
                "unable to select TICC model for uid {}, k with no dups={}, k that handled noise={}".format(uid,
                                                                                                            no_dups,
                                                                                                            handles_noise))
    return best_k


def get_pattern_masks(uid: str, pid: str, idx: Tuple[int, int], target_pts: Iterable[str], sub_ks: Dict[int, int],
                      pattern_lookup: PatternLookup,
                      subseries_lookup: Dict[int, SubSeriesLookup]) -> 'OrderedDict[str, np.ndarray]':
    """
    idx: tuple of the start and end index for (uid, pid) segment in all_series (i.e., puz_idx_lookup entry)
    target_pts: list of pattern type labels (e.g., {'1A', '1C', '2A', '3'})
    sub_ks: maps sub cid to sub k value (needed for indexing into pattern_lookup)
    pattern_lookup: nested mapping for patterns (like what get_pattern_lookup returns) already k-indexed (e.g., pattern_lookup = pattern_lookup[5])
    subseries_lookup: also already k-indexed
    """
    pdict = OrderedDict()
    offset, e = idx
    n = e - offset
    for pt in sorted(target_pts):
        if pt.isnumeric():
            spans = [(p.start_idx, p.end_idx) for p in pattern_lookup[int(pt)][0] if p.uid == uid and p.pid == pid]
        else:
            cid = int(pt[:-1])
            sub_cid = string.ascii_uppercase.index(pt[-1])
            ps = [p for p in pattern_lookup[cid][sub_ks[cid]] if p.uid == uid and p.pid == pid and p.cid == sub_cid]
            spans = []
            for p in ps:
                sub_offsets = [(sub_offset, sub_s, sub_e) for (ui, pi, sub_offset), (sub_s, sub_e) in
                               subseries_lookup[cid].idx_lookup.items() if
                               uid == ui and pid == pi and p.start_idx >= sub_s and p.end_idx <= sub_e]
                assert len(sub_offsets) == 1
                sub_offset, sub_s, sub_e = sub_offsets[0]
                spans.append((p.start_idx - sub_s + sub_offset, p.end_idx - sub_s + sub_offset))
        mask = np.full(n, False)
        for s, e in spans:
            mask[s - offset:e - offset] = True
            # if s == e:
            #     mask[s - offset] = True
        pdict[pt] = mask
    # assert reduce(np.logical_or, pdict.values()).all()
    return pdict


def get_patterns(mrfs: Dict[int, np.ndarray], clusters: np.ndarray, puz_idx_lookup: dict) -> List[PatternInstance]:
    """

    :param mrfs:
    :param clusters:
    :param puz_idx_lookup:
    :return:
    """
    patterns = []
    for cid in mrfs:
        if is_null_cluster(mrfs[cid]):
            continue
        starts = [j for j, c in enumerate(clusters[:-1], 1) if clusters[j] != c and clusters[j] == cid]
        if clusters[0] == cid:
            starts = [0] + starts
        for start in starts:
            key = next(key for key, r in puz_idx_lookup.items() if start in range(*r))
            uid = key[0]
            pid = key[1]
            end = next(i for i, c in enumerate(clusters[start:], start) if c != cid or i + 1 == len(clusters))
            if end == len(clusters) - 1:  # put end one past the last index so the slice actually grabs the last point
                end += 1
            patterns.append(PatternInstance(cid, uid, pid, start, end))
    return patterns


def get_pattern_lookups(krange: Iterable[int], sub_clusters: SubClusters,
                        sub_mrfs: SubMRFs, subseries_lookups: Dict[int, Dict[int, SubSeriesLookup]],
                        cluster_lookup: Dict[int, np.ndarray], mrf_lookup: Dict[int, Dict[int, np.ndarray]],
                        puz_idx_lookup: dict) -> Dict[int, PatternLookup]:
    """
    sub_clusters = sub_lookup["clusters"]
    mrf_lookup and cluster_lookup need to be label-indexed (i.e., mrf_lookup = mrf_lookup["all"])
    """
    # maps k to either "base" -> list of first-order patterns or cluster id -> sub_k -> list of patterns
    patterns: Dict[int, Dict[Union[int, str], Union[List[PatternInstance], Dict[int, List[PatternInstance]]]]] = {}
    for k in krange:
        patterns[k] = {"base": get_patterns(mrf_lookup[k], cluster_lookup[k], puz_idx_lookup)}
        cids: Set[int] = {p.cid for p in patterns[k]["base"]}
        for cid in cids:
            patterns[k][cid] = {0: [p for p in patterns[k]["base"] if p.cid == cid]}
            if cid not in sub_clusters[k]:
                continue
            ps = [p for p in patterns[k]["base"] if p.cid == cid]
            sub_idx_lookup = subseries_lookups[k][cid].idx_lookup
            for sub_k in sub_clusters[k][cid]:
                if not any(is_null_cluster(mrf) for mrf in sub_mrfs[k][cid][sub_k].values()) or \
                        len(sub_idx_lookup) <= sub_k:
                    continue
                patterns[k][cid][sub_k] = get_patterns(sub_mrfs[k][cid][sub_k], sub_clusters[k][cid][sub_k],
                                                       sub_idx_lookup)
    return patterns


def predict_from_saved_model(test_data: np.ndarray, saved_model: dict) -> np.ndarray:
    """

    :param test_data:
    :param saved_model:
    :return:
    """
    test_ticc = TICC(window_size=saved_model["window_size"], number_of_clusters=saved_model["number_of_clusters"], num_proc=1)
    test_ticc.num_blocks = saved_model["num_blocks"]
    test_ticc.switch_penalty = saved_model["switch_penalty"]
    test_ticc.trained_model = saved_model["trained_model"]
    cs = test_ticc.predict_clusters(test_data)
    test_ticc.pool.close()
    return cs


def compute_pattern_times(k: int, subs: tuple, data: pd.DataFrame, cluster_lookup: Dict[int, np.ndarray],
                          mrf_lookup: Dict[int, Dict[int, np.ndarray]], puz_idx_lookup: dict) -> pd.DataFrame:
    """

    :param k:
    :param subs:
    :param data:
    :param cluster_lookup:
    :param mrf_lookup:
    :param puz_idx_lookup:
    :return:
    """
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


def compute_subpattern_times(k: int, subs: Tuple[int, int], data: pd.DataFrame, cluster_lookup: Dict[int, np.ndarray],
                             sub_clusters: SubClusters, subseries_lookups: Dict[int, Dict[int, SubSeriesLookup]],
                             puz_idx_lookup: dict) -> pd.DataFrame:
    """

    :param k:
    :param subs:
    :param data:
    :param cluster_lookup:
    :param sub_clusters:
    :param subseries_lookups:
    :param puz_idx_lookup:
    :return:
    """
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
        cs = sub_clusters[k][cid][sub_k]
        for (_, _, start_idx), (s, e) in subseries_lookups[k][cid].idx_lookup.items():
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
    sub_cluster_times = [v["times"] for v in results.values() if v["valid"]]
    return data.merge(pd.DataFrame(data=sub_cluster_times), on=['pid', 'uid'])


def get_predicted_lookups(all_series: np.ndarray, krange: Iterable[int], model_lookup: Dict[int, dict],
                          sub_models: Dict[int, Dict[int, Dict[int, dict]]],
                          mrf_lookup: Dict[int, Dict[int, np.ndarray]], puz_idx_lookup: dict, noise: np.ndarray) \
        -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, SubSeriesLookup]], Dict[int, Dict[int, Dict[int, np.ndarray]]]]:
    """

    :param all_series:
    :param krange:
    :param model_lookup:
    :param sub_models:
    :param mrf_lookup:
    :param puz_idx_lookup:
    :param noise:
    :return:
    """
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
                    sub_cs[cid][sub_k] = predict_from_saved_model(subseries_lookup[k][cid].series,
                                                                  sub_models[k][cid][sub_k])
        sub_clusters[k] = sub_cs

    return cluster_lookup, subseries_lookup, sub_clusters


def make_selection_lookups(all_series: np.ndarray, pattern_lookups: Dict[int, PatternLookup],
                           subseries_lookups: Dict[int, Dict[int, SubSeriesLookup]], sub_clusters: SubClusters,
                           sub_mrfs: SubMRFs) -> Tuple[Dict[Tuple, np.ndarray], Dict[Tuple, np.ndarray]]:
    """

    :param all_series:
    :param pattern_lookups:
    :param subseries_lookups:
    :param sub_clusters:
    :param sub_mrfs:
    :return:
    """
    dispersions: Dict[Tuple, np.ndarray] = {}
    modes: Dict[Tuple, np.ndarray] = {}
    print("computing selection criteria")
    for k in pattern_lookups:
        print("k =", k)
        for cid in {p.cid for p in pattern_lookups[k]["base"]}:
            print("    cid", cid)
            ss = [all_series[p.start_idx:p.end_idx] for p in pattern_lookups[k]["base"] if p.cid == cid]
            ubiqs = np.array([[len([x for x in s[:, i] if x > 0]) / len(s) for i in range(s.shape[1])] for s in ss])

            dispersions[(k, cid, 0)] = np.mean(robust.mad(ubiqs, axis=0))
            modes[(k, cid, 0)] = stats.mode(np.round(ubiqs, 1)).mode

            if cid not in sub_clusters[k]:
                continue

            idx_lookup = subseries_lookups[k][cid].idx_lookup
            ser = subseries_lookups[k][cid].series
            for sub_k in sub_clusters[k][cid]:
                if not any(is_null_cluster(mrf) for mrf in sub_mrfs[k][cid][sub_k].values()) or len(idx_lookup) <= sub_k:
                    continue
                ps = pattern_lookups[k][cid][sub_k]
                for sub_cid in {p.cid for p in ps}:
                    ss = [ser[p.start_idx:p.end_idx] for p in ps if p.cid == sub_cid]
                    ubiqs = np.array([[len([x for x in s[:, i] if x > 0]) / len(s) for i in range(s.shape[1])] for s in ss])
                    dispersions[(k, cid, sub_k, sub_cid)] = np.mean(robust.mad(ubiqs, axis=0))
                    modes[(k, cid, sub_k, sub_cid)] = stats.mode(np.round(ubiqs, 1)).mode

    dispersion_lookup = {tag: [x[1] for x in sorted(xs)] for tag, xs in groupby(sorted(dispersions.items()), lambda x: x[0][:3])}
    mode_lookup = {tag: [x[1] for x in sorted(xs)] for tag, xs in groupby(sorted(modes.items()), lambda x: x[0][:3])}

    return dispersion_lookup, mode_lookup


def dispersion_score(k: int, candidate: Iterable[Tuple[int, int]], dispersion_lookup: Dict[Tuple, np.ndarray],
                     pattern_lookups: Dict[int, PatternLookup]) -> float:
    """

    :param k:
    :param candidate:
    :param dispersion_lookup:
    :param pattern_lookups:
    :return:
    """
    sub_weights = [[len([p for p in pattern_lookups[k][cid][sub_k] if p.cid == sub_cid]) for sub_cid in
                    sorted({p.cid for p in pattern_lookups[k][cid][sub_k]})] for cid, sub_k in candidate]
    return np.average([np.average(dispersion_lookup[(k, cid, sub_k)], weights=ws) for ws, (cid, sub_k) in zip(sub_weights, candidate)],
                      weights=[len(pattern_lookups[k][cid][sub_k]) for cid, sub_k in candidate])


def mode_score(k: int, candidate: Iterable[Tuple[int, int]], mode_lookup: Dict[Tuple, np.ndarray]):
    """

    :param k:
    :param candidate:
    :param mode_lookup:
    :return:
    """
    ms = sum((mode_lookup[(k, cid, sub_k)] for cid, sub_k in candidate), [])
    return np.mean([min([abs(x - m).sum() for x in ms if x is not m]) for m in ms])


def find_best_dispersion_model(all_series: np.ndarray, pattern_lookups: Dict[int, PatternLookup],
                               subseries_lookups: Dict[int, Dict[int, SubSeriesLookup]],
                               sub_clusters: SubClusters) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
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
    dispersion_lookup, mode_lookup = make_selection_lookups(all_series, pattern_lookups, subseries_lookups, sub_clusters)
    disp_scores = {}
    for k in pattern_lookups:
        print("k =", k)
        candidate = []
        cids: Set[int] = {p.cid for p in pattern_lookups[k]["base"]}
        for cid in cids:
            xs = [(cid, 0)]
            if cid not in sub_clusters[k]:
                candidate.append(xs[0])
                continue
            idx_lookup = subseries_lookups[k][cid].idx_lookup
            for sub_k in sub_clusters[k][cid]:
                if (k, cid, sub_k) not in dispersion_lookup:
                    print("SKIPPING", k, cid, sub_k)
                    continue
                xs.append((cid, sub_k))
            candidate.append(min(xs, key=lambda x: np.mean(dispersion_lookup[(k, x[0], x[1])])))
        disp_scores[(k, tuple(candidate))] = dispersion_score(k, candidate, dispersion_lookup, pattern_lookups)
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
