import argparse
import copy
import sys
from functools import reduce, partial
from operator import itemgetter
from typing import List, Tuple, Dict, Optional, Callable, Any, Iterable
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
import matplotlib

matplotlib.use("Agg")


def combine_user_series(series_lookup: Dict[str, np.ndarray], noise: np.ndarray,
                        default_noise_duration=1000) -> Tuple[Dict[str, Tuple[int, int]], np.ndarray]:
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
             ticc_instance=None) -> Tuple[np.ndarray, Dict[int, np.ndarray], float]:
    if ticc_instance is None:
        ticc = TICC(window_size=window_size, number_of_clusters=k, compute_BIC=True, num_proc=num_proc)
    else:
        ticc = ticc_instance
    clusters, mrfs, bic = ticc.fit(series)
    if ticc_instance is None:  # clean up ticc solver if this isn't a reused instance
        ticc.pool.close()
    return clusters, mrfs, bic


def run_TICC(series_lookup: Dict[str, np.ndarray], datapath: str, krange: list,
             save_model=True, skip_series_fn=None, window_size=1, num_proc=4) -> None:
    if not os.path.exists(datapath):
        raise ValueError("datapath {} does not exist".format(datapath))
    for k in krange:
        logging.debug(f"\n{k} clusters")
        ticc = TICC(window_size=window_size, number_of_clusters=k, compute_BIC=True, num_proc=num_proc)
        for uid, series in series_lookup.items():
            if skip_series_fn and skip_series_fn(series, k):
                logging.info("SKIPPED {} for k = {}".format(uid, k))
            else:
                logging.debug(uid)
                clusters, mrfs, bic = fit_TICC(series, k, ticc_instance=ticc)
                # order clusters by sum of absolute value of edge weights
                # mrfs_ordered = {k: mrf for k, mrf in zip(mrfs.keys(), sorted(mrfs.values(), key=lambda x: abs(
                #     x[mrf_indices]).sum()))}
                # inverse = {mrf.tostring(): k for k, mrf in mrfs_ordered.items()}
                # cluster_remapping = {k: inverse[v.tostring()] for k, v in mrfs.items()}
                # clusters_remapped = np.array([cluster_remapping[x] for x in clusters])
                os.makedirs(f"{datapath}/{uid}", exist_ok=True)
                np.savetxt(f"{datapath}/{uid}/clusters_k{k}.txt", clusters)
                with open(f"{datapath}/{uid}/mrfs_k{k}.pickle", 'wb') as pkl:
                    pickle.dump(mrfs, pkl)
                if save_model:
                    with open(f"{datapath}/{uid}/ticc_model_k{k}.pickle", 'wb') as pkl:
                        d = {
                            "trained_model": ticc.trained_model,
                            "window_size": ticc.window_size,
                            "num_blocks": ticc.num_blocks,
                            "number_of_clusters": ticc.number_of_clusters,
                            "switch_penalty": ticc.switch_penalty
                        }
                        pickle.dump(d, pkl)
                # if the user doesn't remove bics.csv in between multiple runs, there will be duplicate entries
                with open(f"{datapath}/{uid}/bics.csv", 'a') as bic_out:
                    bic_out.write(f"{k},{bic}\n")
        ticc.pool.close()


def make_subseries_lookup(k: int, patterns: List[PatternInstance], mrfs: Dict[int, np.ndarray],
                          all_series: np.ndarray, noise: np.ndarray) -> Dict[int, Dict[str, Any]]:
    """
    returns a dictionary from k (clusters in top-level TICC model) to dictionaries from cluster id (cid) to dictionaries
    with entries for
    patterns: (uid, pid, start_index) -> action time series for that pattern
    series: the series constructed by concatenating the action time series for all patterns with this cid (with separating noise)
    idx_lookup: (uid, pid, start_index) -> index in series where this pattern begins
    """
    lookup = {}
    for cid in range(k):
        if is_null_cluster(mrfs[cid]):
            continue
        logging.debug(f"assembling series for pattern {cid}")
        sslu = {}
        sslu["patterns"] = {(p.uid, p.pid, p.start_idx): all_series[p.start_idx:p.end_idx] for p in patterns if
                            p.cid == cid}
        if len(sslu["patterns"]) == 0:
            logging.debug(f"skipping pattern {cid} because there are no instances of it")
            continue
        sub_idx_lookup, all_subseries = combine_user_series(sslu['patterns'], noise, default_noise_duration=100)
        # if len(all_subseries) - 100 * (len(sslu["patterns"]) - 1) < 20:
        #     logging.debug("skipping pattern {} because it only appears for {} timesteps".format(cid, len(all_subseries) - 100 * (len(sslu["patterns"]) - 1)))
        #     continue
        sslu["series"] = all_subseries
        sslu["idx_lookup"] = sub_idx_lookup
        lookup[cid] = sslu
    return lookup


def run_sub_TICC(subseries_lookup: dict, datapath: str, uid: str, sub_krange: list, save_model=True,
                 skip_series_fn=None, window_size=1, num_proc=4):
    results_dir = f"{datapath}/{uid}/subpatterns"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/subseries_lookup.pickle", 'wb') as fp:
        pickle.dump(subseries_lookup, fp)
    for k in subseries_lookup:
        logging.debug(f"running TICC to get subpatterns for k={k}")
        os.makedirs(results_dir + "/k{}".format(k), exist_ok=True)
        run_TICC({f"cid{cid}": lookup["series"] for cid, lookup in subseries_lookup[k].items()},
                 f"{results_dir}/k{k}", sub_krange, save_model, skip_series_fn, window_size, num_proc)


def mrf_diff(a: np.ndarray, b: np.ndarray) -> float:
    ac = a[np.triu_indices(a.shape[1])]
    bc = b[np.triu_indices(b.shape[1])]

    return ((ac - bc) ** 2).sum()


def weight_mrfs_by_ubiquity(mrfs: Dict[int, np.ndarray], series: np.ndarray,
                            clusters: np.ndarray, scalar=10) -> Dict[int, np.ndarray]:
    mrfs = copy.deepcopy(mrfs)
    for ci, m in mrfs.items():
        weights = [len([x for x in series[clusters == ci][:, i] if x > 0]) / len(series[clusters == ci]) for i in
                   range(len(m))]
        for i in range(len(m)):
            m[i, i] = weights[i] * scalar
    return mrfs


def is_null_cluster(mrf: np.ndarray) -> bool:
    return np.triu(mrf, 1).sum() == 0


def load_TICC_output(datapath: str, uids: List[str], krange: range) -> Tuple[
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


def load_sub_lookup(datapath: str, subseries_lookup: dict, sub_krange=[5, 10]) -> Dict[str, Dict[int, dict]]:
    sub_lookup = {"clusters": {}, # Dict[int, Dict[int, Dict[int, ndarray]]] (k to cid to sub_k to cluster labels)
                  "mrfs": {},     # Dict[
                  "models": {},
                  "bics": {}}
    for k in subseries_lookup:
        dp = "{}/subpatterns/k{}".format(datapath, k)
        cs, mrfs, ms, bs = load_TICC_output(dp, ["cid{}".format(cid) for cid in subseries_lookup[k]], sub_krange)
        sub_lookup["clusters"][k] = {int(k.replace("cid", "")): v for _, v in cs.items()}
        sub_lookup["mrfs"][k] = {int(k.replace("cid", "")): v for _, v in mrfs.items()}
        sub_lookup["models"][k] = {int(k.replace("cid", "")): v for _, v in ms.items()}
        sub_lookup["bics"][k] = {int(k.replace("cid", "")): v for _, v in bs.items()}
    return sub_lookup


# noinspection PyTypeChecker
def mean_cluster_nearest_neighbor_distance(mrfs: Dict[int, np.ndarray]) -> float:
    return np.mean([min([mrf_diff(mrfs[ci], mrfs[cj]) for cj in mrfs if ci != cj]) for ci in mrfs])


def mean_pattern_cohesion(mrfs, patterns, series):
    cluster_instance_lookup = {cid: list(g) for cid, g in
                               groupby(sorted(patterns, key=lambda p: p.cid), lambda p: p.cid)}
    diffs = []
    for cid, ps in cluster_instance_lookup.items():

        instance_mrfs = [mrfs[cid].copy() for p in ps]
        for p, mrf in zip(ps, instance_mrfs):

            ser = series[p.start_idx:p.end_idx]
            weights = [len([x for x in ser[:, i] if x > 0]) / len(ser) for i in range(len(mrf))]
            for i in range(len(mrf)):
                mrf[i, i] = weights[i] * 10
        centroid = np.mean(instance_mrfs, 0)
        diffs.append(np.max([mrf_diff(mrf, centroid) for mrf in instance_mrfs]))
    return np.mean(diffs)


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
                      pattern_lookup: Dict[int, PatternInstance],
                      subseries_lookup: Dict[int, Dict[str, Any]]) -> typing.OrderedDict[str, np.ndarray]:
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
                               subseries_lookup[cid]['idx_lookup'].items() if
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
    patterns = []
    for cid in mrfs:
        if is_null_cluster(mrfs[cid]):
            continue
        starts = [j for j, c in enumerate(clusters[:-1], 1) if clusters[j] != c and clusters[j] == cid]
        if clusters[0] == cid:
            starts = [0] + starts
        for start in starts:
            key, ran = next((key, range(*r)) for key, r in puz_idx_lookup.items() if start in range(*r))
            uid = key[0]
            pid = key[1]
            end = next(i for i, c in enumerate(clusters[start:], start) if c != cid or i + 1 == len(clusters))
            if end == len(clusters) - 1:  # put end one past the last index so the slice actually grabs the last point
                end += 1
            patterns.append(PatternInstance(cid, uid, pid, start, end))
    return patterns


def get_patterns_lookup(krange: Iterable[int], sub_clusters: Dict[str, Dict[int, np.ndarray]],
                        sub_mrfs: Dict[int, Dict[int, Dict[int, Dict[int, np.ndarray]]]],
                        subseries_lookup: Dict[int, Dict[str, Any]],
                        cluster_lookup: Dict[int, np.ndarray], mrf_lookup: Dict[int, Dict[int, np.ndarray]],
                        puz_idx_lookup: dict) -> Dict[int, Dict[str, List[PatternInstance]]]:
    """
    sub_clusters = sub_lookup["clusters"]
    mrf_lookup and cluster_lookup need to be label-indexed (i.e., mrf_lookup = mrf_lookup["all"])
    """
    patterns = {}
    for k in krange:
        patterns[k] = {"base": get_patterns(mrf_lookup[k], cluster_lookup[k], puz_idx_lookup)}
        cids = {p.cid for p in patterns[k]["base"]}
        for cid in cids:
            patterns[k][cid] = {0: [p for p in patterns[k]["base"] if p.cid == cid]}
            if cid not in sub_clusters[k]:
                continue
            ps = [p for p in patterns[k]["base"] if p.cid == cid]
            sub_idx_lookup = subseries_lookup[k][cid]["idx_lookup"]
            for sub_k in sub_clusters[k][cid]:
                if not any(is_null_cluster(mrf) for mrf in sub_mrfs[k][cid][sub_k].values()) or \
                        len(sub_idx_lookup) <= sub_k:
                    continue
                patterns[k][cid][sub_k] = get_patterns(sub_mrfs[k][cid][sub_k], sub_clusters[k][cid][sub_k],
                                                       sub_idx_lookup)
    return patterns


def predict_from_saved_model(test_data: np.ndarray, saved_model: dict) -> np.ndarray:
    test_ticc = TICC(window_size=saved_model["window_size"], number_of_clusters=saved_model["number_of_clusters"], num_proc=1)
    test_ticc.num_blocks = saved_model["num_blocks"]
    test_ticc.switch_penalty = saved_model["switch_penalty"]
    test_ticc.trained_model = saved_model["trained_model"]
    cs = test_ticc.predict_clusters(test_data)
    test_ticc.pool.close()
    return cs


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


def get_predicted_lookups(all_series: np.ndarray, krange: Iterable[int], model_lookup: Dict[int, dict],
                          sub_models: Dict[int, Dict[int, Dict[int, dict]]],
                          mrf_lookup: Dict[int, Dict[int, np.ndarray]], puz_idx_lookup: dict, noise: np.ndarray):
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


def make_selection_lookups(all_series: np.ndarray, pattern_lookup: Dict[int, Dict[str, Iterable[PatternInstance]]],
                           subseries_lookup: Dict[int, Dict[int, Dict[str, Any]]],
                           sub_clusters, sub_mrfs):
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


def find_best_dispersion_model(all_series: np.ndarray, pattern_lookup, subseries_lookup, sub_clusters):
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