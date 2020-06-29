import argparse
import copy
import sys
from functools import reduce, partial
from typing import List, Tuple, Dict, Optional, Callable, Any
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


def load_data(pids: List[str], evolver=False, min_time=3600) -> Tuple[pd.DataFrame, dict]:
    datafiles = ['data/puzzle_solutions/solution_{}/{}_meta.h5'.format(pid, pid) for pid in pids]
    df_frames, bts_frames, puz_frames = zip(*map(load_frame, datafiles))
    df = pd.concat(df_frames)
    puz_metas = {m.pid: m for m in puz_frames}
    data = df[df.time > min_time]
    if not evolver:
        data = data[data.frontier_pdbs.notnull()]

    return data, puz_metas


def make_lookups(data: pd.DataFrame, soln_lookup: dict, parent_lookup: dict, child_lookup: dict) -> None:
    """
    populates the provided lookup dictionaries with soloist data
    :param soln_lookup: maps sid to PDB_Info
    :param parent_lookup: maps sid to parent sid
    :param child_lookup: maps sid to list of child sids
    """
    infos = data.apply(lambda r: sorted(([p for l in r.lines for p in l.pdb_infos] if r.lines else []) +
                                        ([p for l in r.evol_lines for p in l.pdb_infos] if r.evol_lines else []),
                                      key=lambda p: p.timestamp), axis=1)
    for _, xs in infos.items():
        for x in xs:
            soln_lookup[x.sid] = x
            if x.parent_sid:
                parent_lookup[x.sid] = x.parent_sid
    for parent, g in groupby(sorted([(p, c) for c, p in parent_lookup.items()]), lambda x: x[0]):
        child_lookup[parent] = [c for p, c in g]


def get_children(parent: PDB_Info, soln_lookup: Dict[str, PDB_Info], child_lookup: Dict[str, str],
                 same_uid=True) -> List[PDB_Info]:

    if parent.sid in child_lookup:
        if same_uid:
            return [soln_lookup[c] for c in child_lookup[parent.sid] if soln_lookup[c].uid == parent.uid]
        else:
            return [soln_lookup[c] for c in child_lookup[parent.sid]]
    return []


def get_relevant_sids(pdb: PDB_Info, soln_lookup: Dict[str, PDB_Info],
                      child_lookup: Dict[str, str]) -> Optional[List[str]]:
    """
    :return: a list of sids for the solutions on the path to user's best solution, including immediate branches
    """
    if pdb and pdb.pdl[-1]['actions'] != {} and pdb.parent_sid in soln_lookup:
        backbone = []
        cur = pdb
        while cur.parent_sid and soln_lookup[cur.parent_sid].uid == pdb.uid:
            backbone.append(cur)
            cur = soln_lookup[cur.parent_sid]
            if cur.parent_sid and cur.parent_sid not in soln_lookup:
                logging.debug("omitting part of history in relevant sids for {} of {}".format(pdb.sid, pdb.uid))
                break
        backbone.append(cur)
        # add on the oldest pdb in backbone since it won't be included in the children of any backbone pdb
        relevant = sorted([c for c in sum([get_children(x, soln_lookup, child_lookup) for x in backbone], []) if
                           c.timestamp <= pdb.timestamp] + [backbone[-1]], key=lambda x: x.timestamp)
        return [x.sid for x in relevant]
    return None


def get_deltas(pdbs: List[PDB_Info], soln_lookup: Dict[str, PDB_Info]) -> List[SnapshotDelta]:
    """
    :return: list of SnapshotDeltas for each soloist PDB_Info in record
    """
    deltas = []
    for pdb in sorted(pdbs, key=lambda p: p.timestamp):
        if pdb.parent_sid:
            if pdb.parent_sid not in soln_lookup:
                logging.debug("no delta generated for {} of {}, parent pdb missing".format(pdb.sid, pdb.uid))
                continue
            parent = soln_lookup[pdb.parent_sid]
            actions, macros = collect_pdl_entries(pdb)
            pactions, pmacros = collect_pdl_entries(parent)
            action_diff = Counter({a: max(actions.get(a, 0) - pactions.get(a, 0), 0) for a in
                                   set(pactions.keys()).union(set(actions.keys()))})
            macro_diff = Counter({m: max(macros.get(m, 0) - pmacros.get(m, 0), 0) for m in
                                  set(pmacros.keys()).union(set(macros.keys()))})
            sd = SnapshotDelta(pdb.sid, pdb.parent_sid, pdb.timestamp, action_diff, macro_diff,
                               sum(action_diff.values()), pdb.energy - parent.energy)
        else:
            actions, macros = collect_pdl_entries(pdb)
            sd = SnapshotDelta(pdb.sid, None, pdb.timestamp, Counter(actions), Counter(macros), sum(actions.values()), 0)
        deltas.append(sd)
    return deltas


def compute_extra_data(record, soln_lookup, child_lookup, evolver=False):
    best_line = min(record.lines, key=lambda l: min(x.energy for x in l.pdb_infos)) if record.lines else None
    extra_data = {'pid': record.pid,
                  'uid': record.uid,
                  'relevant_sids': get_relevant_sids(min(best_line.pdb_infos, key=lambda x: x.energy, default=None),
                                                     soln_lookup, child_lookup) if best_line else None,
                  'deltas': get_deltas(best_line.pdb_infos if best_line else [], soln_lookup)}
    if evolver:
        extra_data["relevant_evol_sids"] = None
        extra_data["deltas_evol"] = None
        extra_data["evol_target_lines"] = None
        if record.evol_lines:
            by_source = {tag: list(lines) for tag, lines in groupby(sorted(record.evol_lines, key=lambda l: (l.source['header']['uid'], l.source['header']['score'])),
                                                                    lambda l: (l.source['header']['uid'], l.source['header']['score']))}
            target_lines = [min(lines, key=lambda l: min([x.energy for x in l.pdb_infos], default=np.inf)) for lines in by_source.values()]
            share_lines = [line for line in record.evol_lines if line not in target_lines and any(int(x.sharing_gid) > 1 for x in line.pdb_infos)]
            extra_data["relevant_evol_sids"] = [get_relevant_sids(min(line.pdb_infos, key=lambda x: x.energy, default=None),
                                                                  soln_lookup, child_lookup) for line in target_lines + share_lines]
            extra_data["deltas_evol"] = [get_deltas(line.pdb_infos if line else [], soln_lookup)
                                         for line in target_lines + share_lines]
            extra_data["evol_target_lines"] = target_lines + share_lines
    extra_data['relevant_time'] = np.nan
    if extra_data['relevant_sids']:
        extra_data['relevant_time'] = time_played(sorted([d.timestamp for d in extra_data['deltas'] if d.sid in extra_data['relevant_sids']]))
    return extra_data


def load_extend_data(pids: List[str], soln_lookup: dict, parent_lookup: dict, child_lookup: dict, 
                     evolver=False, min_time=3600) -> Tuple[pd.DataFrame, dict]:
    logging.debug("Loading data for {}".format(pids))
    data, puz_metas = load_data(pids, evolver, min_time)
    logging.debug("Creating lookups")
    make_lookups(data, soln_lookup, parent_lookup, child_lookup)

    logging.debug("Computing extra data")
    extra_data = [compute_extra_data(r, soln_lookup, child_lookup, evolver) for _, r in data.iterrows()]
    data = data.merge(pd.DataFrame(data=extra_data), on=['pid', 'uid'])
    data['perf'] = data.apply(
        lambda r: (min(0, min([p.energy for l in r.lines for p in l.pdb_infos]) - puz_metas[r.pid].energy_baseline) / (
                min(puz_metas[r.pid].pfront) - puz_metas[r.pid].energy_baseline)) if r.lines else np.nan, axis=1)
    if evolver:
        evolver_metas = {}
        best_energies = dict(data.groupby("pid").apply(lambda df: df.apply(lambda r: min(([p.energy for l in r.lines for p in l.pdb_infos] if r.lines else []) +
                                                                                         ([p.energy for l in r.evol_lines for p in l.pdb_infos] if r.evol_lines else []),
                                                                                         default=np.nan), axis=1).min()).iteritems())
        pfronts_all = dict(data.groupby("pid").apply(lambda df: np.minimum.accumulate([x.energy for x in sorted(sum(df.apply(lambda r: ([p for l in r.lines for p in l.pdb_infos] if r.lines else []) + ([p for l in r.evol_lines for p in l.pdb_infos] if r.evol_lines else []), axis=1).values, []), key=lambda p: p.timestamp)])).iteritems())

        for pid, energy in best_energies.items():
            evolver_metas[pid] = {"best": energy,
                                  "pfront": pfronts_all[pid]}
        puz_metas["evolver"] = evolver_metas
    # TODO compute efficiency
    """
    see compute_time.py for effort to pre-compute expensive time_played calls
    benchmarks = {pid: {x: [] for x in range(300, 3601, 300)} for pid in pids}
    for _, r in data[data.relevant_deltas.notnull()].iterrows():
        ds = sorted(r.relevant_deltas, key=lambda d: d.timestamp)
        ts = np.array([d.timestamp for d in ds])
        tp = np.array([time_played(ts[:i]) for i in range(1, len(ts) + 1)])
        for t in benchmarks[r.pid]:
            if tp[-1] >= t:
                s = soln_lookup[min(enumerate(ds), key=lambda x: abs(tp[x[0]] - t))[1].sid]
                benchmarks[r.pid][t].append(min(0, s.energy - puz_metas[s.pid].energy_baseline) / (puz_metas[s.pid].pfront.min() - puz_metas[s.pid].energy_baseline))
    """
    return data, puz_metas


def display_action_report(data: pd.DataFrame) -> None:
    """
    prints summary information about each action used in data
    :param data: DataFrame of solving data, must include deltas
    """
    actionset = reduce(set.union, [set(d.action_diff.keys()) for ds in data.deltas for d in ds])

    action_counts = {}
    for action in sorted(actionset):
        action_counts[action] = [[d.action_diff.get(action, 0) for d in ds] for ds in data.deltas]

    for action in sorted(actionset):
        print(action)
        counts = action_counts[action]
        counts_flat = sum(counts, [])
        print("active in {:.2f} percent of deltas".format(len([c for c in counts_flat if c > 0]) / len(counts_flat) * 100))
        print("active in {:.2f} percent of puzzles".format(len([cs for cs in counts if any(c > 0 for c in cs)]) / len(counts) * 100))
        print("median active instances per puzzle (where in use)", np.median([len([c for c in cs if c > 0]) for cs in counts if any(c > 0 for c in cs)]))
        active_counts = [c for c in counts_flat if c > 0]
        print("25th and 75th percentile rate when active", np.percentile(active_counts, 25), np.percentile(active_counts, 75))


def make_action_series(deltas: List[SnapshotDelta]) -> np.ndarray:
    s = []
    for d in deltas:
        ad = d.action_diff
        s.append(get_action_stream(ad))
    return np.array(s)


def make_series(data: pd.DataFrame, noise=None, default_noise_duration=200,
                min_snapshots=10) -> Tuple[Dict[Tuple[str, str], Tuple[int, int]], Dict[str, np.ndarray], np.ndarray]:
    """
    Construct an time series of action counts for each Foldit user in data using the 14 actions or action groups listed
    below
    :param data: DataFrame of solving data, must include deltas
    :param default_noise_duration: number of instances of random noise to insert between data from each puzzle
    :return: dictionary mapping (uid, pid) to start and end index of that puzzle's data in user's series, dictionary mapping uid to n-by-T numpy array, and noise values separating puzzles
    """
    num_features = 14
    if noise is None:
        noise = np.random.randint(50, 100, num_features)
    series = {}
    puz_idx_lookup = {}
    for uid, rows in data.groupby('uid'):
        s = []
        i = 1
        for _, r in rows.sort_values('pid').iterrows():
            if "relevant_evol_sids" in data and r.relevant_evol_sids:
                for evol_count, (sids, deltas) in enumerate(zip(r.relevant_evol_sids, r.deltas_evol)):
                    if sids and len(sids) >= min_snapshots:
                        relevant_deltas = sorted([d for d in deltas if d.sid in sids], key=lambda d: d.timestamp)
                        s.extend(make_action_series(relevant_deltas))
                        puz_idx_lookup[(uid+"evol"+str(evol_count), r.pid)] = (len(s) - len(relevant_deltas), len(s))
                        s.extend(np.tile(noise, (min(default_noise_duration, len(relevant_deltas)), 1)))
            if r.relevant_sids and len(r.relevant_sids) >= min_snapshots:
                relevant_deltas = sorted([d for d in r.deltas if d.sid in r.relevant_sids], key=lambda d: d.timestamp)
                s.extend(make_action_series(relevant_deltas))
                puz_idx_lookup[(uid, r.pid)] = (len(s) - len(relevant_deltas), len(s))
                if i < len(rows):
                    # don't add more noise points than there are real points
                    s.extend(np.tile(noise, (min(default_noise_duration, len(relevant_deltas)), 1)))
            i += 1
        if len(s) > 0:
            series[uid] = np.array(s)
    return puz_idx_lookup, series, noise


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
        idx_lookup[uid] = (len(all_series) + default_noise_duration, len(all_series) + default_noise_duration + len(ser))
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
        logging.debug("\n{} clusters".format(k))
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
                os.makedirs(datapath + "/{}".format(uid), exist_ok=True)
                np.savetxt(datapath + "/{}/clusters_k{}.txt".format(uid, k), clusters)
                with open(datapath + "/{}/mrfs_k{}.pickle".format(uid, k), 'wb') as pkl:
                    pickle.dump(mrfs, pkl)
                if save_model:
                    with open(datapath + "/{}/ticc_model_k{}.pickle".format(uid, k), 'wb') as pkl:
                        d = {
                            "trained_model": ticc.trained_model,
                            "window_size": ticc.window_size,
                            "num_blocks": ticc.num_blocks,
                            "number_of_clusters": ticc.number_of_clusters,
                            "switch_penalty": ticc.switch_penalty
                        }
                        pickle.dump(d, pkl)
                # if the user doesn't remove bics.csv in between multiple runs, there will be duplicate entries
                with open(datapath + "/{}/bics.csv".format(uid), 'a') as bic_out:
                    bic_out.write("{},{}\n".format(k, bic))
        ticc.pool.close()


def make_subseries_lookup(k: int, patterns: List[PatternInstance], mrfs: Dict[int, np.ndarray],
                          all_series: np.ndarray, noise: np.ndarray) -> Dict[int, Dict[int, Dict[str, Any]]]:
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
        logging.debug("assembling series for pattern {}".format(cid))
        sslu = {}
        sslu["patterns"] = {(p.uid, p.pid, p.start_idx): all_series[p.start_idx:p.end_idx] for p in patterns if p.cid == cid}
        if len(sslu["patterns"]) == 0:
            logging.debug("skipping pattern {} because there are no instances of it".format(cid))
            continue
        sub_idx_lookup, all_subseries = combine_user_series(sslu['patterns'], noise, default_noise_duration=100)
        # if len(all_subseries) - 100 * (len(sslu["patterns"]) - 1) < 20:
        #     logging.debug("skipping pattern {} because it only appears for {} timesteps".format(cid, len(all_subseries) - 100 * (len(sslu["patterns"]) - 1)))
        #     continue
        sslu["series"] = all_subseries
        sslu["idx_lookup"] = sub_idx_lookup
        lookup[cid] = sslu
    return lookup


def run_sub_TICC(subseries_lookup: dict, datapath: str, uid:str, sub_krange: list, save_model=True,
                 skip_series_fn=None, window_size=1, num_proc=4):
    results_dir = "{}/{}/subpatterns".format(datapath, uid)
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir + "/subseries_lookup.pickle", 'wb') as fp:
        pickle.dump(subseries_lookup, fp)
    for k in subseries_lookup:
        logging.debug("running TICC to get subpatterns for k={}".format(k))
        os.makedirs(results_dir + "/k{}".format(k), exist_ok=True)
        run_TICC({"cid{}".format(cid): lookup["series"] for cid, lookup in subseries_lookup[k].items()},
                 results_dir + "/k{}".format(k), sub_krange, save_model, skip_series_fn, window_size, num_proc)


def mrf_diff(a: np.ndarray, b: np.ndarray) -> float:
    ac = a[np.triu_indices(a.shape[1])]
    bc = b[np.triu_indices(b.shape[1])]

    return ((ac - bc)**2).sum()


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
        if os.path.exists(datapath + "/" + uid) and any(os.path.exists(datapath + "/{}/clusters_k{}.txt".format(uid, k)) for k in krange):
            cd = {}
            md = {}
            modd = {}
            for k in krange:
                if os.path.exists(datapath + "/{}/clusters_k{}.txt".format(uid, k)):
                    cd[k] = np.loadtxt(datapath + "/{}/clusters_k{}.txt".format(uid, k), np.int32)
                    with open(datapath + "/{}/mrfs_k{}.pickle".format(uid, k), 'rb') as fp:
                        md[k] = pickle.load(fp)
                    with open(datapath + "/{}/ticc_model_k{}.pickle".format(uid, k), 'rb') as fp:
                        modd[k] = pickle.load(fp)
            cluster_lookup[uid] = cd
            mrf_lookup[uid] = md
            model_lookup[uid] = modd
            with open(datapath + "/{}/bics.csv".format(uid)) as fp:
                bic_in = csv.DictReader(fp, fieldnames=["k", "bic"])
                bics = {}
                # guard against duplicate entries by having the most recent one take precedence
                for row in bic_in:
                    bics[int(row["k"])] = float(row["bic"])
                bic_lookup[uid] = bics
        else:
            # logging.warning("TICC output for uid {} not found".format(uid))
            raise FileNotFoundError("TICC output for uid {} not found".format(uid))
    return cluster_lookup, mrf_lookup, model_lookup, bic_lookup


# noinspection PyTypeChecker
def mean_cluster_nearest_neighbor_distance(mrfs: Dict[int, np.ndarray]) -> float:
    return np.mean([min([mrf_diff(mrfs[ci], mrfs[cj]) for cj in mrfs if ci != cj]) for ci in mrfs])


def mean_pattern_cohesion(mrfs, patterns, series):
    cluster_instance_lookup = {cid: list(g) for cid, g in groupby(sorted(patterns, key=lambda p: p.cid), lambda p: p.cid)}
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
        no_dups = {k for k in cd if not any(any((m == md[k][ci]).all() for cj, m in md[k].items() if cj > ci) for ci in range(k))}
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
            logging.info("unable to select TICC model for uid {}, k with no dups={}, k that handled noise={}".format(uid, no_dups, handles_noise))
    return best_k


def get_combined_timestamps(df: pd.DataFrame) -> np.ndarray:
    count = 0
    offset = 0
    td_combined = []
    for _, r in df.iterrows():
        deltas = sorted([d for d in r.deltas if d.sid in r.relevant_sids], key=lambda x: x.timestamp)
        ts = np.array([d.timestamp for d in deltas])
        td_aligned = align_timestamps(ts - ts[0])

        if count < len(df) - 1:
            td_aligned = np.concatenate((td_aligned,
                                         np.linspace(td_aligned[-1], td_aligned[-1] + 10000, min(200, len(td_aligned)),
                                                     dtype=np.int32)))
        td_aligned += offset
        td_combined.extend(td_aligned.tolist())
        offset = td_combined[-1]
        count += 1
    return np.array(td_combined)


def get_pattern_masks(uid, pid, idx, target_pts, sub_ks, pattern_lookup, subseries_lookup):
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


def plot_labeled_series(x: np.ndarray, y: np.ndarray, pattern_masks: Dict[str, np.ndarray], action_labels: List[str],
                        filename: str, scale="log", figsize=(100, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    for j in range(y.shape[1]):
        ax.plot(x, y[:, j], color=plt.cm.get_cmap("tab20").colors[j], alpha=0.7)
    line_legend = ax.legend(action_labels, loc="upper center", ncol=7, fancybox=True, shadow=True, bbox_to_anchor=(0.5, 1.05))
    colors = plt.cm.get_cmap("viridis", len(pattern_masks)).colors
    fills = []
    pattern_labels = []
    for i, (pt, mask) in enumerate(pattern_masks.items()):
        if mask.any():
            fills.append(ax.fill_between(x, 0, y.max(), mask, color=colors[i], alpha=0.4))
            pattern_labels.append(pt)
            if mask[0]:
                ax.text(-2, y.max(), pt)
            for i, (a, b) in enumerate(zip(mask, mask[1:])):
                if b and not a:
                    ax.text(i, y.max(), pt)
    ax.set_xlim(-10, len(x) + 10)
    if len(pattern_masks) > 0:
        ax.legend(fills, pattern_labels, bbox_to_anchor=(1.01, 1), fancybox=True, shadow=True)
    ax.add_artist(line_legend)
    ax.set_yscale(scale, nonposy='clip')
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_model(model_dir, k, subs, all_series, pattern_lookup, subseries_lookup, action_labels):
    for cid, sub_k in subs:
        print("plotting", cid, sub_k)
        os.makedirs("{}/eval/viz/".format(model_dir), exist_ok=True)
        if sub_k == 0:
            ps = [p for p in pattern_lookup[k][cid][0] if p.start_idx < p.end_idx]
            ser = np.concatenate([np.concatenate((all_series[p.start_idx:p.end_idx],
                                                  np.tile(np.array([0]*all_series.shape[1]), (60, 1)))) for p in ps])
            plot_labeled_series(np.arange(len(ser)), ser, {}, action_labels,
                                "{}/eval/viz/{}_pattern.png".format(model_dir, cid), figsize=(200, 6))
        else:
            subpatterns = pattern_lookup[k][cid][sub_k]
            subcids = {p.cid for p in subpatterns}
            for sub_cid in subcids:
                label = str(cid) + string.ascii_uppercase[sub_cid]
                sub_ps = [p for p in subpatterns if p.cid == sub_cid and p.start_idx < p.end_idx]
                ser = np.concatenate([np.concatenate((subseries_lookup[k][cid]["series"][p.start_idx:p.end_idx],
                                                      np.tile(np.array([0]*all_series.shape[1]), (30, 1)))) for p in sub_ps])
                plot_labeled_series(np.arange(len(ser)), ser, {}, action_labels,
                                    "{}/eval/viz/{}_pattern.png".format(model_dir, label))


def plot_user_series(model_dir, k, subs, puz_idx_lookup, all_series, pattern_lookup, pts, subseries_lookup, action_labels):
    os.makedirs("{}/eval/user_series/".format(model_dir), exist_ok=True)
    for i, ((uid, pid), idx) in enumerate(puz_idx_lookup.items()):
        print("plotting user {} of {}\r".format(i, len(puz_idx_lookup)), end="")
        masks = get_pattern_masks(uid, pid, idx, pts, dict(subs), pattern_lookup[k], subseries_lookup[k])
        y = all_series[slice(*idx)]
        if len(y) > 1:
            plot_labeled_series(np.arange(len(y)), y, masks, action_labels,
                                "{}/eval/user_series/{}_{}.png".format(model_dir, uid, pid))
    print()


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
            if end == len(clusters) - 1: # put end one past the last index so the slice actually grabs the last point
                end += 1
            patterns.append(PatternInstance(cid, uid, pid, start, end))
    return patterns


def get_patterns_lookup(krange, sub_clusters, sub_mrfs, subseries_lookup, cluster_lookup, mrf_lookup, puz_idx_lookup):
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
                patterns[k][cid][sub_k] = get_patterns(sub_mrfs[k][cid][sub_k], sub_clusters[k][cid][sub_k], sub_idx_lookup)
    return patterns


def get_patterns_with_pdbs(data: pd.DataFrame, mrfs: Dict[int, np.ndarray], clusters: np.ndarray, 
                 puz_idx_lookup: dict, soln_lookup: dict) -> List[PatternInstanceExt]:
    patterns = []
    for cid in mrfs:
        if is_null_cluster(mrfs[cid]):
            continue
        starts = [j for j, c in enumerate(clusters[:-1], 1) if clusters[j] != c and clusters[j] == cid]
        if clusters[0] == cid:
            starts = [0] + starts
        for start in starts:
            (uid, pid), ran = next((key, range(*r)) for key, r in puz_idx_lookup.items() if start in range(*r))
            end = next(i for i, c in enumerate(clusters[start:], start) if c != cid or i + 1 == len(clusters))
            assert start < end
            assert all(c == cid for c in clusters[start:end])
            #relevant_sids = get_relevant_sids(
            #    min(get_data_value(uid, pid, "lines", data), key=lambda l: min(x.energy for x in l.pdb_infos)),
            #    soln_lookup, child_lookup)
            relevant_sids = get_data_value(uid, pid, "relevant_sids", data)
            relevant_pdbs = sorted([soln_lookup[sid] for sid in relevant_sids], key=lambda s: s.timestamp)
            assert abs(ran.stop - ran.start - len(relevant_pdbs)) <= 1
            assert end - ran.start <= len(relevant_pdbs)
            start_pdb = relevant_pdbs[start - ran.start]
            end_pdb = relevant_pdbs[end - ran.start - 1]
            pre_best = min(relevant_pdbs[:start - ran.start], key=lambda s: s.energy, default=None)
            post_best = min(relevant_pdbs[:end - ran.start], key=lambda s: s.energy)
            patterns.append(PatternInstanceExt(cid, uid, pid, start, end, start_pdb, end_pdb, pre_best, post_best))
    return patterns


if __name__ == "__main__":
    pids = ['2003433', '2003465', '2003490', '2003195', '2003240', '2003206', '2003483',
            '2003583']
    # denovo: '2003433', '2003111', '2003465', '2003490', '2003195'
    # revisiting: '2003240', '2003206', '2003125', '2003483', '2003583'

    parser = argparse.ArgumentParser(prog='pattern_extraction.py')
    parser.add_argument("datapath")
    parser.add_argument('--evolver', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--noplot', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--pids', nargs='+')
    parser.add_argument('--num-patterns', nargs='+', type=int)
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("setting krange to {}".format(args.num_patterns))

    if os.path.exists(args.datapath) and not args.overwrite:
        logging.error("{} exists and no overwrite flag given".format(args.datapath))
        sys.exit(1)

    if not os.path.exists(args.datapath):
        os.makedirs(args.datapath)

    if args.pids:
        pids = args.pids

    krange = range(10, 11)
    if args.num_patterns:
        krange = args.num_patterns

    with open(args.datapath + "/config.json") as fp:
        json.dump({"pids": pids,
                   "evolver": args.evolver,
                   "krange": krange}, fp)

    soln_lookup = {}
    parent_lookup = {}
    child_lookup = {}
    data, puz_metas = load_extend_data(pids, soln_lookup, parent_lookup, child_lookup, args.evolver, 600)

    logging.debug("Constructing time series")
    puz_idx_lookup, series_lookup, noise = make_series(data)
    num_features = next(x for x in series_lookup.values()).shape[1]
    # filtered_series_lookup = {uid: ser for uid, ser in series_lookup.items() if
    #                           len(ser) < 200 * len(pids) or len(data[data.uid.isin([uid])].pid.unique()) < len(pids) // 2}
    idx_lookup, all_series = combine_user_series(series_lookup, noise)
    puz_idx_lookup = {(uid, pid): (s + idx_lookup[uid][0], e + idx_lookup[uid][0]) for (uid, pid), (s, e) in puz_idx_lookup.items()}
    np.savetxt(args.datapath + "/noise_values.txt", noise)
    np.savetxt(args.datapath + "/all_series.txt", all_series)
    with open(args.datapath + "/puz_idx_lookup.pickle", 'wb') as fp:
        pickle.dump(puz_idx_lookup, fp)
    with open(args.datapath + "/idx_lookup.pickle", 'wb') as fp:
        pickle.dump(idx_lookup, fp)

    if args.evolver:
        evol_series_lookup = {}
        for _, r in data.iterrows():
            if r.evol_lines:
                for idx, line in enumerate(r.evol_lines):
                    if len(line.pdb_infos) <= 1 or line.pdb_infos[0].parent_sid not in soln_lookup:
                        continue
                    deltas = sorted(get_deltas(sorted(line.pdb_infos, key=lambda p: p.timestamp), soln_lookup), key=lambda x: x.timestamp)
                    if len(deltas) > 0 and time_played([d.timestamp for d in deltas]) > 600:
                        ser = make_action_series(deltas)
                        evol_series_lookup["{}_{}_{}".format(r.uid, r.pid, idx)] = ser
        evol_idx_lookup, evol_all_series = combine_user_series(evol_series_lookup, noise)
        np.savetxt(args.datapath + "/evol_all_series.txt", evol_all_series)
        with open(args.datapath + "/evol_idx_lookup.pickle", 'wb') as fp:
            pickle.dump(evol_idx_lookup, fp)

    logging.debug("Running TICC")
    run_TICC({"all": all_series}, args.datapath, krange)
    if args.evolver:
        run_TICC({"all_evol": evol_all_series}, args.datapath, krange)

    if args.noplot:
        sys.exit(0)

    logging.debug("Loading TICC output")
    cluster_lookup, mrf_lookup, model_lookup, bic_lookup = load_TICC_output(args.datapath, ["all", "all_evol"], krange)
    uid = "all"
    for k in mrf_lookup[uid]:
        mrf_lookup[uid][k] = weight_mrfs_by_ubiquity(mrf_lookup[uid][k], all_series, cluster_lookup[uid][k])
    uid = "all_evol"
    for k in mrf_lookup[uid]:
        mrf_lookup[uid][k] = weight_mrfs_by_ubiquity(mrf_lookup[uid][k], evol_all_series, cluster_lookup[uid][k])

    logging.debug("Selecting models")
    best_k = select_TICC_model(cluster_lookup, mrf_lookup, {"all": all_series}, noise,
                               mean_cluster_nearest_neighbor_distance, False)

    logging.debug("Plotting")
    k = best_k["all"]
    all_clusters = cluster_lookup["all"][k]
    action_labels = get_action_labels()
    pattern_labels = list(range(k))
    for uid in series_lookup:
        ser = series_lookup[uid]
        cs = all_clusters[slice(*idx_lookup[uid])]
        plot_labeled_series(np.arange(len(ser)), ser, cs, action_labels, pattern_labels,
                            "{}/all/{}_series_all_k{}.png".format(args.datapath, uid, k))

    patterns = get_patterns(data, mrf_lookup["all"][k], all_clusters, puz_idx_lookup)

    cluster_times = []
    for uid, ser in series_lookup.items():
        # print(uid)
        sample = data.groupby('uid').get_group(uid).sort_values('pid')
        for _, r in sample.iterrows():
            if r.relevant_sids is None or len(r.relevant_sids) == 0:
                continue
            # print("   ", r.pid)
            deltas = sorted([d for d in r.deltas if d.sid in r.relevant_sids], key=lambda x: x.timestamp)
            ts = np.array([d.timestamp for d in deltas])
            puz_cs = all_clusters[slice(*puz_idx_lookup[(uid, r.pid)])]
            assert len(ts) == len(puz_cs)
            result_dict = {'uid': uid, 'pid': r.pid}
            for ci in range(k):
                result_dict["cluster_{}_time".format(ci)] = time_played(ts[puz_cs == ci])
            cluster_times.append(result_dict)

    results = data.merge(pd.DataFrame(data=cluster_times), on=['pid', 'uid'])

    """
    all_timestamps = np.zeros(len(all_series))
    for (uid, pid), (s, e) in puz_idx_lookup.items():
        # print(uid, pid)
        sids = get_data_value(uid, pid, "relevant_sids", data)
        ts = sorted([d.timestamp for d in get_data_value(uid, pid, "deltas", data) if d.sid in sids])
        assert len(ts) == e - s
        all_timestamps[s:e] = ts


    def get_pattern_improve(p):
        return min(0, p.post_best.energy - (
            p.pre_best.energy if p.pre_best and p.pre_best.energy < puz_metas[p.pid].energy_baseline else puz_metas[
                p.pid].energy_baseline)) / (puz_metas[p.pid].pfront.min() - puz_metas[p.pid].energy_baseline)


    sequences = {(uid, pid): sorted(g, key=lambda p: p.start_idx) for (uid, pid), g in
                 groupby(sorted(patterns, key=lambda p: (p.uid, p.pid)), lambda p: (p.uid, p.pid))}


    def make_ticc_from_saved_model(saved_model):
        test_ticc = TICC(window_size=saved_model["window_size"], number_of_clusters=saved_model["number_of_clusters"], num_proc=1)
        test_ticc.num_blocks = saved_model["num_blocks"]
        test_ticc.switch_penalty = saved_model["switch_penalty"]
        test_ticc.trained_model = saved_model["trained_model"]
        return test_ticc


    def get_labels(deltas, soln_lookup, radius=5):
        labels = ['r']
        for i, x in enumerate(deltas[1:]):
            window = [a.parent_sid for a in deltas[max(i - radius, 0): i + radius + 1]]
            starts = [window[0]] + [p for j, p in enumerate(window[1:]) if window[j] != p]
            if len(set(starts)) < len(starts): # multiple, non-consecutive instances of at least one parent in the window indicates growing multiple parts of the tree
                labels.append('m')
            elif x.parent_sid == deltas[i].sid:
                labels.append('d')
            elif x.parent_sid == deltas[i].parent_sid:
                labels.append('b')
            else:
                if labels[i] == 'b' and soln_lookup[x.parent_sid].parent_sid == deltas[i].parent_sid:
                    labels.append('d')
                else:
                    labels.append('o')
        return labels


    sequences = {(uid, pid): sorted(g, key=lambda p: p.start_idx) for (uid, pid), g in
             groupby(sorted(patterns, key=lambda p: (p.uid, p.pid)), lambda p: (p.uid, p.pid))}
    grams = []
    for seq in sequences.values():
        for l in range(2, len(seq)):
            for i in range(len(seq) - l + 1):
                grams.append(seq[i:i + l])
    gram_map = {gram: list(seqs) for gram, seqs in groupby(sorted(grams, key=lambda g: "".join([str(p.cid) for p in g])), lambda g: "".join([str(p.cid) for p in g]))}
    {k: np.mean([sum(get_pattern_improve(p) for p in seq) / time_played(all_timestamps[seq[0].start_idx:seq[-1].end_idx]) for seq in seqs if seq[0].start_idx != puz_idx_lookup[(seq[0].uid, seq[0].pid)][0]]) for k, seqs in gram_map.items() if len(k) == 4} # k is length of the sequence, != for non-start sequences

    subseries_lookup = {}
    for k in krange:
        logging.debug("getting patterns for k={}".format(k))
        patterns = get_patterns(data, mrf_lookup["all"][k], cluster_lookup["all"][k], puz_idx_lookup, soln_lookup)
        if not os.path.exists(args.datapath + "/subpatterns_k{}".format(k)):
            os.makedirs(args.datapath + "/subpatterns_k{}".format(k))
        subseries_lookup.setdefault(k, {})
        for cid in range(k):
            if is_null_cluster(mrf_lookup["all"][k][cid]) or cid in subseries_lookup[k]:
                continue
            logging.debug("assembling series for pattern {}".format(cid))
            sslu = {}
            sslu["patterns"] = {(p.uid, p.pid, p.start_idx): all_series[p.start_idx:p.end_idx] for p in patterns if p.cid == cid}
            sub_idx_lookup, all_subseries = combine_user_series(sslu['patterns'], noise, default_noise_duration=100)
            if len(all_subseries) - 100 * (len(sslu["patterns"]) - 1) < 20:
                logging.debug("skipping pattern {} because it only appears for {} timesteps".format(cid, len(all_subseries) - 100 * len(sslu["patterns"])))
                continue
            sslu["series"] = all_subseries
            sslu["idx_lookup"] = sub_idx_lookup
            subseries_lookup[k][cid] = sslu
            os.makedirs(args.datapath + "/subpatterns_k{}/subpatterns{}".format(k, cid))
            with open(args.datapath + "/subpatterns_k{}/subpatterns{}/sslu.pickle".format(k, cid), 'wb') as fp:
                pickle.dump(sslu, fp)
            run_TICC({"subpatterns{}".format(cid): all_subseries}, args.datapath + "/subpatterns_k{}".format(k), [3, 6, 9, 12])


    sub_cluster_lookup, sub_mrf_lookup, sub_model_lookup, sub_bic_lookup = load_TICC_output(args.datapath, ["subpatterns_1", "subpatterns_2", "subpatterns_3", "subpatterns_4"], krange)

    subpattern_lookup = {}
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for p in patterns:
        sp = subpattern_lookup.setdefault(p.uid, {})
        cs = sub_cluster_lookup['subpatterns_{}'.format(p.cid)][10][slice(*subseries_lookup[p.cid]['idx_lookup'][(p.uid, p.pid, p.start_idx)])]
        change_idx = np.array([0] + [i for i in range(1, len(cs)) if cs[i] != cs[i - 1]])
        subs = []
        offset = p.start_idx - idx_lookup[p.uid][0]
        for start in change_idx:
            end = next(i for i, c in enumerate(cs[start:], start) if c != cs[start] or i + 1 == len(cs))
            subs.append((offset + start, offset + end, p, labels[cs[start]]))
        sp[p.start_idx - idx_lookup[p.uid][0]] = subs

    for cid in range(k):
        if is_null_cluster(mrf_lookup["all"][k][cid]):
            print("null cluster", cid)
            continue
        print("cluster", cid)
        print("    used by {:.2%} of users".format(len([uid for uid, rows in results.groupby('uid') if any(r["cluster_{}_time".format(cid)] > 0 for _, r in rows.iterrows())]) / len(results.uid.unique())))
        users = [rows for uid, rows in results.groupby('uid') if any(r["cluster_{}_time".format(cid)] > 0 for _, r in rows.iterrows())]
        if len(users) > 0:
            print("    of its users, {:.2%} use it on every puzzle".format(len([rows for rows in users if all(r["cluster_{}_time".format(cid)] > 0 for _, r in rows.iterrows())]) / len(
            users)))
            ps = [p for p in patterns if p.cid == cid and p.start_idx > 0]
            print("    {} instances in the dataset".format(len(ps)))
            print("    {:.4f} mean duration".format(np.mean([time_played(all_timestamps[p.start_idx:p.end_idx]) for p in ps])))
            print("    {:.4f} median improve".format(np.median([get_pattern_improve(p) for p in ps])))
            print("    {:.4e} median improve rate".format(np.nanmedian([get_pattern_improve(p) / (time_played(all_timestamps[p.start_idx:p.end_idx]) if p.start_idx + 1 < p.end_idx else 300) for p in ps])))
            for prev_cid, g in groupby(sorted(ps, key=lambda p: all_clusters[p.start_idx - 1]), lambda p: all_clusters[p.start_idx - 1]):
                xs = list(g)
                print("    mean improve of {:.4f} when preceeded by {} (n={})".format(np.mean([get_pattern_improve(x) for x in xs]), prev_cid, len(xs)))
            print("subpatterns")
            for sub_cid in sub_mrf_lookup["subpatterns_{}".format(cid)][10]:
                if is_null_cluster(sub_mrf_lookup["subpatterns_{}".format(cid)][10][sub_cid]):
                    print("    null subpattern", sub_cid)
                    continue
                print("    subpattern", labels[sub_cid])
                sub_users = {(p.uid, p.pid) for d in subpattern_lookup.values() for sub_ps in d.values() for si, ei, p, label in sub_ps if p.cid == cid and label == labels[sub_cid]}
                print(max(sub_users, key=lambda x: get_data_value(x[0], x[1], "perf", results)))
                non_sub_users = {(r.uid, r.pid) for rows in users for _, r in rows.iterrows() if (r.uid, r.pid) not in sub_users}
                sig_test([get_data_value(uid, pid, "perf", results) for uid, pid in sub_users], [get_data_value(uid, pid, "perf", results) for uid, pid in non_sub_users])
        print()
        sig_test([rows.perf.median() for uid, rows in results.groupby('uid') if any(r["cluster_{}_time".format(cid)] > 0 for _, r in rows.iterrows())], [rows.perf.median() for uid, rows in results.groupby('uid') if all(r["cluster_{}_time".format(cid)] == 0 for _, r in rows.iterrows())])


    def predict_from_saved_model(test_data, saved_model):
        test_ticc = TICC(window_size=saved_model["window_size"], number_of_clusters=saved_model["number_of_clusters"], num_proc=1)
        test_ticc.num_blocks = saved_model["num_blocks"]
        test_ticc.switch_penalty = saved_model["switch_penalty"]
        test_ticc.trained_model = saved_model["trained_model"]
        cs = test_ticc.predict_clusters(test_data)
        test_ticc.pool.close()
        return cs

    subpatterns = [SubPatternInstance(p, label, si + idx_lookup[uid][0], ei + idx_lookup[uid][0]) for uid, d in subpattern_lookup.items() for sub_ps in d.values() for si, ei, p, label in sub_ps]
    subsequences = {(uid, pid): sorted(g, key=lambda sp: sp.start_idx) for (uid, pid), g in groupby(sorted(subpatterns, key=lambda sp: (sp.p.uid, sp.p.pid)), lambda sp: (sp.p.uid, sp.p.pid))}

    all_subclusters = all_clusters.astype(np.str)
for p in patterns:
    all_subclusters[p.start_idx:p.end_idx] = ["{}{}".format(p.cid, labels[x]) for x in sub_cluster_lookup["subpatterns_{}".format(p.cid)][10][slice(*subseries_lookup[p.cid]['idx_lookup'][(p.uid, p.pid, p.start_idx)])]]

subcluster_times = []
for _, r in data.iterrows():
    if r.relevant_sids is None or len(r.relevant_sids) == 0:
        continue
    deltas = sorted([d for d in r.deltas if d.sid in r.relevant_sids], key=lambda x: x.timestamp)
    ts = np.array([d.timestamp for d in deltas])
    puz_cs = all_subclusters[slice(*puz_idx_lookup[(r.uid, r.pid)])]
    assert len(ts) == len(puz_cs)
    result_dict = {'uid': r.uid, 'pid': r.pid}
    for ci in np.unique(all_subclusters):
        result_dict["subcluster_{}_time".format(ci)] = time_played(ts[puz_cs == ci])
    subcluster_times.append(result_dict)

results = data.merge(pd.DataFrame(data=subcluster_times), on=['pid', 'uid'])
"""
