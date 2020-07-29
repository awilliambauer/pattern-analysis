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


def load_data(pids: List[str], evolver=False, min_time=3600, data_dir="../data") -> Tuple[pd.DataFrame, dict]:
    datafiles = [f"{data_dir}/puzzle_{pid}/{pid}_meta.h5" for pid in pids]
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
    infos = data.apply(lambda r: sorted(r.foldit.solo_pdbs + r.foldit.evol_pdbs, key=lambda p: p.timestamp), axis=1)
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
            sd = SnapshotDelta(pdb.sid, None, pdb.timestamp, Counter(actions),
                               Counter(macros), sum(actions.values()), 0)
        deltas.append(sd)
    return deltas


def compute_extra_data(record, soln_lookup, child_lookup, evolver=False):
    best_line = min(record.lines, key=lambda l: min(l.energies)) if record.lines else None
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
            def fn(l): return (l.source['header']['uid'], l.source['header']['score'])
            by_source = {tag: list(lines) for tag, lines in groupby(sorted(record.evol_lines, key=fn), fn)}
            target_lines = [min(lines, key=lambda l: min(l.energies, default=np.inf)) for lines in by_source.values()]
            share_lines = [line for line in record.evol_lines if
                           line not in target_lines and any(int(x.sharing_gid) > 1 for x in line.pdb_infos)]
            def fn(line): return get_relevant_sids(min(line.pdb_infos, key=lambda x: x.energy, default=None),
                                                   soln_lookup, child_lookup)
            extra_data["relevant_evol_sids"] = [fn(line) for line in target_lines + share_lines]
            extra_data["deltas_evol"] = [get_deltas(line.pdb_infos if line else [], soln_lookup)
                                         for line in target_lines + share_lines]
            extra_data["evol_target_lines"] = target_lines + share_lines
    extra_data['relevant_time'] = np.nan
    if extra_data['relevant_sids']:
        extra_data['relevant_time'] = time_played(sorted([d.timestamp for d in extra_data['deltas']
                                                          if d.sid in extra_data['relevant_sids']]))
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
        lambda r: (min(0, min(r.foldit.solo_energies) - puz_metas[r.pid].energy_baseline) / (
                min(puz_metas[r.pid].pfront) - puz_metas[r.pid].energy_baseline)) if r.lines else np.nan, axis=1)
    if evolver:
        evolver_metas = {}
        def fn(df): return (df.foldit.solo_energies + df.foldit.evol_energies).apply(lambda r: min(r, default=np.nan)).min()
        best_energies = dict(data.groupby("pid").apply(fn).iteritems())
        def pdbs_to_frontier(pdbs):
            return np.minimum.accumulate([x.energy for x in sorted(pdbs, key=lambda p: p.timestamp)])
        def fn(df):
            return pdbs_to_frontier(sum((df.foldit.solo_pdbs + df.foldit.evol_pdbs).values, []))
        pfronts_all = dict(data.groupby("pid").apply(fn).iteritems())

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
        print("active in {:.2f} percent of deltas".format(
            len([c for c in counts_flat if c > 0]) / len(counts_flat) * 100))
        print("active in {:.2f} percent of puzzles".format(
            len([cs for cs in counts if any(c > 0 for c in cs)]) / len(counts) * 100))
        print("median active instances per puzzle (where in use)",
              np.median([len([c for c in cs if c > 0]) for cs in counts if any(c > 0 for c in cs)]))
        active_counts = [c for c in counts_flat if c > 0]
        print("25th and 75th percentile rate when active", np.percentile(active_counts, 25),
              np.percentile(active_counts, 75))


def make_action_series(deltas: List[SnapshotDelta]) -> np.ndarray:
    """

    :param deltas:
    :return:
    """
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
                        puz_idx_lookup[(f"{uid}evol{evol_count}", r.pid)] = (len(s) - len(relevant_deltas), len(s))
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


# def get_patterns_with_pdbs(data: pd.DataFrame, mrfs: Dict[int, np.ndarray], clusters: np.ndarray,
#                            puz_idx_lookup: dict, soln_lookup: dict) -> List[PatternInstanceExt]:
#     patterns = []
#     for cid in mrfs:
#         if is_null_cluster(mrfs[cid]):
#             continue
#         starts = [j for j, c in enumerate(clusters[:-1], 1) if clusters[j] != c and clusters[j] == cid]
#         if clusters[0] == cid:
#             starts = [0] + starts
#         for start in starts:
#             (uid, pid), ran = next((key, range(*r)) for key, r in puz_idx_lookup.items() if start in range(*r))
#             end = next(i for i, c in enumerate(clusters[start:], start) if c != cid or i + 1 == len(clusters))
#             assert start < end
#             assert all(c == cid for c in clusters[start:end])
#             # relevant_sids = get_relevant_sids(
#             #    min(get_data_value(uid, pid, "lines", data), key=lambda l: min(x.energy for x in l.pdb_infos)),
#             #    soln_lookup, child_lookup)
#             relevant_sids = get_data_value(uid, pid, "relevant_sids", data)
#             relevant_pdbs = sorted([soln_lookup[sid] for sid in relevant_sids], key=lambda s: s.timestamp)
#             assert abs(ran.stop - ran.start - len(relevant_pdbs)) <= 1
#             assert end - ran.start <= len(relevant_pdbs)
#             start_pdb = relevant_pdbs[start - ran.start]
#             end_pdb = relevant_pdbs[end - ran.start - 1]
#             pre_best = min(relevant_pdbs[:start - ran.start], key=lambda s: s.energy, default=None)
#             post_best = min(relevant_pdbs[:end - ran.start], key=lambda s: s.energy)
#             patterns.append(PatternInstanceExt(cid, uid, pid, start, end, start_pdb, end_pdb, pre_best, post_best))
#     return patterns
