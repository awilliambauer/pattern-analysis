import csv
import random
import numpy as np
from collections import Counter
import json
from itertools import groupby, chain, combinations, takewhile, product
from functools import partial
from util import get_atoms, weighted_rmsd, EnergyComponent, skip_pids, iden, get_sessions, get_time_splits, PDB_Info, \
    session_duration, get_children, ROOT_NID, get_nid, output_atoms, SolvingLine, tmscore, SolvingLineVariant, rmsd,\
    EvolvingLine, PuzzleMeta
from collab_viz import is_corrupted
from datetime import datetime, timedelta
from operator import itemgetter
import scipy
import scipy.spatial
import scipy.signal
import scipy.stats as stats
import scipy.cluster
import pandas as pd
import os
import sys
import subprocess
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import re
import argparse
import logging
import dill
from typing import List


def expand_breakthrough(bt, frontier_changes, timestamps, pause_threshold, energy_threshold, rate_threshold):
    pre_offset = post_offset = 0
    pre_improve = frontier_changes[bt - pre_offset] - frontier_changes[bt - pre_offset - 1]
    time_gap = timestamps[bt - pre_offset] - timestamps[bt - pre_offset - 1]
    while bt - pre_offset - 2 >= 0 and frontier_changes[
        bt - pre_offset - 1] < energy_threshold and pre_improve / time_gap < rate_threshold and time_gap < pause_threshold:
        pre_offset += 1
        pre_improve = frontier_changes[bt - pre_offset] - frontier_changes[bt - pre_offset - 1]
        time_gap = timestamps[bt - pre_offset] - timestamps[bt - pre_offset - 1]

    if bt + post_offset + 1 < len(frontier_changes):
        post_improve = frontier_changes[bt + post_offset + 1] - frontier_changes[bt + post_offset]
        time_gap = timestamps[bt + post_offset + 1] - timestamps[bt + post_offset]
        while bt + post_offset + 2 < len(frontier_changes) and frontier_changes[
            bt + post_offset + 1] < energy_threshold and post_improve / time_gap < rate_threshold and time_gap < pause_threshold:
            post_offset += 1
            post_improve = frontier_changes[bt + post_offset + 1] - frontier_changes[bt + post_offset]
            time_gap = timestamps[bt + post_offset + 1] - timestamps[bt + post_offset]
    return bt - pre_offset, bt + post_offset


def find_breakthroughs(lines: List[SolvingLine], energy_threshold_frac, rate_threshold, diff_threshold, tm_threshold):
    ret = []
    for line in lines:
        timediffs = np.array([s.timestamp - line.pdb_infos[0].timestamp for s in line.pdb_infos])
        energies = np.array([s.energy if s.energy < 0 else 0 for s in line.pdb_infos])
        scores = np.array([s.tmscore for s in line.pdb_infos])
        energy_threshold = min(energies) * energy_threshold_frac
        frontier = np.minimum.accumulate(energies)
        frontier_change_idx = np.array([i for i, f in enumerate(frontier[1:], 1) if frontier[i - 1] > f])
        if len(frontier_change_idx) < 2:
            return []
        improves = np.gradient(frontier, timediffs)[frontier_change_idx]

        bt_candidates = np.concatenate((scipy.signal.argrelmin(improves)[0],
                                        np.where(np.diff(frontier[frontier_change_idx]) < diff_threshold)[0] + 1))
        bts = sorted({bt for bt in bt_candidates if
                      improves[bt] < rate_threshold and frontier[frontier_change_idx][bt] < energy_threshold and
                      frontier[frontier_change_idx][bt] - frontier[frontier_change_idx][bt - 1] <= diff_threshold and
                      scores[frontier_change_idx][bt] < tm_threshold})
        regions = []
        for bt in bts:
            if len(regions) == 0 or bt > regions[-1][1]:
                regions.append(expand_breakthrough(bt, frontier[frontier_change_idx],
                                                   timediffs[frontier_change_idx],
                                                   600, energy_threshold, rate_threshold))
        ret.append([(frontier_change_idx[start], frontier_change_idx[end]) for start, end in regions])
    return ret


# def process_snapshots(ss, delta, breakthrough_params, parent_lookup, child_lookup, nid_to_sid, get_tmscore_pkl):
def process_snapshots(ss, delta, breakthrough_params, parent_lookup, child_lookup, nid_to_sid):
    pid = ss[0]['pid']
    uid = ss[0]['uid']
    logging.debug("{} {} start".format(pid, uid))
    sys.stdout.flush()
    ret = {}
    ss.sort(key=lambda s: s['timestamp'])
    solo = np.array([s for s in ss if s['scoretype'] == '1'])
    soln_lookup = {get_nid(s): s for s in solo}  # restrict to solo to try and squash missing atom file bugs
    evol = np.array([s for s in ss if s['scoretype'] == '2' and not is_corrupted(s['pdl'], s['uid'])])
    evol_empty = np.array([s for s in ss if s['scoretype'] == '2' and not s['pdl'][-1]['actions']])
    timestamps = np.array([s['timestamp'] for s in solo])
    timestamps_evol = np.array([s['timestamp'] for s in evol])
    time_evol = sum(session_duration(ses, timestamps_evol) for ses in get_sessions(timestamps_evol))
    sessions = get_sessions(timestamps)
    time = sum(session_duration(ses, timestamps) for ses in sessions)

    # get_tmscore = dill.loads(get_tmscore_pkl)

    descendants_memo = {}

    def get_descendants(nid):
        if nid in descendants_memo:
            return descendants_memo[nid]
        # soln_lookup is generated from the list of solutions passed in which are all from a single user
        # the history may include evolver children, which we have to avoid trying to look up
        # sometimes intermediate solutions weren't captured (i.e., not in soln_lookup), but we'll include them
        # so long as any of their descendants were captured
        children = [c for c in child_lookup[nid] if c in soln_lookup or any(x in soln_lookup for x in get_descendants(c))] if nid in child_lookup else []
        descendants_memo[nid] = children + [d for c in children for d in get_descendants(c)]
        return descendants_memo[nid]

    def get_extent(s, stop_nids):
        nid = get_nid(s)
        children = [soln_lookup[c] for c in child_lookup[nid] if
                    c not in stop_nids and c in soln_lookup] if nid in child_lookup else []
        return children + [d for c in children for d in get_extent(c, stop_nids)]

    def get_action_count(line):
        count = 0
        for s in line:
            parent_nid = parent_lookup[get_nid(s)]
            s_actions = s['pdl'][-1]['actions']
            if parent_nid == ROOT_NID:
                count += sum(s_actions.values())
            else:
                parent_actions = soln_lookup[parent_nid]['pdl'][-1]['actions'] if parent_nid in soln_lookup else {}
                for k in s_actions:
                    count += max(s_actions[k] - parent_actions.setdefault(k, 0), 0)
        return count

    def get_source(s):
        cands = s['pdl'][-2::-1]
        i = 0
        while cands[i]['header']['score'] == 9999.99 or cands[i]['actions'] == {} or cands[i]['header']['uid'] == s[
            'uid']:
            i += 1
        return cands[i]

    def is_line_start(s):
        pres = [parent_lookup[get_nid(s)]]
        while pres[-1] != ROOT_NID:
            pres.append(parent_lookup[pres[-1]])
        return all(p not in soln_lookup for p in pres)


    if (len(solo) > 0 or len(evol) > 0) and (any(s['energy'] < 0 for s in solo) or time > delta or time_evol > delta):
        energies = np.array([s['energy'] for s in solo])
        frontier = np.minimum.accumulate(energies)
        frontier_change_idx = np.array([i for i, f in enumerate(frontier[1:], 1) if frontier[i - 1] > f])
        atoms_lookup = {s['sid']: s['atoms'] for s in ss}
        tmscore_lookup = {}
        if len(frontier_change_idx) < 2:
            breakthroughs = None
            # ret['score_dist'] = None
            # ret['frontier_rmsd'] = None
            # ret['frontier_deviations'] = None
            # ret['dist_mat'] = None
            ret['frontier_pdbs'] = None
            ret['frontier_tmscores'] = None
            ret['first_pdb'] = None
            ret['upload_rate'] = None
            ret['lines'] = None
            # ret['lines_tmscores'] = None
        else:
            # ret['score_dist'] = np.array([abs(f - e) for e, f in zip(energies, frontier)])
            # logging.debug("frontier_rmsd")
            # full_frontier = [min(solo[:i+1], key=lambda s: s['energy']) for i in range(len(solo))]
            # rmsd = [weighted_rmsd(s['atoms'], f['atoms'], 90) for s, f in zip(solo, full_frontier)]
            # ret['frontier_rmsd'] = np.array([r for r,w,dev in rmsd])
            # ret['frontier_deviations'] = np.array([dev for r,w,dev in rmsd])
            ret['frontier_pdbs'] = solo[frontier_change_idx]
            ret['first_pdb'] = solo[0] if len(solo) > 0 else None

            # logging.debug("{} {} frontier tmscores".format(pid, uid))
            # frontier_pairs = list(combinations([s['sid'] for s in solo[frontier_change_idx]], 2))
            # for k, v in tmscore([c for c in frontier_pairs if np.isnan(get_tmscore(c))],
            #                     "tmp_data/{}_{}_frontier".format(pid, uid), atoms_lookup).items():
            #     tmscore_lookup[k] = v
            # ret['frontier_tmscores'] = {c: tmscore_lookup.get(c, get_tmscore(c)) for c in frontier_pairs}

            logging.debug("{} {} soloist lines".format(pid, uid))
            lines = [[s] + sorted([soln_lookup[x] for x in get_descendants(get_nid(s)) if x in soln_lookup],
                                  key=lambda x: x['timestamp']) for s in solo if is_line_start(s)]
            ret['lines'] = []
            for line in lines:
                ts = [x['timestamp'] for x in line]
                # logging.debug("{} {} computing tmscores".format(pid, uid))
                # score_pairs = [(s['sid'], soln_lookup[parent_lookup[get_nid(s)]]['sid']) for s in line[1:]]
                # for k, v in tmscore([c for c in score_pairs if np.isnan(get_tmscore(c))],
                #                     "tmp_data/{}_{}".format(pid, uid), atoms_lookup):
                #     tmscore_lookup[k] = v
                devs = [np.nan] + [rmsd(s['atoms'], soln_lookup[parent_lookup[get_nid(s)]]['atoms'])[1] if parent_lookup[get_nid(s)] in soln_lookup else np.nan for s in line[1:]]
                pdb_infos = []
                variants = []
                for s, dev in zip(line, devs):
                    parent_sid = soln_lookup[parent_lookup[get_nid(s)]]['sid'] if parent_lookup[get_nid(s)] in soln_lookup else None
                    if parent_sid is None and parent_lookup[get_nid(s)] != ROOT_NID and not is_line_start(s):
                        cur = parent_lookup[parent_lookup[get_nid(s)]]
                        while cur not in soln_lookup:
                            cur = parent_lookup[cur]
                        parent_sid = nid_to_sid[cur]
                    info = PDB_Info(s['sid'], s['pid'], s['uid'], s['gid'], s['sharing_gid'], s['scoretype'], s['pdl'],
                                    s['energy'], s['energies'], s['timestamp'], parent_sid, np.nan, dev)
                                    # tmscore_lookup.get((s['sid'], parent_sid), get_tmscore((s['sid'], parent_sid))), dev)
                    pdb_infos.append(info)
                #     if tmscore_lookup.get((s['sid'], parent_sid), get_tmscore((s['sid'], parent_sid))) < 0.5:
                #         variants.append(s)
                # variant_nids = [get_nid(s) for s in variants]
                # variants = [[soln] + sorted(get_extent(soln, variant_nids), key=lambda x: x['timestamp'])
                #             for soln in variants]

                def make_variant(variant, line):
                    indices = [line.index(v) for v in variant]
                    ts = [x['timestamp'] for x in variant]
                    return SolvingLineVariant(get_action_count(variant),
                                              sum(session_duration(ses, ts) for ses in get_sessions(ts)), indices)

                # computing action count here is dumb, replacing it with None to avoid needing to keep it up to date
                sl = SolvingLine(None, sum(session_duration(ses, ts) for ses in get_sessions(ts)),
                                 pdb_infos, [make_variant(variant_line, line) for variant_line in variants])
                ret['lines'].append(sl)
            # ret['lines_tmscores'] = tmscore(list(combinations([min(line, key=lambda x: x['energy'])['sid'] for line in lines], 2)),
            #                                 "tmp_data/{}_{}_lines".format(pid, uid))

            # breakthroughs
            # logging.debug("{} {} breakthroughs".format(pid, uid))
            # regions = []
            # for params in breakthrough_params:
            #     regions.append(find_breakthroughs(ret['lines'], **params))
            # breakthroughs = pd.Series(regions,
            #                           index=pd.MultiIndex.from_tuples([tuple([pid, uid] + list(x.values())) for x in breakthrough_params], names=['pid', 'uid'] + list(params.keys())))
            breakthroughs = None

            # logging.debug("dist_mat")
            # def dist_wrapper(a, b):
            #     # [0] because pdist takes 2d array with 1-element rows
            #     return weighted_rmsd(a[0]['atoms'], b[0]['atoms'], 90)[0]
            # dist_array = np.array(ret['frontier_pdbs']).reshape(-1, 1)
            # if len(ret['frontier_pdbs']) > 100:
            #     dist_array = np.array([min(ret['frontier_pdbs'][s:e], key=lambda p: p['energy']) for s, e
            #                            in get_time_splits(time, [p['timestamp'] for p in ret['frontier_pdbs']], 100)]).reshape(-1, 1)
            # logging.debug(len(dist_array))
            # ret['dist_mat'] = scipy.spatial.distance.pdist(dist_array, dist_wrapper)
            ret['upload_rate'] = np.array(
                [len([x for x in solo if s['timestamp'] >= x['timestamp'] > s['timestamp'] - delta]) for s in solo])

        logging.debug("{} {} evolver lines".format(pid, uid))
        soln_lookup = {get_nid(s): s for s in ss}
        evol_lines = [[s] + sorted([soln_lookup[x] for x in get_descendants(get_nid(s)) if x in soln_lookup],
                                   key=lambda x: x['timestamp']) for s in evol if parent_lookup[get_nid(s)] not in soln_lookup] # parents are from another user
        ret['evol_lines'] = [] if len(evol_lines) > 0 else None
        for line in evol_lines:
            # score_pairs = [(s['sid'], soln_lookup[parent_lookup[get_nid(s)]]['sid']) for s in line[1:]]
            # for k, v in tmscore([c for c in score_pairs if np.isnan(get_tmscore(c))],
            #                     "tmp_data/{}_{}".format(pid, uid), atoms_lookup):
            #     tmscore_lookup[k] = v
            devs = [np.nan] + [rmsd(s['atoms'], soln_lookup[parent_lookup[get_nid(s)]]['atoms'])[1] if parent_lookup[get_nid(s)] in soln_lookup else np.nan for s in line[1:]]
            pdb_infos = [
                PDB_Info(s['sid'], s['pid'], s['uid'], s['gid'], s['sharing_gid'], s['scoretype'], s['pdl'],
                         s['energy'], s['energies'], s['timestamp'], nid_to_sid[parent_lookup[get_nid(s)]],
                         # tmscore_lookup.get((s['sid'], nid_to_sid[parent_lookup[get_nid(s)]]),
                         #                    get_tmscore((s['sid'], nid_to_sid[parent_lookup[get_nid(s)]]))),
                         np.nan, dev) for s, dev in zip(line, devs) if parent_lookup[get_nid(s)] in soln_lookup]
            el = EvolvingLine(get_source(line[0]), pdb_infos)
            ret['evol_lines'].append(el)

        ret['energies'] = energies
        # ret['solo_count'] = len(solo)
        # ret['evol_count'] = len(evol)
        ret['timestamps'] = timestamps
        # ret['timestamps_evol'] = timestamps_evol
        ret['time'] = time
        # ret['time_evol'] = time_evol
        # ret['pdb_infos'] = np.array([PDB_Info(s['sid'], s['pid'], s['uid'], s['gid'], s['sharing_gid'], s['scoretype'],
        #                                       s['pdl'], s['energy'], s['timestamp'],
        #                                       soln_lookup[parent_lookup[get_nid(s)]]['sid']) for s in ss])
        ret['pid'] = pid
        ret['uid'] = uid
        logging.debug("{} {} done".format(pid, uid))
        return ret, breakthroughs
    logging.debug("{} {} done".format(pid, uid))
    return None, None


def process_puzzle_meta(pid, overwrite=False, snapshot_threads=15):
    metafile = "data/puzzle_solutions/solution_{}/{}_meta.h5".format(pid, pid)
    tmscore_file = "data/puzzle_solutions/solution_{}/{}_tmscore.csv".format(pid, pid)
    soln_csv_file = "data/puzzle_solutions/solution_{}/{}_soln.csv".format(pid, pid)
    hist_csv_file = "data/puzzle_solutions/solution_{}/{}_hist.csv".format(pid, pid)

    if not os.path.exists(metafile) or overwrite:
        # tmscore_lookup = {}
        # if os.path.exists(tmscore_file):
        #     with open(tmscore_file) as fp:
        #         print(pid, "loading tmscores")
        #         tmscore_in = csv.DictReader(fp, fieldnames=['sid_a', 'sid_b', 'tmscore'])
        #         tmscore_lookup = {(r['sid_a'], r['sid_b']): float(r['tmscore']) for r in tmscore_in}

        # def get_tmscore(key):
        #     return tmscore_lookup.get(key, np.nan)

        soln_lookup = {}
        nid_to_sid = {}
        history = {}
        # with open("data/puzzle_solutions/solution_{}/{}_soln.pickle".format(pid, pid), 'rb') as fp:
        #     solns_clean = pickle.load(fp)
        #     soln_lookup = {get_nid(s): s for s in solns_clean}
        #     solvers = {k: sorted(g, key=lambda x: int(x['timestamp'])) for k, g in
        #                groupby(sorted(solns_clean, key=lambda s: s['uid']), lambda s: s['uid'])}
        # with open("data/puzzle_solutions/solution_{}/{}_hist.pickle".format(pid, pid), 'rb') as fp:
        #     history = pickle.load(fp)

        if not os.path.exists(soln_csv_file):
            print(pid, "fetching soln csv")
            sys.stdout.flush()
            if not os.path.exists("data/puzzle_solutions/solution_{}".format(pid)):
                os.makedirs("data/puzzle_solutions/solution_{}".format(pid))
            subprocess.run(["scp", "wannacut:~/foldit/{}".format(soln_csv_file), soln_csv_file], stdout=subprocess.DEVNULL)
        with open("data/puzzle_solutions/solution_{}/{}_soln.csv".format(pid, pid)) as fp:
            print(pid, "processing", soln_csv_file)
            sys.stdout.flush()
            soln_in = csv.DictReader(fp, lineterminator='\n')
            for r in soln_in:
                r['pdl'] = json.loads(r['pdl'])
                r['guide_used'] = False
                for p in r['pdl']:
                    try:  # a pdl entries have a different header structure and were parsed incorrectly
                        p['header']['score'] = float(p['header']['score'])
                    except ValueError:
                        continue
                    if p['header']['score'] == 9999.99:
                        r['guide_used'] = True
                r['energy'] = float(r['energy'])
                r['timestamp'] = int(r['timestamp'])
                r['atoms'] = get_atoms(r)
                r.pop('ca')
                r['energies'] = [EnergyComponent(*e) for e in json.loads(r['energies'])] if r['energies'] else None
                if len(r['pdl']) > 0 and (r['uuid'], int(r['count'])) != ROOT_NID:
                    #and not all(sum(p['actions'].values()) == 0 for p in r['pdl']):
                    soln_lookup.setdefault((r['uuid'], int(r['count'])), []).append(r)

            solns_pre = []
            for nid, ss in soln_lookup.items():
                s = min(ss, key=lambda x: x['energy'])
                soln_lookup[nid] = s
                nid_to_sid[nid] = s['sid']
                if len(s['pdl']) > 0:
                    solns_pre.append(s)
            protein_size, _ = Counter([len(s['atoms']) for s in solns_pre]).most_common(1)[0]
            solns_clean = [s for s in solns_pre if len(s['atoms']) == protein_size]
            solvers = {k: sorted(g, key=lambda x: int(x['timestamp'])) for k, g in
                       groupby(sorted(solns_clean, key=lambda s: s['uid']), lambda s: s['uid'])}

        if not os.path.exists(hist_csv_file):
            print(pid, "fetching hist csv")
            sys.stdout.flush()
            subprocess.run(["scp", "wannacut:~/foldit/{}".format(hist_csv_file), hist_csv_file], stdout=subprocess.DEVNULL)
        with open("data/puzzle_solutions/solution_{}/{}_hist.csv".format(pid, pid)) as fp:
            print(pid, "processing", hist_csv_file)
            hist_in = csv.DictReader(fp, fieldnames=["pid", "uuid", "count", "parent_uuid", "parent_count"])
            for r in hist_in:
                key = (r['parent_uuid'], int(r['parent_count']))
                r['count'] = int(r['count'])
                r['parent_count'] = int(r['parent_count'])
                history.setdefault(key, []).append(r)

        parents = {}
        children = get_children(ROOT_NID, history)
        children = [(ROOT_NID, c) for c in children]
        while len(children) > 0:
            for p, c in children:
                assert c not in parents
                parents[c] = p
            children = [(c, nc) for p, c in children for nc in get_children(c, history)]

        logging.debug("{} generating lookups".format(pid))
        parent_lookup = {}
        for k in soln_lookup:
            parent = parents[k]
            while parent not in soln_lookup and parent != ROOT_NID:
                parent = parents[parent]
            assert parent in soln_lookup or parent == ROOT_NID
            parent_lookup[k] = parent
        child_lookup = {parent: [c for p, c in g] for parent, g in
                        groupby(sorted([(p, c) for c, p in parent_lookup.items()]), lambda x: x[0])}

        descendants_memo = {}

        def get_descendants(nid):
            if nid in descendants_memo:
                return descendants_memo[nid]
            # soln_lookup is generated from the list of solutions passed in which are all from a single user
            # the history may include evolver children, which we have to avoid trying to look up
            children = [c for c in child_lookup[nid] if c in soln_lookup or any(x in soln_lookup for x in get_descendants(c))] if nid in child_lookup else []
            descendants_memo[nid] = children + [d for c in children for d in get_descendants(c)]
            return descendants_memo[nid]

        logging.debug("{} correcting timestamps".format(pid))
        bases = [get_nid(s) for s in soln_lookup.values() if parent_lookup[get_nid(s)] == ROOT_NID]
        while len(bases) > 0:
            nid = bases.pop(0)
            if nid in child_lookup:
                if nid in soln_lookup:
                    cur = soln_lookup[nid]
                    descendants = [soln_lookup[x] for x in get_descendants(nid) if x in soln_lookup]
                    if cur['timestamp'] > min(c['timestamp'] for c in descendants):
                        grandparent = {'timestamp': 0}
                        if parent_lookup[nid] in soln_lookup:
                            grandparent = soln_lookup[parent_lookup[nid]]
                        assert grandparent['timestamp'] <= min(c['timestamp'] for c in descendants)
                        cur['timestamp'] = max(min(c['timestamp'] for c in descendants) - 300, grandparent['timestamp'] + 1)
                bases.extend([c for c in child_lookup[nid]])

        delta = 3600

        print(pid, "computing soln metrics")
        sys.stdout.flush()

        param_ranges = {
            "energy_threshold_frac": [0.25, 0.5, 0.75],
            "rate_threshold": [-0.001, -0.01],
            "diff_threshold": [-1, -10, -25],
            "tm_threshold": [0.5, 0.9, 1]
        }

        breakthrough_params = [dict(d) for d in product(*[[(k, v) for v in vs] for k, vs in param_ranges.items()])]

        logging.debug("{} passing parent_lookup, size {} and child_lookup, size {} to threads".format(pid, sys.getsizeof(parent_lookup), sys.getsizeof(child_lookup)))
        with Pool(snapshot_threads) as snapshot_pool:
            acc = snapshot_pool.map_async(partial(process_snapshots, delta=delta, breakthrough_params=breakthrough_params,
                                                  parent_lookup=parent_lookup, child_lookup=child_lookup,
                                                  # nid_to_sid=nid_to_sid, get_tmscore_pkl=dill.dumps(get_tmscore)),
                                                  nid_to_sid=nid_to_sid),
                                          sorted(solvers.values(), key=len, reverse=True), chunksize=1).get()
        df = pd.DataFrame(data=[d for d, _ in acc if d is not None])
        # breakthroughs = pd.concat([b for _, b in acc if b is not None])
        breakthroughs = pd.DataFrame()

        print(pid, 'metrics computed')
        sys.stdout.flush()

        best = df[df.frontier_pdbs.notnull()].frontier_pdbs.apply(lambda x: x[-1])
        # logging.debug("{} puzzle frontier tmscores".format(pid))
        # atoms_lookup = {s['sid']: s['atoms'] for s in best}
        # best_pairs = list(combinations([s['sid'] for s in best], 2))
        # for k, v in tmscore([c for c in best_pairs if c not in tmscore_lookup],
        #                     "tmp_data/{}_best".format(pid), atoms_lookup):
        #     tmscore_lookup[k] = v
        # best_tmscores = {c: tmscore_lookup[c] if c in tmscore_lookup else np.nan for c in best_pairs}

        en_lookup = {}
        for _, z in df.apply(lambda r: zip(r['timestamps'], r['energies']), axis=1).iteritems():
            for t, e in z:
                if t not in en_lookup:
                    en_lookup[t] = []
                en_lookup[t].append(e)
        pfront = np.minimum.accumulate([min(es) for t, es in sorted(en_lookup.items())])

        upload_baseline = max(stats.mode(np.concatenate(df.upload_rate[df.upload_rate.notnull()].values)).mode)
        df = df.assign(upload_ratio=df.upload_rate / upload_baseline)

        # it appears there's a clustering of energies for solutions that have only one or two actions (usually repack), so we'll use that as the energy baseline
        energy_baseline = scipy.stats.mode(df[df.first_pdb.notnull() & df.first_pdb.apply(
            lambda p: p and sum(p['pdl'][0]['actions'].values()) < 3)].first_pdb.apply(
            lambda p: round(p['energy']))).mode.min()

        print(pid, "getting structure")
        struct_file = "data/puzzle_solutions/solution_{}/{:010}.ir_puzzle.pdb".format(pid, int(pid))
        # setup_file = "data/puzzle_solutions/solution_{}/{:010}.ir_puzzle.puzzle_setup".format(pid, int(pid))
        if not os.path.exists(struct_file):
            subprocess.run(["scp", "wannacut:~/foldit/{}".format(struct_file), struct_file], stdout=subprocess.DEVNULL)
            # subprocess.run(["scp", "wannacut:~/foldit/{}".format(setup_file), setup_file], stdout=subprocess.DEVNULL)
        with open(struct_file) as init_pdb:
            content = init_pdb.read()
            sec_struct = {i: l for i, l in [x.split()[:2] for x in re.findall('^(?!ATOM)\s+?\d+.*',
                                                                              content, re.MULTILINE)]}
            assert all(v in ['H', 'E', 'L', 'C'] for v in sec_struct.values())
            atoms = Counter([x.split()[5] for x in re.findall('^ATOM.*', content, re.MULTILINE)])
            structure = {
                'loop': [atoms[i] for i, l in sec_struct.items() if l == 'C' or l == 'L'],
                'helix': [atoms[i] for i, l in sec_struct.items() if l == 'H'],
                'sheet': [atoms[i] for i, l in sec_struct.items() if l == 'E']
            }
        # meta = PuzzleMeta(pid, best_tmscores, pfront, upload_baseline, energy_baseline, structure)
        meta = PuzzleMeta(pid, None, pfront, upload_baseline, energy_baseline, structure)
        print(pid, 'puzzle metrics computed')
        sys.stdout.flush()

        print(pid, "writing soln output")
        sys.stdout.flush()
        if os.path.exists(metafile) and overwrite:
            logging.debug("{} deleting existing meta file".format(pid))
            subprocess.run(['rm', metafile]) # remove to avoid ever accumulating data files
        store = pd.HDFStore(metafile)
        store["df"] = df
        store["bts"] = breakthroughs
        store["puz"] = pd.Series([meta]) # must be wrapped in a pandas data structure
        store.close()
        subprocess.run(["rm", soln_csv_file])
        print(pid, "done")
    else:
        print(metafile, "exists, will not overwrite")

if __name__ == '__main__':
    # with open("wannacut_pids.txt") as fp:
    #     potential_pids = [p.strip() for p in fp.readlines() if p.strip() not in bad_pids]
    # target_pids = random.sample(potential_pids, 25)
    # prev_pids = [pid for pid in ["996547", "997461", "998071", "2002244", "2003628", "2003976", "2004044", "998001", "2004018", "2003868", "2004053", "2001044", "2004059", "2004016", "1998468", "2003996", "997439", "2003982", "2003205", "1998729", "2004021", "2000778", "997597", "2002399", "2002914", "2002544", "1998526", "2003903", "2003990", "1998709", "2000644", "2001757", "2004058", "998219", "2001695", "2003177"] if pid not in skip_pids]
    # prev_pids = ['1998468', '1998526', '2000644', '2000778', '2001695', '2001757', '2002399', '2002914', '2003177', '2003868', '2003903', '2003976', '2003982', '2003996', '2004016', '2004018', '2004044', '2004053', '2004058', '2004059', '997439', '997597']
    # denovo_pids = ['989270', '989362', '989506', '989602', '989733', '989868', '989920', '990046', '990126', '990185', '990281', '990858', '993615', '993654', '993710', '994320', '994368', '994551', '994710', '994845', '994928', '994978', '995014', '995086', '995131', '995306', '995347', '995443', '995483', '995662', '995716', '995805', '995844', '995875', '995904', '995947', '995972', '995978', '996010', '996055', '996072', '996078', '996197', '996198', '996267', '996312', '996547', '996621', '996752', '996837', '997111', '997118', '997173', '997238', '997275', '997315', '997362', '997364', '997414', '997418', '997439', '997463', '997473', '997476', '997486', '997529', '997542', '997597', '997611', '1998493', '1998526', '1998565', '1998587', '1998619', '1998660', '1998696', '2000151', '2000196', '2000230', '2000263', '2000398', '2000481', '2000512', '2000655', '2000678', '2000778', '2000801', '2000856', '2000926', '2000953', '2000995', '2001022', '2001054', '2001081', '2001113', '2001145', '2001183', '2001234', '2001287', '2001308', '2001512', '2001552', '2001603', '2001694', '2001697', '2001738', '2001779', '2001794', '2001825', '2001857', '2002058', '2002093', '2002110', '2002141', '2002166', '2002196', '2002232', '2002255', '2002294', '2002346', '2002399', '2002438', '2002454', '2002475', '2002499', '2002522', '2002546', '2002567', '2002582', '2002669', '2002682', '2002722', '2002747', '2002771', '2002792', '2002840', '2003008', '2003015', '2003059', '2003111', '2003195', '2003236', '2003270', '2003287', '2003313', '2003340', '2003374', '2003388', '2003433', '2003465', '2003490', '2003534', '2003578', '2003642', '2003670', '2003784', '2003823', '2003854', '2003881', '2003900', '2003912', '2003928', '2003949', '2003976', '2003990', '2004016', '2004044', '2004058', '2004124', '2004204', '2004205', '2004391', '2004442', '2004481', '2004640', '2004679', '2004719', '2004781']
    # target_pids = [pid for pid in set(prev_pids + denovo_pids) if pid not in skip_pids]

    # clean list of not-too-old denovo puzzles
    # target_pids = ['2000926', '2000995', '2001054', '2001081', '2001234', '2001512', '2001552', '2001603',
    #                '2001697', '2001738', '2001779', '2001794', '2001825', '2001857', '2002058', '2002093',
    #                '2002110', '2002141', '2002166', '2002196', '2002232', '2002255', '2002294', '2002346',
    #                '2002475', '2002567', '2002669', '2002682', '2002747', '2002771', '2002792', '2002840',
    #                '2003008', '2003111', '2003195', '2003236', '2003270', '2003287', '2003313', '2003340',
    #                '2003374', '2003388', '2003433', '2003465', '2003578', '2003642', '2003784', '2003823', '2003854',
    #                '2003881', '2003900', '2003912', '2003949', '2003976', '2003990', '2004016', '2004044',
    #                '2004058']

    # clean list of all pids from 2016 and 2017
    target_pids = ['2001697', '2001723', '2001734', '2001757', '2001779', '2001783', '2001803', '2001822', '2001825',
                   '2001845', '2002051', '2002056', '2002089', '2002100', '2002108', '2002110', '2002126', '2002141',
                   '2002150', '2002171', '2002184', '2002196', '2002212', '2002290', '2002294', '2002299', '2002308',
                   '2002327', '2002334', '2002356', '2002399', '2002411', '2002425', '2002429', '2002442', '2002460',
                   '2002469', '2002475', '2002481', '2002486', '2002500', '2002513', '2002527', '2002546', '2002549',
                   '2002553', '2002554', '2002562', '2002565', '2002567', '2002572', '2002578', '2002582', '2002583',
                   '2002590', '2002595', '2002605', '2002607', '2002610', '2002613', '2002614', '2002652', '2002664',
                   '2002669', '2002682', '2002689', '2002703', '2002715', '2002722', '2002735', '2002745', '2002760',
                   '2002766', '2002781', '2002787', '2002840', '2002877', '2002878', '2002906', '2002922', '2002949',
                   '2003010', '2003111', '2003125', '2003169', '2003195', '2003205', '2003206', '2003236', '2003240',
                   '2003265', '2003270', '2003281', '2003285', '2003287', '2003303', '2003308', '2003313', '2003322',
                   '2003333', '2003340', '2003355', '2003360', '2003374', '2003382', '2003383', '2003388', '2003414',
                   '2003416', '2003433', '2003455', '2003460', '2003465', '2003483', '2003485', '2003490', '2003530',
                   '2003532', '2003534', '2003535', '2003561', '2003578', '2003583', '2003594', '2003621', '2003624',
                   '2003628', '2003639', '2003642', '2003648', '2003668', '2003670', '2003682', '2003698', '2003707',
                   '2003730', '2003734', '2003744', '2003762', '2003766', '2003773', '2003780', '2003784', '2003791',
                   '2003796', '2003814', '2003817', '2003823', '2003825', '2003828', '2003841', '2003846', '2003850',
                   '2003854', '2003856', '2003859', '2003868', '2003876', '2003877', '2003881', '2003893', '2003894',
                   '2003900', '2003903', '2003908', '2003912', '2003918', '2003921', '2003928', '2003929', '2003939',
                   '2003949', '2003955', '2003958', '2003976', '2003982', '2003984', '2003990', '2004007', '2004016',
                   '2004018', '2004021', '2004044', '2004053', '2004055', '2004058', '2004059', '2004082', '2004124',
                   '2004133', '2004151', '2004158', '2004159', '2004161',]
                   # collection in progress:
                   # '2004180', '2004202', '2004204', '2004205',
                   # '2004234', '2004248', '2004263', '2004269', '2004292', '2004307', '2004322', '2004332', '2004354',
                   # '2004375', '2004391', '2004400', '2004401', '2004442', '2004448', '2004456', '2004467', '2004472',
                   # '2004481', '2004483', '2004499', '2004506', '2004520', '2004523', '2004548', '2004549', '2004575',
                   # '2004579']

    parser = argparse.ArgumentParser(prog='EDM_2018_processing.py')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--puzthreads', type=int, default=8)
    parser.add_argument('--solverthreads', type=int, default=15)
    parser.add_argument('--pids', nargs='+')
    args = parser.parse_args()
    if args.pids:
        target_pids = args.pids
    print("targeting", target_pids)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    sys.setrecursionlimit(10000)

    if args.puzthreads == 1:
        for pid in target_pids:
            process_puzzle_meta(pid, overwrite=args.overwrite, snapshot_threads=args.solverthreads)
    else:
        with ProcessPoolExecutor(args.puzthreads) as puzzle_pool:
            puzzle_pool.map(partial(process_puzzle_meta, overwrite=args.overwrite, snapshot_threads=args.solverthreads),
                            target_pids, chunksize=1)
