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
from typing import Dict, Tuple
from itertools import combinations, chain
import matplotlib
matplotlib.use("Agg")


def get_relevant_sids(pdb: PDB_Info, soln_lookup: Dict[str, PDB_Info], child_lookup: Dict[str, str]) -> List[str]:
    assert pdb.pdl[-1]['actions'] != {}
    backbone = []
    cur = pdb
    while cur.parent_sid:
        backbone.append(cur)
        cur = soln_lookup[cur.parent_sid]
    backbone.append(cur)
    relevant = sorted([c for c in sum([get_children(x, soln_lookup, child_lookup) for x in backbone], []) if
                       c.timestamp <= pdb.timestamp], key=lambda x: x.timestamp)
    return [x.sid for x in relevant]


def get_action_step(pdbs: List[PDB_Info], soln_lookup: dict) -> List:
    diffs = []
    for pdb in pdbs:
        actions, macros = collect_pdl_entries(pdb)
        if pdb.parent_sid is None:
            diffs.append(Counter(actions))
        parent = soln_lookup[pdb.parent_sid]
        pactions, pmacros = collect_pdl_entries(parent)
        action_diff = Counter({a: actions.get(a, 0) - pactions.get(a, 0) for a in
                           set(pactions.keys()).union(set(actions.keys()))})
        diffs.append(action_diff)
    ad = sum(diffs, Counter())
    return get_action_stream(ad)


def make_collab_series(pdbs, soln_lookup):
    pdbs = sorted(pdbs, key=lambda p: p.timestamp)
    ts = np.array([p.timestamp for p in pdbs])
    steps = np.arange(ts[0], ts[-1], 60)
    steps = np.concatenate((steps, [steps[-1] + 60])) # ensure the last timestamp is included in steps
    uids = np.unique([p.uid for p in pdbs])
    step_scoretypes = {uid: np.zeros(len(steps) - 1) for uid in uids}
    num_features = 14
    uid_slice_lookup = {uid: slice(i * num_features, (i + 1) * num_features) for i, uid in enumerate(uids)}
    series = np.zeros((len(steps) - 1, len(uids) * num_features))
    pdb_idx = 0
    for i in range(1, len(steps)):
        next_pdb_idx = pdb_idx
        while next_pdb_idx < len(pdbs) and pdbs[next_pdb_idx].timestamp <= steps[i]:
            next_pdb_idx += 1
        step_pdbs = pdbs[pdb_idx: next_pdb_idx]
        pdb_idx = next_pdb_idx
        if len(step_pdbs) > 0:
            for uid in uids:
                uid_pdbs = [p for p in step_pdbs if p.uid == uid]
                if len(uid_pdbs) > 0:
                    series[i - 1, uid_slice_lookup[uid]] = get_action_step(uid_pdbs, soln_lookup)
                    scoretypes = {int(p.scoretype) for p in uid_pdbs}
                    step_scoretypes[uid][i - 1] = sum(scoretypes)
    return series, steps[:-1], uid_slice_lookup, step_scoretypes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='collab_models.py')
    parser.add_argument("datapath")
    parser.add_argument("pid")
    parser.add_argument("gid")
    parser.add_argument("-k", "--krange", nargs="+", type=int, default=list(range(5, 15)))
    args = parser.parse_args()
    output_id = "{}_{}".format(args.pid, args.gid)

    soln_lookup = {}
    parent_lookup = {}
    child_lookup = {}
    data, puz_metas = load_extend_data([args.pid], soln_lookup, parent_lookup, child_lookup, True, 600)

    relevant_sids = []
    for s in soln_lookup.values():
        if s.gid == args.gid:
            if s.scoretype == '2':
                relevant_sids.append(s.sid)
            elif s.sharing_gid == args.gid:
                relevant_sids.extend(get_relevant_sids(s, soln_lookup, child_lookup))
    relevant_pdbs = [soln_lookup[sid] for sid in set(relevant_sids)]

    if os.path.exists("{}/{}".format(args.datapath, output_id)):
        ser = np.loadtxt("{}/{}_series.txt".format(args.datapath, output_id), dtype=np.int32)
        steps = np.loadtxt("{}/{}_steps.txt".format(args.datapath, output_id), dtype=np.int32)
        with open("{}/{}_uid_slice_lookup.pickle".format(args.datapath, output_id), 'rb') as fp:
            uid_slice_lookup = pickle.load(fp)
        with open("{}/{}_step_scoretypes.pickle".format(args.datapath, output_id), 'rb') as fp:
            step_scoretypes = pickle.load(fp)
    else:
        logging.info("No TICC output found for gid {}, pid {}. Running TICC for krange={} now".format(args.gid, args.pid, 
                                                                                                      args.krange))
        ser, steps, uid_slice_lookup, step_scoretypes = make_collab_series(relevant_pdbs, soln_lookup)
        np.savetxt("{}/{}_series.txt".format(args.datapath, output_id), ser)
        np.savetxt("{}/{}_steps.txt".format(args.datapath, output_id), steps)
        with open("{}/{}_uid_slice_lookup.pickle".format(args.datapath, output_id), 'wb') as fp:
            pickle.dump(uid_slice_lookup, fp)
        with open("{}/{}_step_scoretypes.pickle".format(args.datapath, output_id), 'wb') as fp:
            pickle.dump(step_scoretypes, fp)
        run_TICC({output_id: ser}, args.datapath, range(5, 15))
    
    cluster_lookup, mrf_lookup, model_lookup, _ = load_TICC_output(args.datapath, [output_id], args.krange)

    clusters = cluster_lookup[output_id][10]

    patterns = get_patterns(mrf_lookup[output_id][10], cluster_lookup[output_id][10], {("993077", "2003433"): (0, len(ser))})
    cids = {p.cid for p in patterns}
    
    action_labels = get_action_labels()
    uids = list(uid_slice_lookup.keys())
    fig, axs = plt.subplots(len(uids), 1, sharex=True, figsize=(100, 6*len(uids)))
    fig.subplots_adjust(hspace=0)
    fills = []
    for i, uid in enumerate(uids):
        print(uid)
        ax = axs[i]
        s = ser[:, uid_slice_lookup[uid]]
        print("plotting actions")
        for j in range(s.shape[1]):
            ax.plot(steps, s[:, j], color=plt.cm.get_cmap("tab20").colors[j], alpha=0.7)
        print("plotting patterns")
        for i in range(10):
            fills.append(ax.fill_between(steps, 0, s.max(), (clusters == i), color=plt.cm.get_cmap("Set3").colors[i], alpha=0.4))
        print("adjusting scoretypes")
        scoretypes = step_scoretypes[uid].copy()
        for j, st in enumerate(scoretypes):
            c = clusters[j]
            i = next((i for i in range(j - 1, 0, -1) if clusters[i] == c and scoretypes[i] != st), None)
            k = next((k for k in range(j + 1, len(clusters)) if clusters[k] == c and scoretypes[k] != st), None)
            if st == 0:
                if i and (clusters[i:j] == c).all():
                    scoretypes[j] = scoretypes[i]
                elif k and (clusters[j:k] == c).all():
                    scoretypes[j] = scoretypes[k]
            elif st == 1 or st == 2:
                if i and i >= j - 5 and scoretypes[i] != 0:
                    scoretypes[j] = 3
                elif k and k <= j + 5 and scoretypes[k] != 0:
                    scoretypes[j] = 3
        print("plotting scoretypes")
        ax.fill_between(steps, 0, s.max(), (scoretypes == 1), hatch="/", facecolor="none", edgecolor="black")
        ax.fill_between(steps, 0, s.max(), (scoretypes == 2), hatch="\\", facecolor="none", edgecolor="black")
        ax.fill_between(steps, 0, s.max(), (scoretypes == 3), hatch="x", facecolor="none", edgecolor="black")
        ax.set_yscale("log", nonposy="clip")
    axs[0].legend(action_labels, loc=1)
    axs[0].legend(fills, list(range(10)), loc=4)
    fig.tight_layout()
    fig.savefig("{}/{}_viz.png".format(args.datapath, output_id))
    plt.close()
