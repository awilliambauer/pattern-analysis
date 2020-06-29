import subprocess
import argparse
import os
import sys
import csv
import json
import pickle
import logging
from itertools import groupby
from functools import partial
import pandas as pd
import numpy as np
from util import PDB_Info
from typing import NamedTuple, Tuple
from pattern_extraction import get_relevant_sids, load_extend_data, load_TICC_output
from check_models import load_sub_lookup
from process_puzzle_meta import process_puzzle_meta
from collab_viz import *
from dump_predicted_patterns import pattern_count_from_pdbs


def get_team_json(root: Collaborator, pid: str, pattern_count_fn) -> dict:
    ts = [p.timestamp for p in root.pdbs]
    team = {"uid": root.uid, "pid": pid, "gid": root.gid, "energy": root.tag.energy, "energy_comps": root.energy_comps,
            "source": root.source, "start": min(ts, default=None), "end": max(ts, default=None), "children": [], "pattern_count": {}}
    if len(root.pdbs) > 0:
        team["pattern_count"] = pattern_count_fn(root.pdbs)
    for child in root.children:
        team["children"].append(get_team_json(child, pid, pattern_count_fn))
    return team


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='dump_team_structure.py')
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--append", action='store_true')
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    sys.setrecursionlimit(10000)
    version_tag = "v3"
    user_filename = "data/user_teams_{}.json".format(version_tag)
    pid_tracker_filename = "pid_tracker_teams_{}.txt".format(version_tag)

    if not args.overwrite and not args.append and os.path.exists(user_filename):
        logging.info("{} already exists".format(user_filename))
        sys.exit(1)

    if args.overwrite and args.append:
        logging.info("cannot both --overwrite and --append")
        sys.exit(2)

    pids = np.loadtxt("wannacut_prediction_pids_v6.txt", dtype=np.str).tolist()
    bad_pids = np.loadtxt("bad_prediction_pids.txt", dtype=np.str).tolist()
    pids = [pid for pid in pids if pid not in bad_pids]

    if args.append:
        user_teams_out = open(user_filename, 'a')
        pid_tracker = open(pid_tracker_filename, "r+")
        for pid in pid_tracker:
            pid = pid.strip()
            logging.info("skipping {}, existing entry".format(pid))
            try:
                pids.remove(pid)
            except: # as list of pids is revised, we might try to remove previous entry that is no longer in the list
                logging.info("previously included {} no longer on list".format(pid))
    else:
        user_teams_out = open(user_filename, 'w')
        pid_tracker = open(pid_tracker_filename, 'w')

    best_k = 5
    noise = np.loadtxt("patterns_comp/noise_values.txt")
    _, mrf_lookup, model_lookup, _ = load_TICC_output("patterns_comp", ["all"], [best_k])
    with open("patterns_comp/all/subpatterns/subseries_lookup.pickle", 'rb') as fp:
        orig_subseries_lookup = pickle.load(fp)
    sub_lookup = load_sub_lookup("patterns_comp/all", orig_subseries_lookup)

    for pid in pids:
        soln_lookup = {}
        parent_lookup = {}
        child_lookup = {}
        print(pid)
        if not os.path.exists("data/puzzle_solutions/solution_{}/{}_meta.h5".format(pid, pid)):
            process_puzzle_meta(pid)
        data, puz_metas = load_extend_data([pid], soln_lookup, parent_lookup, child_lookup, True, 0)
        pid_tracker.write("{}\n".format(pid))

        pattern_count_fn = partial(pattern_count_from_pdbs, soln_lookup=soln_lookup, best_k=best_k, noise=noise,
                                   mrf_lookup=mrf_lookup["all"][best_k], model_lookup=model_lookup["all"][best_k],
                                   sub_mrfs=sub_lookup["mrfs"][best_k], sub_models=sub_lookup["models"][best_k])
        collab = get_team_structures(data, soln_lookup, child_lookup)[pid]
        for root in collab:
            user_teams_out.write("{}\n".format(json.dumps(get_team_json(root, pid, pattern_count_fn))))
    
    pid_tracker.close()
    user_teams_out.close()
