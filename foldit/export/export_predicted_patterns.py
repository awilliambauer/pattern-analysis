from foldit.foldit_data import make_action_series, get_deltas, load_extend_data, make_series
from pattern_extraction import *
from foldit.check_models import predict_from_saved_model, load_sub_lookup
from util import get_pattern_label
import csv
import argparse
import logging
import os
import sys
import numpy as np
import pickle
from itertools import groupby
from foldit.raw.process_puzzle_meta import process_puzzle_meta


def get_pattern_count(p, pt, series, subseries_lookup):
    if pt.isnumeric():
        return int(series[p.start_idx:p.end_idx].sum())
    else:
        return int(subseries_lookup[int(pt[:-1])]["series"][p.start_idx:p.end_idx].sum())


def pattern_count_from_pdbs(pdbs, soln_lookup, best_k, noise, mrf_lookup, model_lookup, sub_mrfs, sub_models):
    best_subs = ((1, 10), (2, 10), (3, 10), (4, 0))
    target_pts = ["1A", "1D", "1E", "1F", "1G", "1H", "1I", "1J", "2A", "2C", "2D", "2E", "2F", "2G",
                  "2H", "2I", "2J", "3A", "3C", "3D", "3E", "3F", "3G", "3H", "3I", "3J", "4"]
    # for a single series
    uid = pdbs[0].uid
    pid = pdbs[0].pid
    ser = make_action_series(get_deltas(pdbs, soln_lookup))
    if len(ser) <= 1:
        return {}
    puz_idx_lookup = {(uid, pid): (0, len(ser))}
    cs = predict_from_saved_model(ser, model_lookup)
    patterns = get_patterns(mrf_lookup, cs, puz_idx_lookup)
    subseries_lookup = make_subseries_lookup(best_k, patterns, mrf_lookup, ser, noise)
    sub_cs = {}
    for cid in range(len(mrf_lookup)):
        if cid in subseries_lookup:
            sub_cs[cid] = predict_from_saved_model(subseries_lookup[cid]["series"], sub_models[cid][10])

    patterns = {"base": get_patterns(mrf_lookup, cs, puz_idx_lookup)}
    cids = {p.cid for p in patterns["base"]}
    for cid in cids:
        patterns[cid] = {0: [p for p in patterns["base"] if p.cid == cid]}
        if cid not in subseries_lookup:
            continue
        ps = patterns[cid][0]
        sub_idx_lookup = subseries_lookup[cid]["idx_lookup"]
        patterns[cid][10] = get_patterns(sub_mrfs[cid][10], sub_cs[cid], sub_idx_lookup)

    ps = sum([[(get_pattern_label(p, cid, sub_k), p) for p in patterns[cid][sub_k]] for cid, sub_k in best_subs if cid in patterns], [])
    pc = {pt: sum(get_pattern_count(y, pt, ser, subseries_lookup) for _, y in ys)
          for pt, ys in groupby(sorted(ps), lambda x: x[0])}
    return pc


def compute_pattern_counts(pid, best_k, target_pts, noise, mrf_lookup, model_lookup, sub_mrfs, sub_models):
    best_subs = ((1, 10), (2, 10), (3, 10), (4, 0))

    soln_lookup = {}
    parent_lookup = {}
    child_lookup = {}
    if not os.path.exists("data/puzzle_solutions/solution_{}/{}_meta.h5".format(pid, pid)):
        process_puzzle_meta(pid)
    data, puz_metas = load_extend_data([pid], soln_lookup, parent_lookup, child_lookup, True, 0)

    puz_idx_lookup, series_lookup, _ = make_series(data, noise)
    idx_lookup, all_series = combine_user_series(series_lookup, noise)
    puz_idx_lookup = {(uid, pid): (s + idx_lookup[uid[:uid.index('e')] if 'e' in uid else uid][0],
                                   e + idx_lookup[uid[:uid.index('e')] if 'e' in uid else uid][0])
                       for (uid, pid), (s, e) in puz_idx_lookup.items()}
    
    cs = predict_from_saved_model(all_series, model_lookup)
    patterns = get_patterns(mrf_lookup, cs, puz_idx_lookup)
    subseries_lookup = make_subseries_lookup(best_k, patterns, mrf_lookup, all_series, noise)
    sub_cs = {}
    for cid in range(len(mrf_lookup)):
        if cid in subseries_lookup:
            sub_cs[cid] = predict_from_saved_model(subseries_lookup[cid]["series"], sub_models[cid][10])
    
    patterns = {"base": get_patterns(mrf_lookup, cs, puz_idx_lookup)}
    cids = {p.cid for p in patterns["base"]}
    for cid in cids:
        patterns[cid] = {0: [p for p in patterns["base"] if p.cid == cid]}
        if cid not in subseries_lookup:
            continue
        ps = patterns[cid][0]
        sub_idx_lookup = subseries_lookup[cid]["idx_lookup"]
        patterns[cid][10] = get_patterns(sub_mrfs[cid][10], sub_cs[cid], sub_idx_lookup)
    
    ps = sum([[(get_pattern_label(p, cid, sub_k), p) for p in patterns[cid][sub_k]] for cid, sub_k in best_subs], [])
    ps_uid_pid = {k: sorted(xs) for k, xs in groupby(sorted(ps, key=lambda p: (p[1].uid, p[1].pid)), lambda p: (p[1].uid, p[1].pid))}
    pattern_count_lookup = {k: {pt: sum(get_pattern_count(y, pt, all_series, subseries_lookup) for _, y in ys) 
                                for pt, ys in groupby(xs, lambda x: x[0])} for k, xs in ps_uid_pid.items()}
    
    rows = []
    for (uid, pid), pc in pattern_count_lookup.items():
        row = {"uid": uid, "pid": pid}
        for pt in target_pts:
            row[pt] = pc.get(pt, 0)
        rows.append(row)
    return rows



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='export_predicted_patterns.py')
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--append", action='store_true')
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    sys.setrecursionlimit(10000)
    version_tag = "v1"
    best_k = 5
    target_pts = ["1A", "1D", "1E", "1F", "1G", "1H", "1I", "1J", "2A", "2C", "2D", "2E", "2F", "2G",
                  "2H", "2I", "2J", "3A", "3C", "3D", "3E", "3F", "3G", "3H", "3I", "3J", "4"]
    user_filename = "data/user_patterns_{}.csv".format(version_tag)
    pid_tracker_filename = "pid_tracker_patterns_{}.txt".format(version_tag)

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
        user_patterns_fp = open(user_filename, 'a')
        user_patterns_out = csv.DictWriter(user_patterns_fp, ["uid", "pid"] + target_pts)

        pid_tracker = open(pid_tracker_filename, "r+")
        for pid in pid_tracker:
            pid = pid.strip()
            logging.info("skipping {}, existing entry".format(pid))
            try:
                pids.remove(pid)
            except: # as list of pids is revised, we might try to remove previous entry that is no longer in the list
                logging.info("previously included {} no longer on list".format(pid))
    else:
        user_patterns_fp = open(user_filename, 'w')
        user_patterns_out = csv.DictWriter(user_patterns_fp, ["uid", "pid"] + target_pts)
        user_patterns_out.writeheader()

        pid_tracker = open(pid_tracker_filename, 'w')

    noise = np.loadtxt("patterns_comp/noise_values.txt")
    _, mrf_lookup, model_lookup, _ = load_TICC_output("patterns_comp", ["all"], [best_k])
    with open("patterns_comp/all/subpatterns/subseries_lookup.pickle", 'rb') as fp:
        orig_subseries_lookup = pickle.load(fp)
    sub_lookup = load_sub_lookup("patterns_comp/all", orig_subseries_lookup)

    for pid in pids:
        print(pid)
        pid_tracker.write("{}\n".format(pid))
        rows = compute_pattern_counts(pid, best_k, target_pts, noise, mrf_lookup["all"][best_k], model_lookup["all"][best_k],
                                      sub_lookup["mrfs"][best_k], sub_lookup["models"][best_k])
        for row in rows:
            user_patterns_out.writerow(row)

    pid_tracker.close()
    user_patterns_fp.close()
