from pattern_extraction import load_extend_data, get_deltas
from util import collect_pdl_entries, time_played, get_action_labels, get_action_keys
import csv
import argparse
import logging
import os
import sys
import numpy as np
from process_puzzle_meta import process_puzzle_meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='dump_ubiq.py')
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--append", action='store_true')
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    sys.setrecursionlimit(10000)
    version_tag = "v1"
    user_filename = "data/user_ubiq_{}.csv".format(version_tag)
    pid_tracker_filename = "pid_tracker_ubiq_{}.txt".format(version_tag)

    if not args.overwrite and not args.append and os.path.exists(user_filename):
        logging.info("{} already exists".format(user_filename))
        sys.exit(1)

    if args.overwrite and args.append:
        logging.info("cannot both --overwrite and --append")
        sys.exit(2)

    pids = np.loadtxt("wannacut_prediction_pids_v6.txt", dtype=np.str).tolist()

    if args.append:
        user_ubiq_fp = open(user_filename, 'a')
        user_ubiq_out = csv.DictWriter(user_ubiq_fp, ["uid", "pid"] + 
                                       [a + "_ubiq_all" for a in get_action_labels()] + 
                                       [a + "_ubiq_relevant" for a in get_action_labels()] + 
                                       [a + "_frac_best" for a in get_action_labels()])

        pid_tracker = open(pid_tracker_filename, "r+")
        for pid in pid_tracker:
            logging.info("skipping {}, existing entry".format(r))
            try:
                pids.remove(pid)
            except: # as list of pids is revised, we might try to remove previous entry that is no longer in the list
                logging.info("previously included {} no longer on list".format(r["pid"]))
    else:
        user_ubiq_fp = open(user_filename, 'w')
        user_ubiq_out = csv.DictWriter(user_ubiq_fp, ["uid", "pid"] + 
                                       [a + "_ubiq_all" for a in get_action_labels()] + 
                                       [a + "_ubiq_relevant" for a in get_action_labels()] + 
                                       [a + "_frac_best" for a in get_action_labels()])
        user_ubiq_out.writeheader()

        pid_tracker = open(pid_tracker_filename, 'w')

    for pid in pids:
        soln_lookup = {}
        parent_lookup = {}
        child_lookup = {}
        print(pid)
        if not os.path.exists("data/puzzle_solutions/solution_{}/{}_meta.h5".format(pid, pid)):
            process_puzzle_meta(pid)
        data, puz_metas = load_extend_data([pid], soln_lookup, parent_lookup, child_lookup, False, 0)
        pid_tracker.write("{}\n".format(pid))
        for i, r in data.iterrows():
            print(i+1, "out of", len(data), "\r", end="")
            if r.relevant_sids:
                row = {"uid": r.uid, "pid": r.pid}
                deltas_all = [d for line in r.lines for d in get_deltas(sorted(line.pdb_infos, key=lambda p: p.timestamp), soln_lookup)]
                deltas_relevant = [d for d in deltas_all if d.sid in r.relevant_sids]
                best_actions, _ = collect_pdl_entries(soln_lookup[min(r.relevant_sids, key=lambda sid: soln_lookup[sid].energy)])
                action_sum = sum(best_actions.get(a, 0) for actions in get_action_keys() for a in actions)
                for label, actions in zip(get_action_labels(), get_action_keys()):
                    row[label + "_ubiq_all"] = len([d for d in deltas_all if 
                                                    sum(d.action_diff.get(a, 0) for a in actions) > 0]) / len(deltas_all)
                    row[label + "_ubiq_relevant"] = len([d for d in deltas_relevant if 
                                                         sum(d.action_diff.get(a, 0) for a in actions) > 0]) / len(deltas_relevant)
                    row[label + "_frac_best"] = (sum(best_actions.get(a, 0) for a in actions) / action_sum) if action_sum > 0 else np.nan
                user_ubiq_out.writerow(row)
        print()

    pid_tracker.close()
    user_ubiq_fp.close()
