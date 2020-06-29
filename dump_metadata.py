from pattern_extraction import load_extend_data
from util import collect_pdl_entries, time_played
import csv
import argparse
import logging
import os
import sys
import numpy as np
from process_puzzle_meta import process_puzzle_meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='dump_metadata.py')
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--append", action='store_true')
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    sys.setrecursionlimit(10000)
    version_tag = "v4"
    user_filename = "data/user_metadata_{}.csv".format(version_tag)
    puz_filename = "data/puz_metadata_{}.csv".format(version_tag)

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
        user_metas_fp = open(user_filename, 'a')
        user_metas_out = csv.DictWriter(user_metas_fp, ["uid", "pid", "time", "relevant_time", "action_count_all",
                                                        "action_count_relevant", "action_count_best", "best_energy",
                                                        "best_energy_time", "solo_perf", "perf"])
        puz_metas_fp = open(puz_filename, 'r+')
        puz_metas_in = csv.DictReader(puz_metas_fp)
        for r in puz_metas_in:
            logging.info("skipping {}, existing entry".format(r["pid"]))
            try:
                pids.remove(r["pid"])
            except: # as list of pids is revised, we might try to remove previous entry that is no longer in the list
                logging.info("previously included {} no longer on list".format(r["pid"]))
        puz_metas_out = csv.DictWriter(puz_metas_fp, ["pid", "start", "end", "baseline", "best_solo", "best"])

    else:
        user_metas_fp = open(user_filename, 'w')
        user_metas_out = csv.DictWriter(user_metas_fp, ["uid", "pid", "time", "relevant_time", "action_count_all",
                                                        "action_count_relevant", "action_count_best", "best_energy",
                                                        "best_energy_time", "solo_perf", "perf"])
        user_metas_out.writeheader()
        puz_metas_fp = open(puz_filename, 'w')
        puz_metas_out = csv.DictWriter(puz_metas_fp, ["pid", "start", "end", "baseline", "best_solo", "best"])
        puz_metas_out.writeheader()

    for pid in pids:
        soln_lookup = {}
        parent_lookup = {}
        child_lookup = {}
        print(pid)
        if not os.path.exists("data/puzzle_solutions/solution_{}/{}_meta.h5".format(pid, pid)):
            process_puzzle_meta(pid)
        data, puz_metas = load_extend_data([pid], soln_lookup, parent_lookup, child_lookup, True, 0)
        energy_baseline = puz_metas[pid].energy_baseline
        overall_best_energy = puz_metas["evolver"][pid]["best"]
        puz_metas_out.writerow({"pid": pid, "baseline": energy_baseline, "start": data.timestamps.apply(min).min(),
                                "end": data.timestamps.apply(max).max(),
                                "best_solo": data.energies.apply(np.min).min(), "best": overall_best_energy})
        for i, r in data.iterrows():
            print(i+1, "out of", len(data), "\r", end="")
            if r.relevant_sids:
                row = {"uid": r.uid, "pid": r.pid, "time": r.time, "relevant_time": r.relevant_time, "solo_perf": r.perf,
                       "best_energy": r.energies.min()}
                row["perf"] = min(0, row["best_energy"] - energy_baseline) / (overall_best_energy - energy_baseline)
                row["action_count_all"] = sum(d.action_count for d in r.deltas)
                row["action_count_relevant"] = sum(d.action_count for d in r.deltas if d.sid in r.relevant_sids)
                best_soln = soln_lookup[min(r.deltas, key=lambda d: soln_lookup[d.sid].energy).sid]
                best_actions, _ = collect_pdl_entries(best_soln)
                row["action_count_best"] = sum(best_actions.values())
                row["best_energy_time"] = time_played(sorted([d.timestamp for d in r.deltas if d.timestamp <= best_soln.timestamp]))
                user_metas_out.writerow(row)
            if r.relevant_evol_sids:
                for evol_count, (relevant_sids, deltas) in enumerate(zip(r.relevant_evol_sids, r.deltas_evol)):
                    if relevant_sids:
                        row = {"uid": r.uid + "evol" + str(evol_count), "pid": r.pid}
                        best_soln = soln_lookup[min(deltas, key=lambda d: soln_lookup[d.sid].energy).sid]
                        best_actions, _ = collect_pdl_entries(best_soln)
                        row["best_energy"] = best_soln.energy
                        row["perf"] = min(0, row["best_energy"] - energy_baseline) / (overall_best_energy - energy_baseline)
                        row["time"] = time_played(sorted([d.timestamp for d in deltas]))
                        row["relevant_time"] = time_played(sorted([d.timestamp for d in deltas if d.sid in relevant_sids]))
                        row["action_count_all"] = sum(d.action_count for d in deltas)
                        row["action_count_relevant"] = sum(d.action_count for d in deltas if d.sid in relevant_sids)
                        row["action_count_best"] = sum(best_actions.values())
                        row["best_energy_time"] = time_played(sorted([d.timestamp for d in deltas if d.timestamp <= best_soln.timestamp]))
                        user_metas_out.writerow(row)

        print()

    puz_metas_fp.close()
    user_metas_fp.close()
