from pattern_extraction import get_relevant_sids, get_deltas, make_lookups
from util import load_frame
import csv
import pandas as pd
import argparse
import logging
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='dump_events.py')
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("datapath")
    parser.add_argument("pid")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    logging.info("dumping {}".format(args.pid))

    if not args.overwrite and os.path.exists("{}/solution_{}/foldit_user_events_{}.csv".format(args.datapath, args.pid, args.pid)):
        logging.info("{}/solution_{}/foldit_user_events_{}.csv already exists".format(args.datapath, args.pid, args.pid))
        sys.exit(1)

    logging.info("loading data")
    data, bts, puz = load_frame("{}/solution_{}/{}_meta.h5".format(args.datapath, args.pid, args.pid))

    soln_lookup = {}
    parent_lookup = {}
    child_lookup = {}
    make_lookups(data, soln_lookup, parent_lookup, child_lookup)

    descendants_memo = {}

    def get_descendants(s):
        sid = s.sid
        if sid in descendants_memo:
            return descendants_memo[sid]
        # soln_lookup is generated from the list of solutions passed in which are all from a single user
        # the history may include evolver children, which we have to avoid trying to look up
        children = [soln_lookup[c] for c in child_lookup[sid] if c in soln_lookup] if sid in child_lookup else []
        descendants_memo[sid] = children + [d for c in children for d in get_descendants(c)]
        return descendants_memo[sid]

    logging.info("computing deltas")
    extra_data = []
    for _, record in data.iterrows():
        best_line = min(record.lines, key=lambda l: min(x.energy for x in l.pdb_infos)) if record.lines else None
        extra_data.append({'pid': record.pid,
                           'uid': record.uid,
                           'relevant_sids': get_relevant_sids(min(best_line.pdb_infos, key=lambda x: x.energy, default=None),
                                                              soln_lookup, child_lookup) if best_line else None,
                           'deltas': get_deltas(sorted(best_line.pdb_infos, key=lambda p: p.timestamp) if best_line else [], soln_lookup)})
    data = data.merge(pd.DataFrame(data=extra_data), on=['pid', 'uid'])

    logging.info("writing csv")
    with open("{}/solution_{}/foldit_user_events_{}.csv".format(args.datapath, args.pid, args.pid), "w") as fp:
        with open("{}/solution_{}/foldit_outcomes_{}.csv".format(args.datapath, args.pid, args.pid), "w") as outcomes_fp:
            events_out = csv.DictWriter(fp, ["puzzle_id", "user_id", "timestamp", "tool"])
            events_out.writeheader()
            outcomes_out = csv.DictWriter(outcomes_fp, ["puzzle_id", "user_id", "timestamp", "current_energy", "best_energy_so_far"])#, "best_descendant"])
            outcomes_out.writeheader()            
            for _, r in data.iterrows():
                if r.relevant_sids:
                    best_energy = float('inf')
                    for d in sorted([d for d in r.deltas if d.sid in r.relevant_sids], key=lambda x: x.timestamp):
                        if best_energy > soln_lookup[d.sid].energy:
                            best_energy = soln_lookup[d.sid].energy
                        outcomes_out.writerow({"puzzle_id": r.pid, "user_id": r.uid, "timestamp": d.timestamp, "current_energy": soln_lookup[d.sid].energy, "best_energy_so_far": best_energy})#, "best_descendant": min([x.energy for x in get_descendants(d)], default=float("nan"))})
                        for action, count in d.action_diff.items():
                            for i in range(count):
                                events_out.writerow(
                                    {"puzzle_id": r.pid, "user_id": r.uid, "timestamp": d.timestamp, "tool": action})
