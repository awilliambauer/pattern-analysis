import os
import numpy as np
import pandas as pd
from itertools import combinations, groupby, takewhile, accumulate
from util import time_played, EnergyComponent, load_frame
from multiprocessing import Pool
import argparse
import pickle
import sys

descendants_memo = {}
def get_descendants(s):
    if s.sid in descendants_memo:
        return descendants_memo[s.sid]
    # soln_lookup is generated from the list of solutions passed in which are all from a single user
    # the history may include evolver children, which we have to avoid trying to look up
    children = [soln_lookup[c] for c in child_lookup[s.sid] if c in soln_lookup] if s.sid in child_lookup else []
    descendants_memo[s.sid] = children + [d for c in children for d in get_descendants(c)]
    return descendants_memo[s.sid]


def get_origin(s):
    """
    For an evolver solution s, return the first solution from this user that is an ancestor of s
    """
    if s.scoretype != '2':
        raise ValueError("solution must be from an evolver to have an origin")
    if not s.parent_sid:
        raise ValueError("can't find the origin for a solution with no parent")
    cur = soln_lookup[s.parent_sid]
    while soln_lookup[cur.parent_sid].uid == s.uid:
        cur = soln_lookup[cur.parent_sid]
    return cur


def get_time(s):
    if s.scoretype == '1':
        i = np.searchsorted([x.timestamp for x in user_solns[s.uid]], s.timestamp)
        ts = [x.timestamp for x in user_solns[s.uid][:i + 1]]
        t = time_played(ts)
        return t
    try:
        origin = get_origin(s)
        evolves = [x for x in get_descendants(origin) if x.uid == s.uid and x.timestamp <= s.timestamp]
        ts = sorted([x.timestamp for x in evolves])
        t = time_played(ts)
        return t + get_time(soln_lookup[origin.parent_sid])
    except KeyError:
        return np.nan


def get_time_wrapper(s):
    return s.sid, get_time(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='collab_viz.py')
    parser.add_argument('pid')
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    pid = args.pid

    sys.setrecursionlimit(10000)

    if os.path.exists("data/puzzle_solutions/solution_{}/{}_times.pickle".format(pid, pid)) and not args.overwrite:
        print("data/puzzle_solutions/solution_{}/{}_times.pickle".format(pid, pid), "already exists")
        sys.exit(0)

    print("loading frame for", pid)
    df, bts, puz = load_frame("data/puzzle_solutions/solution_{}/{}_meta.h5".format(pid, pid))

    print("assembling solutions")
    infos = df.apply(lambda r: sorted(([p for l in r.lines for p in l.pdb_infos] if r.lines else []) +
                                      ([p for l in r.evol_lines for p in l.pdb_infos] if r.evol_lines else []),
                                      key=lambda p: p.timestamp), axis=1)
    soln_lookup = {}
    parent_lookup = {}
    for _, xs in infos.items():
        for x in xs:
            soln_lookup[x.sid] = x
            if x.parent_sid:
                parent_lookup[x.sid] = x.parent_sid
    child_lookup = {parent: [c for p, c in g] for parent, g in
                    groupby(sorted([(p, c) for c, p in parent_lookup.items()]), lambda x: x[0])}

    user_solns = {uid: sorted(g, key=lambda x: x.timestamp) for uid, g in
                  groupby(sorted(soln_lookup.values(), key=lambda x: x.uid), lambda x: x.uid)}

    print("computing times")
    with Pool(40) as pool:
        time_lookup = dict(pool.map(get_time_wrapper, soln_lookup.values()))

    print("lookup failed for", len([x for x in time_lookup.values() if np.isnan(x)]))
    with open("data/puzzle_solutions/solution_{}/{}_times.pickle".format(pid, pid), 'wb') as fp:
        pickle.dump(time_lookup, fp)