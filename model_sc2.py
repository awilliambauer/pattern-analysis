import csv
import numpy as np
import math
from typing import NamedTuple
from itertools import groupby
import matplotlib
matplotlib.use("Agg")
from pattern_extraction import *

class SC2Event(NamedTuple):
    uid: str
    game_id: str
    frame: int
    type: str

features = [
    "TrainWorker",
    "TrainT1",
    "TrainT2",
    "TrainT3",
    "BuildStandard",
    "BuildSupply",
    "BuildStaticDefense",
    "BuildEcon",
    "OrderEcon",
    "OrderMilitary"
]

krange = [5,6,7,8,9,10,12,15,20]

raw = []
with open("starcraft/sc2_test_events.csv") as fp:
    inr = csv.DictReader(fp)
    for r in inr:
        raw.append(SC2Event(r["uid"], r["game_id"], int(r["frame"]), r["type"]))

events = {k: {gid: sorted(xs, key=lambda e: e.frame) for gid, xs in groupby(sorted(es, key=lambda e: e.game_id), lambda e: e.game_id)} for k, es in groupby(sorted(raw, key=lambda e: e.uid), lambda e: e.uid)}

series_lookup = {}
puz_idx_lookup = {}
binwidth = 22.4
noise = [100] * len(features)
for uid, d in events.items():
    s = []
    i = 1
    print(uid)
    for pid, es in d.items():
        if len(es) == 1:
            continue
        print(pid, len(es))
        num_bins = math.floor((es[-1].frame - es[0].frame) / binwidth)
        bins = np.linspace(es[0].frame, es[-1].frame, num_bins + 1)
        for start, end in zip(bins, bins[1:]):
            in_bin = [e for e in es if start <= e.frame < end + 1]
            s.append([len([e for e in in_bin if e.type == f]) for f in features])
        puz_idx_lookup[(uid, pid)] = (len(s) - num_bins, len(s))
        print(len(s))
        if i < len(d):
            s.extend(np.tile(noise, (100, 1)))
        i += 1
    if len(s) > 0:
        print(len([r for r in s if sum(r) == 0]) / len(s))
        series_lookup[uid] = np.array(s)

idx_lookup, all_series = combine_user_series(series_lookup, noise, 200)
puz_idx_lookup = {(uid, pid): (s + idx_lookup[uid][0], e + idx_lookup[uid][0]) for (uid, pid), (s, e) in puz_idx_lookup.items() if uid in idx_lookup}
run_TICC({"all": all_series}, "sc2", krange)
