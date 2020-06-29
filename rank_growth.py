import argparse
import csv
import os
import pickle
from collections import namedtuple, Counter
from itertools import groupby, takewhile
from operator import itemgetter
from typing import List

import numpy as np
from scipy import stats
from pprint import pprint
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import linregress
from datetime import datetime
import matplotlib.dates as mdates
from util import *

categories = {
    'overall' : '992758',
    'beginner' : '992759',
    'prediction' : '992760',
    'design' : '992761',
    'electron' : '994237',
    'contacts' : '997946',
    'symmetry' : '992769',
    'casp10' : '992762',
    'casp11' : '997398',
    'casp_roll' : '993715',
    'all': 'all', # dummy to allow select of all categorized puzzles
}

parser = argparse.ArgumentParser(prog='team_roles.py')
parser.add_argument('category', choices=categories.keys())
args = parser.parse_args()

puzzles = get_ranks("data/rprp_puzzle_ranks_v2")
# a bunch of empty/broken puzzle data that we'll filter out
# this appears to be things like test puzzles, ones that were reposted or beginner puzzles
pre_count = len(puzzles)
for pid in puzzles.copy():
    if len(puzzles[pid]['soloists']) == 0 or len(puzzles[pid]['evolvers']) == 0:
        puzzles.pop(pid)
print("discarded {} empty/broken puzzles".format(pre_count - len(puzzles)))
print("{} puzzles remaining".format(len(puzzles)))
puzzles_pdbs = puzzles.copy()
add_pdbs_to_ranks(puzzles_pdbs)
print("{} puzzles with pdbs".format(len(puzzles_pdbs)))

with open("data/puzzle_categories_v2.csv") as fp:
    cat_in = csv.DictReader(fp)
    for r in cat_in:
        pid = r['nid']
        if pid in puzzles:
            puzzles[pid]['categories'] = r['categories'].split(',')
            puzzles[pid]['categories'].append('all')

with open("data/puzzle_timestamps_v2.csv") as fp:
    time_in = csv.DictReader(fp)
    puzzles_created = {row['nid']: int(row['created']) for row in time_in}
pre_count = len(puzzles)
for pid in puzzles.copy():
    if pid not in puzzles_created:
        puzzles.pop(pid)
print("{} puzzles discarded for having no entry in rpnode table".format(pre_count - len(puzzles)))

with open("data/last_played_v3.csv") as fp:
    last_played = {}
    for r in csv.DictReader(fp):
        last_played.setdefault(r['pid'], {}).setdefault(r['uid'], int(r['lastplayed']))

SoloistRecord = namedtuple('SoloistRecord', 'uid pid rank perf score gid')

# performance approximated as fraction of highest soloist score
def get_participants(category, min_puzzles):
    cat_puzzles = {k:v for k, v in puzzles.items() if categories[category] in v['categories']}
    raw = [SoloistRecord(s['uid'],pid,s['rank'],s['best_score'] / p['soloists'][0]['best_score'],s['best_score'], s['gid']) 
           for pid,p in cat_puzzles.items() for s in p['soloists'] if s['best_score'] != 0]
    users = {k : sorted(g, key=lambda x: last_played[x.pid][x.uid]) for k,g in
             groupby(sorted(raw, key=lambda x: x.uid), lambda x: x.uid)}
    targets = {k: v for k,v in users.items() if len(v) > min_puzzles}
    return targets

EvolveRecord = namedtuple('EvolveRecord', 'uid pid rank perf score gid improve')

def get_evolvers(category):
    cat_puzzles = {k:v for k, v in puzzles.items() if categories[category] in v['categories']}
    raw = [EvolveRecord(s['uid'],pid,s['rank'],s['best_score'] / p['evolvers'][0]['best_score'], s['best_score'], s['gid'],
                        np.float64(s['best_score']) / next((x['best_score'] for x in p['soloists'] if x['gid'] == s['gid']), np.nan))
           for pid,p in cat_puzzles.items() for s in p['evolvers'] if s['gid'] != 'NULL']
    users = {k : sorted(g, key=lambda x: last_played[x.pid][x.uid]) for k,g in
             groupby(sorted(raw, key=lambda x: x.uid), lambda x: x.uid)}
    return users

GroupRecord = namedtuple('GroupRecord', 'gid type uid pid rank perf score pdl')

def get_groups(category, min_puzzles):
    cat_puzzles = {k:v for k, v in puzzles.items() if categories[category] in v['categories']}
    raw = [GroupRecord(g['gid'], g['type'], g['uid'], pid, g['rank'], g['score'] / p['groups'][0]['score'], g['score'],  None)
           for pid,p in cat_puzzles.items() for g in p['groups'] if g['score'] != 0]
    # pdbs exist for solutions entirely missing from the database, seems best to discard them
    groups = {k : sorted(g, key=lambda x: last_played[x.pid][x.uid]) for k,g in
             groupby(sorted(raw, key=lambda x: x.gid), lambda x: x.gid)}
    targets = {k: v for k,v in groups.items() if len(v) > min_puzzles}
    return targets

def get_groups_pdbs(category, min_puzzles):
    cat_puzzles = {k:v for k, v in puzzles_pdbs.items() if categories[category] in v['categories']}
    raw = [GroupRecord(g['gid'], g['type'], g['uid'], pid, g['rank'], g['score'] / p['groups'][0]['score'], g['score'], 
                       next((pdb for pdb in p['pdbs'] if pdb['ID'] == "solution_gid_{:04d}".format(g['rank'])), None))
           for pid,p in cat_puzzles.items() for g in p['groups'] if g['score'] != 0]
    # some PDLs are corrupted/missing and some have a comma in the group name, and so got parsed wrong
    raw = [r._replace(pdl=r.pdl['PDL']) for r in raw if r.pdl is not None and all(pdl['header']['uid'].isnumeric() for pdl in r.pdl['PDL'])]
    # pdbs exist for solutions entirely missing from the database, seems best to discard them
    groups = {k : sorted([x for x in g if x.uid in last_played[x.pid]],
                         key=lambda x: last_played[x.pid][x.uid]) for k,g in
             groupby(sorted(raw, key=lambda x: x.gid), lambda x: x.gid)}
    targets = {k: v for k,v in groups.items() if len(v) > min_puzzles}
    return targets

def rank_arr(records, cutoff = 15):
    return [1 if r.rank < cutoff else 0 for r in records]

def rank_frac(records, start, end, cutoff = 15):
    return sum(1 if r.rank < cutoff else 0 for r in records[start:end]) / (end - start)

# takes list of records
def cum_rank_frac(records, cutoff: int = 15):
    output = []
    arr = rank_arr(records, cutoff)
    for i in range(1, len(records) + 1):
        output.append(sum(arr[:i]) / i)
    return output

# takes list of records
def cum_rank_median(records):
    output = []
    for i in range(1, len(records) + 1):
        output.append(np.median([x.rank for x in records[:i]]))
    return output

def improv_arr(records):
    rank_med = cum_rank_median(records)
    return [1 if records[i + 1].rank < rank_med[i] else 0 for i in range(0, len(records) - 1)]

def improv_frac(records, meds, start, end, fn = lambda r,m: r.rank < m):
    return sum(1 if fn(records[i + 1], meds[i]) else 0 for i in range(start, end)) / (end - start)

def cum_improv_frac(records):
    output = []
    arr = improv_arr(records)
    for i in range(1, len(records)):
        output.append(sum(arr[:i]) / i)
    return output

def perf_arr(records, cutoff):
    return [1 if r.perf > cutoff else 0 for r in records]

def perf_frac(records, start, end, cutoff):
    return sum(1 if r.perf > cutoff else 0 for r in records[start:end]) / (end - start)

def cum_perf_frac(records, cutoff):
    output = []
    arr = perf_arr(records, cutoff)
    for i in range(1, len(records) + 1):
        output.append(sum(arr[:i]) / i)
    return output

def cum_perf_median(records):
    output = []
    for i in range(1, len(records) + 1):
        output.append(np.median([x.perf for x in records[:i]]))
    return output


print('gathering players')
players_raw = get_participants(args.category, 50)
evolvers_raw = get_evolvers(args.category)
PlayerStats = namedtuple('PlayerStats', ['uid', 'records', 'evolve_records', 'rank_frac', 'rank_med', 'perf_frac', 'perf_med',
                                         'last_played', 'last_played_evlove', 'group_count', 'group_changes'])
players = {uid: PlayerStats(uid, records, evolvers_raw[uid] if uid in evolvers_raw else [], np.array(cum_rank_frac(records)),
                            np.array(cum_rank_median(records)),
                            np.array(cum_perf_frac(records, 0.99)), np.array(cum_perf_median(records)),
                            np.array([datetime.fromtimestamp(last_played[x.pid][x.uid]) for x in records]),
                            np.array([datetime.fromtimestamp(last_played[x.pid][x.uid]) for x in evolvers_raw[uid]]) if uid in evolvers_raw else [],
                            Counter(r.gid for r in records), []) for uid,records in players_raw.items()}

print('gathering groups')
GroupStats = namedtuple('GroupStats', ['gid', 'records', 'rank_frac', 'rank_med', 'perf_frac', 'perf_med',
                                       'last_played'])

groups_raw = get_groups(args.category, 4)
groups = {gid: GroupStats(gid, records, np.array(cum_rank_frac(records, 2)),
                          np.array(cum_rank_median(records)),
                          np.array(cum_perf_frac(records, 0.99)), np.array(cum_perf_median(records)),
                          np.array([datetime.fromtimestamp(last_played[x.pid][x.uid]) for x in records]))
          for gid,records in groups_raw.items()}

groups_pdbs_raw = get_groups_pdbs(args.category, 4)
groups_pdbs = {gid: GroupStats(gid, records, np.array(cum_rank_frac(records, 2)),
                          np.array(cum_rank_median(records)),
                          np.array(cum_perf_frac(records, 0.99)), np.array(cum_perf_median(records)),
                          np.array([datetime.fromtimestamp(last_played[x.pid][x.uid]) for x in records]))
          for gid,records in groups_pdbs_raw.items()}

top = [player for player in players.values() if np.median([x.rank for x in player.records]) < 100]

def normalize_counter(orig):
    c = orig.copy()
    total = sum(c.values())
    for k in c:
        c[k] /= total
    return c

# count as a group-switcher any player who was a member of multiple groups (including no group) and where
# their most common group accounts for less than 85% of their prediction puzzles
# 85% was chosen arbitrarily, but goal is to ensure the presence of interesting inflection points
switchers = [p for p in players.values() if len(p.group_count) > 1 and 'NULL' == p.records[0].gid and 0.9 > len(list(takewhile(lambda r: r.gid == 'NULL', p.records))) / len(p.records) > 0.1]
loners = [p for p in players.values() if len(p.group_count) == 1 and 'NULL' in p.group_count]

# this analysis indicates 'gaps' of NULL gid between records with the same non-NULL gid consist of one to three puzzles
# inflections = [(s, [i for i in range(len(s.records) - 1) if s.records[i].gid != s.records[i+1].gid]) for s in switchers]
# for s,ii in inflections:
#     print([(s.records[i].gid, j - i, s.records[j+1].gid) for i,j in zip(ii,ii[1:]) if s.records[i+1].gid == 'NULL'])
# given this, we repair these gaps by filling them in with the surrounding gid
print('cleaning gid gaps')
for s in switchers:
    ii = [i for i in range(len(s.records) - 1) if s.records[i].gid != s.records[i+1].gid]
    for i,j in zip(ii,ii[1:]):
        if s.records[i+1].gid == 'NULL' and s.records[i].gid == s.records[j+1].gid and j - i < 4:
            fill_gid = s.records[i].gid
            for k in range(i+1,j+1):
                s.records[k] = s.records[k]._replace(gid=fill_gid)
    s.group_changes.extend([i+1 for i in range(len(s.records) - 1) if s.records[i].gid != s.records[i+1].gid])

simple_switchers = [s for s in switchers if len(s.group_changes) == 1 and 'NULL' in s.group_count and s.records[0].gid == 'NULL']
