from pattern_extraction import *
import argparse
import numpy as np
import pandas as pd
import logging
import json
import sys
import os
import string
import re
from typing import Dict, Tuple, List
from itertools import combinations, groupby, chain
from util import category_lookup
from plot_util import make_boxplot
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("data/user_ubiq_v1.csv") as fp:
    ubiqs = [{k: float(v) if not k.endswith("id") else v for k, v in r.items()}
             for r in csv.DictReader(fp)]
ubiq_lookup = {uid: sorted(us, key=lambda x: x['pid']) for uid, us in groupby(sorted(ubiqs, key=lambda u: u['uid']), lambda u: u['uid'])}

with open("data/user_metadata_v4.csv") as fp:
    user_metas = {(r['uid'], r['pid']): r for r in csv.DictReader(fp)}
    for v in user_metas.values():
        v['time'] = int(v['time'])
        v['relevant_time'] = int(float(v['relevant_time']))
        v['best_energy_time'] = int(v['best_energy_time'])
        v['action_count_all'] = int(v['action_count_all'])
        v['action_count_relevant'] = int(v['action_count_relevant'])
        v['action_count_best'] = int(v['action_count_best'])
        v['best_energy'] = float(v['best_energy'])
        v['perf'] = float(v['perf'])
        v['solo_perf'] = float(v['solo_perf']) if v['solo_perf'] != "" else np.nan
user_meta_lookup = {uid: sorted(metas, key=lambda x: x['pid']) for uid, metas in groupby(sorted(user_metas.values(), key=lambda m: m['uid']),
                                                                                         lambda m: m['uid'])}

with open("data/puz_metadata_v4.csv") as fp:
    puz_infos = {r['pid']: {'start':     int(r['start']),
                            'end':       int(r['end']),
                            'baseline':  float(r['baseline']),
                            'best':      float(r['best']),
                            'best_solo': float(r['best_solo'])
                           } for r in csv.DictReader(fp)}

with open("data/user_patterns_v1.csv") as fp:
    reader = csv.DictReader(fp)
    pattern_count_lookup = {}
    for r in reader:
        pattern_count_lookup[(r['uid'], r['pid'])] = {pt: float(c) for pt, c in r.items() if pt != 'uid' and pt != 'pid'}
uid_to_pattern_fracs = {uid: {pid: {pt: c / sum(pattern_count_lookup[(uid, pid)].values()) for pt, c in pattern_count_lookup[(uid, pid)].items()}
                              for uid, pid in tags if sum(pattern_count_lookup[(uid, pid)].values()) > 0}
                        for uid, tags in groupby(sorted(pattern_count_lookup.keys()), lambda x: x[0]) if 'evol' not in uid}

with open("data/puzzle_categories_v4.csv") as fp:
    puz_cat = {r['nid']: r['categories'].split(',') for r in csv.DictReader(fp)}

with open("data/puzzle_labels_v1.json") as fp:
    puz_labels = {x['pid']: x for x in json.load(fp)}


# def make_markers(uid):
#     markers = {"r.": [re.match(r"\d\d\d\d: Revisiting", puz_labels[ubiq['pid']]["title"]) != None for ubiq in ubiq_lookup[uid]],
#                "b.": [re.match(r"\d\d\d\d: Unsolved", puz_labels[ubiq['pid']]["title"]) != None and puz_labels[ubiq['pid']]["title"].count(":") == 1 for ubiq in ubiq_lookup[uid]]}
#     markers["g."] = [not a and not b for a, b in zip(markers["r."], markers["b."])]
#     return markers


# target_uids = ['716281', '236506', '476462', '447652', '306768', '513977', '492685', '482875', '455069', '126983', '71954', '398373']
# for uid in target_uids:
#     make_boxplot([[ubiq[a + "_ubiq_relevant"] for ubiq in ubiq_lookup[uid]] for a in get_action_labels()],
#                  [a for a in get_action_labels()], "ubiq", "variation_viz/uid_{}.png".format(uid), markers=make_markers(uid))

denovo_pids = [pid for pid in puz_labels if re.match(r"\d\d\d\d: Unsolved", puz_labels[pid]["title"]) != None and puz_labels[pid]["title"].count(":") == 1]
revisit_pids = [pid for pid in puz_labels if re.match(r"\d\d\d\d: Revisiting", puz_labels[pid]["title"]) != None]
target_pids = denovo_pids + revisit_pids

target_pts = ["1A", "1D", "1E", "1F", "1G", "1H", "1I", "1J", "2A", "2C", "2D", "2E", "2F", "2G",
              "2H", "2I", "2J", "3A", "3C", "3D", "3E", "3F", "3G", "3H", "3I", "3J", "4"]

experience_cohorts = {"exp{}".format(thresh): [uid for uid, pfs in uid_to_pattern_fracs.items() if len([pid for pid in pfs if pid in target_pids]) > thresh] for thresh in [5, 10, 25, 50]}
# perf_cohorts = {"perf{}".format(thresh): [uid for uid, pfs in uid_to_pattern_fracs.items() if
#                                                                        len([pid for pid in pfs if pid in target_pids]) > 5
#                                                                        and np.median([x['perf'] for x in user_meta_lookup[uid] if x['pid'] in target_pids]) > thresh]
#                 for thresh in [0.8, 0.9, 0.95]}

for label, uids in experience_cohorts.items(): #chain(experience_cohorts.items(), perf_cohorts.items()):
    print("{} (n = {})".format(label, len(uids)))
    print("correlations of IQR with perf")
    for a in get_action_labels():
        print(a, stats.spearmanr([stats.iqr([u[a + "_ubiq_relevant"] for u in us if u['pid'] in target_pids]) for uid, us in ubiq_lookup.items() if uid in uids],
                                 [np.median([x['perf'] for x in user_meta_lookup[uid] if x['pid'] in target_pids]) for uid, us in ubiq_lookup.items() if uid in uids],
                                 nan_policy="omit"))
    print()
    for pt in target_pts:
        pt_uids = [uid for uid in uids if any(pf[pt] > 0 for pid, pf in uid_to_pattern_fracs[uid].items())]
        print(pt, stats.spearmanr([stats.iqr([pf[pt] for pid, pf in pfs.items() if pid in target_pids]) for uid, pfs in uid_to_pattern_fracs.items() if uid in pt_uids],
                                  [np.median([x['perf'] for x in user_meta_lookup[uid] if x['pid'] in target_pids]) for uid, pfs in uid_to_pattern_fracs.items() if uid in pt_uids],
                                  nan_policy="omit"))

    # coefs = {uid: np.corrcoef([[u[a + "_ubiq_all"] for u in us if u['pid'] in target_pids] for a in get_action_labels()]) for uid, us in ubiq_lookup.items() if uid in uids}
    coefs = np.corrcoef([[u[a + "_ubiq_all"] for uid in uids for u in ubiq_lookup[uid] if u['pid'] in target_pids] for a in get_action_labels()])
    fig, ax = plt.subplots(figsize=(10,10))
    # im = ax.matshow(np.nanmean(list(coefs.values()), axis=0))
    im = ax.matshow(coefs)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(get_action_labels())))
    ax.set_yticks(np.arange(len(get_action_labels())))
    ax.set_xticklabels(get_action_labels())
    ax.set_yticklabels(get_action_labels())
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    fig.savefig("variation_viz/all_coef_mat_ubiq_{}cohort.png".format(label))
    plt.close()

    # coefs = {uid: np.corrcoef([[pf[pt] for pid, pf in pfs.items() if pid in target_pids] for pt in target_pts]) for uid, pfs in uid_to_pattern_fracs.items() if uid in uids}
    coefs = np.corrcoef([[pf[pt] for uid in uids for pid, pf in uid_to_pattern_fracs[uid].items() if pid in target_pids] for pt in target_pts])
    fig, ax = plt.subplots(figsize=(10,10))
    # im = ax.matshow(np.nanmean(list(coefs.values()), axis=0), vmax=0.5)
    im = ax.matshow(coefs)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(target_pts)))
    ax.set_yticks(np.arange(len(target_pts)))
    ax.set_xticklabels(target_pts)
    ax.set_yticklabels(target_pts)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    fig.savefig("variation_viz/all_coef_mat_patterns_{}cohort.png".format(label))
    plt.close()
    print("\n\n")

    # shown = []
    # print()
    # print("correlations of two-action correlation with perf (only those with p < 0.05 shown")
    # for a in get_action_labels():
    #     series = []
    #     for b in get_action_labels():
    #         s = [stats.spearmanr([u[a + "_ubiq_all"] for u in us if u['pid'] in target_pids],
    #                              [u[b + "_ubiq_all"] for u in us if u['pid'] in target_pids]) for uid, us in ubiq_lookup.items() if uid in uids]
    #         series.append([r.correlation for r in s if not np.isnan(r.correlation)])
    #         if {a, b} not in shown:
    #             shown.append({a, b})
    #             cor = stats.spearmanr([x.correlation for x in s], [np.median([x['perf'] for x in user_meta_lookup[uid]]) for uid, us in ubiq_lookup.items() if uid in uids], nan_policy="omit")
    #             if cor.pvalue < 0.05:
    #                 print("cor({}, {}) {}".format(a, b, cor))
    #     make_boxplot(series, get_action_labels(), "spearman r", "variation_viz/{}_correlations_{}cohort.png".format(a, min_exp))
    # print()
    # print()

    # shown = []
    print()
    print("correlations of two-pattern correlation with perf (only those with p < 0.01 shown)")
    for a, b in combinations(target_pts, 2):
        # series = []
        # for b in target_pts:
        s = [stats.spearmanr([pf[a] for pid, pf in pfs.items() if pid in target_pids],
                             [pf[b] for pid, pf in pfs.items() if pid in target_pids]) for uid, pfs in uid_to_pattern_fracs.items() if uid in uids]
            # series.append([r.correlation for r in s if not np.isnan(r.correlation)])
            # if {a, b} not in shown:
            #     shown.append({a, b})
        cor = stats.spearmanr([x.correlation for x in s], [np.median([x['perf'] for x in user_meta_lookup[uid]]) for uid, pfs in uid_to_pattern_fracs.items() if uid in uids], nan_policy="omit")
        if cor.pvalue < 0.01:
            print("cor({}, {}) {}".format(a, b, cor))
        # make_boxplot(series, get_action_labels(), "spearman r", "variation_viz/{}_correlations_{}cohort.png".format(a, min_exp))
    print()
    print()

# devs = {}
# for uid, us in ubiq_lookup.items():
#     target_us = [u for u in us if u['pid'] in target_pids]
#     devs[uid] = {}
#     for i, u in enumerate(target_us[5:], 5):
#         pid = u['pid']
#         devs[uid][pid] = {}
#         for a in get_action_labels():
#             mean = np.mean([u[a + "_ubiq_all"] for u in target_us[:i] if a != "build" or u['pid'] >= '2002327'])
#             std = np.std([u[a + "_ubiq_all"] for u in target_us[:i] if a != "build" or u['pid'] >= '2002327'])
#             devs[uid][pid][a] = (u[a + "_ubiq_all"] - mean) / std
