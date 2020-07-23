import argparse
import logging
import os
import sys
import json
from itertools import groupby
from typing import Dict, Any

import numpy as np
import pickle
import pandas as pd
from types import SimpleNamespace
sys.path.append("../")

from pattern_viz import plot_labeled_series, plot_user_series
from util import time_played, get_action_labels, SubSeriesLookup, get_pattern_label
from .foldit_data import load_extend_data, make_series, get_deltas, make_action_series
from pattern_extraction import combine_user_series, run_TICC, load_TICC_output, \
    select_TICC_model, get_patterns, make_subseries_lookup, run_sub_TICC, \
    load_sub_lookup, get_pattern_lookups, find_best_dispersion_model, get_pattern_masks

if __name__ == "__main__":
    # pids = ['2003433', '2003465', '2003490', '2003195', '2003240', '2003206', '2003483',
    #         '2003583']
    # denovo: '2003433', '2003111', '2003465', '2003490', '2003195'
    # revisiting: '2003240', '2003206', '2003125', '2003483', '2003583'

    parser = argparse.ArgumentParser(prog='pattern_extraction.py')
    parser.add_argument("config")
    args = parser.parse_args()

    try:
        with open(args.config) as fp:
            config = SimpleNamespace(**json.load(fp))
    except FileNotFoundError:
        print(f"{args.config} could not be opened")

    if config.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    results_path = f"{config.results_dir}/{config.name}"
    if os.path.exists(results_path) and not config.overwrite:
        logging.error(f"{results_path} exists and no overwrite flag given")
        sys.exit(1)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    pids = config.pids

    krange = config.krange

    soln_lookup = {}
    parent_lookup = {}
    child_lookup = {}
    data, puz_metas = load_extend_data(pids, soln_lookup, parent_lookup, child_lookup, config.evolver, 600)

    logging.debug("Constructing time series")
    puz_idx_lookup, series_lookup, noise = make_series(data)
    # num_features = next(x for x in series_lookup.values()).shape[1]
    # filtered_series_lookup = {uid: ser for uid, ser in series_lookup.items() if
    #                           len(ser) < 200 * len(pids) or len(data[data.uid.isin([uid])].pid.unique()) < len(pids) // 2}
    idx_lookup, all_series = combine_user_series(series_lookup, noise)
    puz_idx_lookup = {(uid, pid): (s + idx_lookup[uid][0], e + idx_lookup[uid][0]) for (uid, pid), (s, e) in
                      puz_idx_lookup.items()}
    # noinspection PyTypeChecker
    np.savetxt(f"{results_path}/noise_values.txt", noise)
    # noinspection PyTypeChecker
    np.savetxt(f"{results_path}/all_series.txt", all_series)
    with open(f"{results_path}/puz_idx_lookup.pickle", 'wb') as fp:
        pickle.dump(puz_idx_lookup, fp)
    with open(f"{results_path}/idx_lookup.pickle", 'wb') as fp:
        pickle.dump(idx_lookup, fp)

    if config.evolver:
        evol_series_lookup = {}
        for _, r in data.iterrows():
            if r.evol_lines:
                for idx, line in enumerate(r.evol_lines):
                    if len(line.pdb_infos) <= 1 or line.pdb_infos[0].parent_sid not in soln_lookup:
                        continue
                    deltas = sorted(get_deltas(sorted(line.pdb_infos, key=lambda p: p.timestamp), soln_lookup),
                                    key=lambda x: x.timestamp)
                    if len(deltas) > 0 and time_played([d.timestamp for d in deltas]) > 600:
                        ser = make_action_series(deltas)
                        evol_series_lookup["{}_{}_{}".format(r.uid, r.pid, idx)] = ser
        evol_idx_lookup, evol_all_series = combine_user_series(evol_series_lookup, noise)
        # noinspection PyTypeChecker
        np.savetxt(f"{results_path}/evol_all_series.txt", evol_all_series)
        with open(f"{results_path}/evol_idx_lookup.pickle", 'wb') as fp:
            pickle.dump(evol_idx_lookup, fp)

    logging.debug("Running TICC")
    run_TICC({"all": all_series}, results_path, krange)
    if config.evolver:
        run_TICC({"all_evol": evol_all_series}, results_path, krange)

    logging.debug("Loading TICC output")
    cluster_lookup, mrf_lookup, model_lookup, bic_lookup = load_TICC_output(results_path, ["all"], krange)

    logging.debug("Making subseries")
    subseries_lookups: Dict[int, Dict[int, SubSeriesLookup]] = {}
    for k in krange:
        patterns = get_patterns(mrf_lookup["all"][k], cluster_lookup["all"][k], puz_idx_lookup)
        subseries_lookups[k] = make_subseries_lookup(k, patterns, mrf_lookup["all"][k], all_series, noise)

    logging.debug("Running recursive TICC")
    run_sub_TICC(subseries_lookups, results_path, "all", config.sub_krange)
    sub_lookup = load_sub_lookup(f"{results_path}/all", subseries_lookups, config.sub_krange)

    sub_clusters = sub_lookup["clusters"]
    pattern_lookup = get_pattern_lookups(krange, sub_clusters, sub_lookup["mrfs"], subseries_lookups,
                                         cluster_lookup["all"], mrf_lookup["all"], puz_idx_lookup)
    os.makedirs(f"{results_path}/eval", exist_ok=True)
    with open(f"{results_path}/eval/cluster_lookup.pickle", "wb") as fp:
        pickle.dump(cluster_lookup, fp)
    with open(f"{results_path}/eval/subseries_lookup.pickle", "wb") as fp:
        pickle.dump(subseries_lookups, fp)
    with open(f"{results_path}/eval/sub_clusters.pickle", "wb") as fp:
        pickle.dump(sub_clusters, fp)
    with open(f"{results_path}/eval/pattern_lookup.pickle", "wb") as fp:
        pickle.dump(pattern_lookup, fp)

    best_k, best_subs = find_best_dispersion_model(all_series, pattern_lookup, subseries_lookups, sub_clusters)
    with open(f"{results_path}/eval/best_model.txt", 'w') as fp:
        fp.write(str((best_k, best_subs)) + "\n")

    logging.debug("Plotting")
    all_clusters = cluster_lookup["all"][best_k]
    action_labels = get_action_labels()

    ps = sum([[(get_pattern_label(p, cid, sub_k), p) for p in pattern_lookup[best_k][cid][sub_k]]
              for cid, sub_k in best_subs], [])
    ps_uid_pid = {tag: sorted(xs) for tag, xs in
                  groupby(sorted(ps, key=lambda p: (p[1].uid, p[1].pid)), lambda p: (p[1].uid, p[1].pid))}
    pattern_use_lookup = {tag: {pt for pt, _ in xs} for tag, xs in ps_uid_pid.items()}
    pts = {pt for pt, p in ps}

    for uid in series_lookup:
        plot_user_series(results_path, best_k, best_subs, puz_idx_lookup, all_series, pattern_lookup, pts,
                         subseries_lookups, action_labels)
        ser = series_lookup[uid]
        cs = all_clusters[slice(*idx_lookup[uid])]
        plot_labeled_series(np.arange(len(ser)), ser, {}, action_labels,
                            f"{results_path}/all/{uid}_series_all_k{best_k}.png")

    patterns = get_patterns(mrf_lookup["all"][best_k], all_clusters, puz_idx_lookup)

    cluster_times = []
    for uid, ser in series_lookup.items():
        # print(uid)
        sample = data.groupby('uid').get_group(uid).sort_values('pid')
        for _, r in sample.iterrows():
            if r.relevant_sids is None or len(r.relevant_sids) == 0:
                continue
            # print("   ", r.pid)
            deltas = sorted([d for d in r.deltas if d.sid in r.relevant_sids], key=lambda x: x.timestamp)
            ts = np.array([d.timestamp for d in deltas])
            puz_cs = all_clusters[slice(*puz_idx_lookup[(uid, r.pid)])]
            assert len(ts) == len(puz_cs)
            result_dict = {'uid': uid, 'pid': r.pid}
            for ci in range(k):
                result_dict[f"cluster_{ci}_time"] = time_played(ts[puz_cs == ci])
            cluster_times.append(result_dict)

    results = data.merge(pd.DataFrame(data=cluster_times), on=['pid', 'uid'])

    """
    all_timestamps = np.zeros(len(all_series))
    for (uid, pid), (s, e) in puz_idx_lookup.items():
        # print(uid, pid)
        sids = get_data_value(uid, pid, "relevant_sids", data)
        ts = sorted([d.timestamp for d in get_data_value(uid, pid, "deltas", data) if d.sid in sids])
        assert len(ts) == e - s
        all_timestamps[s:e] = ts


    def get_pattern_improve(p):
        return min(0, p.post_best.energy - (
            p.pre_best.energy if p.pre_best and p.pre_best.energy < puz_metas[p.pid].energy_baseline else puz_metas[
                p.pid].energy_baseline)) / (puz_metas[p.pid].pfront.min() - puz_metas[p.pid].energy_baseline)


    sequences = {(uid, pid): sorted(g, key=lambda p: p.start_idx) for (uid, pid), g in
                 groupby(sorted(patterns, key=lambda p: (p.uid, p.pid)), lambda p: (p.uid, p.pid))}


    def make_ticc_from_saved_model(saved_model):
        test_ticc = TICC(window_size=saved_model["window_size"], number_of_clusters=saved_model["number_of_clusters"], num_proc=1)
        test_ticc.num_blocks = saved_model["num_blocks"]
        test_ticc.switch_penalty = saved_model["switch_penalty"]
        test_ticc.trained_model = saved_model["trained_model"]
        return test_ticc


    def get_labels(deltas, soln_lookup, radius=5):
        labels = ['r']
        for i, x in enumerate(deltas[1:]):
            window = [a.parent_sid for a in deltas[max(i - radius, 0): i + radius + 1]]
            starts = [window[0]] + [p for j, p in enumerate(window[1:]) if window[j] != p]
            if len(set(starts)) < len(starts): # multiple, non-consecutive instances of at least one parent in the window indicates growing multiple parts of the tree
                labels.append('m')
            elif x.parent_sid == deltas[i].sid:
                labels.append('d')
            elif x.parent_sid == deltas[i].parent_sid:
                labels.append('b')
            else:
                if labels[i] == 'b' and soln_lookup[x.parent_sid].parent_sid == deltas[i].parent_sid:
                    labels.append('d')
                else:
                    labels.append('o')
        return labels


    sequences = {(uid, pid): sorted(g, key=lambda p: p.start_idx) for (uid, pid), g in
             groupby(sorted(patterns, key=lambda p: (p.uid, p.pid)), lambda p: (p.uid, p.pid))}
    grams = []
    for seq in sequences.values():
        for l in range(2, len(seq)):
            for i in range(len(seq) - l + 1):
                grams.append(seq[i:i + l])
    gram_map = {gram: list(seqs) for gram, seqs in groupby(sorted(grams, key=lambda g: "".join([str(p.cid) for p in g])), lambda g: "".join([str(p.cid) for p in g]))}
    {k: np.mean([sum(get_pattern_improve(p) for p in seq) / time_played(all_timestamps[seq[0].start_idx:seq[-1].end_idx]) for seq in seqs if seq[0].start_idx != puz_idx_lookup[(seq[0].uid, seq[0].pid)][0]]) for k, seqs in gram_map.items() if len(k) == 4} # k is length of the sequence, != for non-start sequences

    subseries_lookup = {}
    for k in krange:
        logging.debug("getting patterns for k={}".format(k))
        patterns = get_patterns(data, mrf_lookup["all"][k], cluster_lookup["all"][k], puz_idx_lookup, soln_lookup)
        if not os.path.exists(results_path + "/subpatterns_k{}".format(k)):
            os.makedirs(results_path + "/subpatterns_k{}".format(k))
        subseries_lookup.setdefault(k, {})
        for cid in range(k):
            if is_null_cluster(mrf_lookup["all"][k][cid]) or cid in subseries_lookup[k]:
                continue
            logging.debug("assembling series for pattern {}".format(cid))
            sslu = {}
            sslu["patterns"] = {(p.uid, p.pid, p.start_idx): all_series[p.start_idx:p.end_idx] for p in patterns if p.cid == cid}
            sub_idx_lookup, all_subseries = combine_user_series(sslu['patterns'], noise, default_noise_duration=100)
            if len(all_subseries) - 100 * (len(sslu["patterns"]) - 1) < 20:
                logging.debug("skipping pattern {} because it only appears for {} timesteps".format(cid, len(all_subseries) - 100 * len(sslu["patterns"])))
                continue
            sslu["series"] = all_subseries
            sslu["idx_lookup"] = sub_idx_lookup
            subseries_lookup[k][cid] = sslu
            os.makedirs(results_path + "/subpatterns_k{}/subpatterns{}".format(k, cid))
            with open(results_path + "/subpatterns_k{}/subpatterns{}/sslu.pickle".format(k, cid), 'wb') as fp:
                pickle.dump(sslu, fp)
            run_TICC({"subpatterns{}".format(cid): all_subseries}, results_path + "/subpatterns_k{}".format(k), [3, 6, 9, 12])


    sub_cluster_lookup, sub_mrf_lookup, sub_model_lookup, sub_bic_lookup = load_TICC_output(results_path, ["subpatterns_1", "subpatterns_2", "subpatterns_3", "subpatterns_4"], krange)

    subpattern_lookup = {}
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for p in patterns:
        sp = subpattern_lookup.setdefault(p.uid, {})
        cs = sub_cluster_lookup['subpatterns_{}'.format(p.cid)][10][slice(*subseries_lookup[p.cid]['idx_lookup'][(p.uid, p.pid, p.start_idx)])]
        change_idx = np.array([0] + [i for i in range(1, len(cs)) if cs[i] != cs[i - 1]])
        subs = []
        offset = p.start_idx - idx_lookup[p.uid][0]
        for start in change_idx:
            end = next(i for i, c in enumerate(cs[start:], start) if c != cs[start] or i + 1 == len(cs))
            subs.append((offset + start, offset + end, p, labels[cs[start]]))
        sp[p.start_idx - idx_lookup[p.uid][0]] = subs

    for cid in range(k):
        if is_null_cluster(mrf_lookup["all"][k][cid]):
            print("null cluster", cid)
            continue
        print("cluster", cid)
        print("    used by {:.2%} of users".format(len([uid for uid, rows in results.groupby('uid') if any(r["cluster_{}_time".format(cid)] > 0 for _, r in rows.iterrows())]) / len(results.uid.unique())))
        users = [rows for uid, rows in results.groupby('uid') if any(r["cluster_{}_time".format(cid)] > 0 for _, r in rows.iterrows())]
        if len(users) > 0:
            print("    of its users, {:.2%} use it on every puzzle".format(len([rows for rows in users if all(r["cluster_{}_time".format(cid)] > 0 for _, r in rows.iterrows())]) / len(
            users)))
            ps = [p for p in patterns if p.cid == cid and p.start_idx > 0]
            print("    {} instances in the dataset".format(len(ps)))
            print("    {:.4f} mean duration".format(np.mean([time_played(all_timestamps[p.start_idx:p.end_idx]) for p in ps])))
            print("    {:.4f} median improve".format(np.median([get_pattern_improve(p) for p in ps])))
            print("    {:.4e} median improve rate".format(np.nanmedian([get_pattern_improve(p) / (time_played(all_timestamps[p.start_idx:p.end_idx]) if p.start_idx + 1 < p.end_idx else 300) for p in ps])))
            for prev_cid, g in groupby(sorted(ps, key=lambda p: all_clusters[p.start_idx - 1]), lambda p: all_clusters[p.start_idx - 1]):
                xs = list(g)
                print("    mean improve of {:.4f} when preceeded by {} (n={})".format(np.mean([get_pattern_improve(x) for x in xs]), prev_cid, len(xs)))
            print("subpatterns")
            for sub_cid in sub_mrf_lookup["subpatterns_{}".format(cid)][10]:
                if is_null_cluster(sub_mrf_lookup["subpatterns_{}".format(cid)][10][sub_cid]):
                    print("    null subpattern", sub_cid)
                    continue
                print("    subpattern", labels[sub_cid])
                sub_users = {(p.uid, p.pid) for d in subpattern_lookup.values() for sub_ps in d.values() for si, ei, p, label in sub_ps if p.cid == cid and label == labels[sub_cid]}
                print(max(sub_users, key=lambda x: get_data_value(x[0], x[1], "perf", results)))
                non_sub_users = {(r.uid, r.pid) for rows in users for _, r in rows.iterrows() if (r.uid, r.pid) not in sub_users}
                sig_test([get_data_value(uid, pid, "perf", results) for uid, pid in sub_users], [get_data_value(uid, pid, "perf", results) for uid, pid in non_sub_users])
        print()
        sig_test([rows.perf.median() for uid, rows in results.groupby('uid') if any(r["cluster_{}_time".format(cid)] > 0 for _, r in rows.iterrows())], [rows.perf.median() for uid, rows in results.groupby('uid') if all(r["cluster_{}_time".format(cid)] == 0 for _, r in rows.iterrows())])


    def predict_from_saved_model(test_data, saved_model):
        test_ticc = TICC(window_size=saved_model["window_size"], number_of_clusters=saved_model["number_of_clusters"], num_proc=1)
        test_ticc.num_blocks = saved_model["num_blocks"]
        test_ticc.switch_penalty = saved_model["switch_penalty"]
        test_ticc.trained_model = saved_model["trained_model"]
        cs = test_ticc.predict_clusters(test_data)
        test_ticc.pool.close()
        return cs

    subpatterns = [SubPatternInstance(p, label, si + idx_lookup[uid][0], ei + idx_lookup[uid][0]) for uid, d in subpattern_lookup.items() for sub_ps in d.values() for si, ei, p, label in sub_ps]
    subsequences = {(uid, pid): sorted(g, key=lambda sp: sp.start_idx) for (uid, pid), g in groupby(sorted(subpatterns, key=lambda sp: (sp.p.uid, sp.p.pid)), lambda sp: (sp.p.uid, sp.p.pid))}

    all_subclusters = all_clusters.astype(np.str)
for p in patterns:
    all_subclusters[p.start_idx:p.end_idx] = ["{}{}".format(p.cid, labels[x]) for x in sub_cluster_lookup["subpatterns_{}".format(p.cid)][10][slice(*subseries_lookup[p.cid]['idx_lookup'][(p.uid, p.pid, p.start_idx)])]]

subcluster_times = []
for _, r in data.iterrows():
    if r.relevant_sids is None or len(r.relevant_sids) == 0:
        continue
    deltas = sorted([d for d in r.deltas if d.sid in r.relevant_sids], key=lambda x: x.timestamp)
    ts = np.array([d.timestamp for d in deltas])
    puz_cs = all_subclusters[slice(*puz_idx_lookup[(r.uid, r.pid)])]
    assert len(ts) == len(puz_cs)
    result_dict = {'uid': r.uid, 'pid': r.pid}
    for ci in np.unique(all_subclusters):
        result_dict["subcluster_{}_time".format(ci)] = time_played(ts[puz_cs == ci])
    subcluster_times.append(result_dict)

results = data.merge(pd.DataFrame(data=subcluster_times), on=['pid', 'uid'])
"""
