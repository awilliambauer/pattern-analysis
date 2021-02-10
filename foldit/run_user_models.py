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
from foldit_data import load_extend_data, make_series, get_deltas, make_action_series
from pattern_extraction import combine_user_series, run_TICC, load_TICC_output, \
    select_TICC_model, get_patterns, make_subseries_lookup, run_sub_TICC, \
    load_sub_lookup, get_pattern_lookups, find_best_dispersion_model, get_pattern_masks, \
    handles_noise, is_null_cluster

if __name__ == "__main__":


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
    os.makedirs(results_path, exist_ok=True)

    pids = config.pids
    krange = config.krange

    soln_lookup = {}
    parent_lookup = {}
    child_lookup = {}
    data, puz_metas = load_extend_data(pids, soln_lookup, parent_lookup, child_lookup, config.evolver, 600)

    if os.path.exists(f"{results_path}/noise_values.txt") and not config.overwrite:
        logging.debug(f"{results_path}/noise_values.txt already exists and overwrite not set, attempting to load existing series")
        noise = np.loadtxt(f"{results_path}/noise_values.txt")
        with open(f"{results_path}/series_lookup.pickle", 'rb') as fp:
            series_lookup = pickle.load(fp)
        with open(f"{results_path}/puz_idx_lookup.pickle", 'rb') as fp:
            puz_idx_lookup = pickle.load(fp)
    else:
        logging.debug("Constructing time series")
        puz_idx_lookup, series_lookup, noise = make_series(data, min_puzzles=3)
        # noinspection PyTypeChecker
        np.savetxt(f"{results_path}/noise_values.txt", noise)
        with open(f"{results_path}/series_lookup.pickle", 'wb') as fp:
            pickle.dump(series_lookup, fp)
        with open(f"{results_path}/puz_idx_lookup.pickle", 'wb') as fp:
            pickle.dump(puz_idx_lookup, fp)

    uids_to_run = []
    for uid in series_lookup:
        if all(os.path.exists(f"{results_path}/{uid}/clusters_k{k}.txt") for k in krange) and not config.overwrite:
            logging.warning(f"results found at {results_path}/{uid} and overwrite not set, skipping first-pass TICC")
        else:
            uids_to_run.append(uid)

    logging.debug(f"Running TICC for {uids_to_run}")
    run_TICC({uid: series_lookup[uid] for uid in uids_to_run}, results_path, krange)

    logging.debug("Loading TICC output")
    cluster_lookup, mrf_lookup, model_lookup, bic_lookup = load_TICC_output(results_path, list(series_lookup.keys()),
                                                                            krange)

    logging.debug("Making subseries")
    subseries_lookups: Dict[str, Dict[int, Dict[int, SubSeriesLookup]]] = {}
    for uid, series in series_lookup.items():
        subseries_lookups[uid] = {}
        for k in krange:
            null_clusters = [cid for cid in mrf_lookup[uid][k] if is_null_cluster(mrf_lookup[uid][k][cid])]
            if handles_noise(series, null_clusters, cluster_lookup[uid][k], noise):
                # get_patterns assumes the ranges in puz_idx_lookup correspond to the indexes of cluster_lookup[uid][k]
                # in this context, that's only true for entries for the current uid, so we filter
                patterns = get_patterns(mrf_lookup[uid][k], cluster_lookup[uid][k],
                                        {(u, pid): v for (u, pid), v in puz_idx_lookup.items() if u == uid})
                subseries_lookups[uid][k] = make_subseries_lookup(k, patterns, mrf_lookup[uid][k], series, noise)
            else:
                logging.debug(f"k = {k} model fails to handle noise for uid = {uid}")

    for uid in series_lookup:
        if all(os.path.exists(f"{results_path}/{uid}/subpatterns/k{k}") for k in config.sub_krange) and not config.overwrite:
            logging.warning(f"results found at {results_path}/{uid}/subpatterns and overwrite not set, skipping recursive TICC")
        else:
            logging.debug(f"Running recursive TICC for {uid}")
            run_sub_TICC(subseries_lookups[uid], results_path, uid, config.sub_krange)

    logging.debug("Loading recursive TICC output")
    sub_lookup = load_sub_lookup(results_path, list(series_lookup.keys()), subseries_lookups, config.sub_krange)
