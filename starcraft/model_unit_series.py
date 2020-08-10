"""
Applying pattern extraction to series of current unit and building counts
"""
from multiprocessing import Pool
import traceback
import argparse
import numpy as np
import sc2reader
import os
import sys
import pickle
import logging
from typing import Iterable, List, Optional, Tuple, Dict, Hashable
from itertools import groupby
from functools import partial
import matplotlib.pyplot as plt

sys.path.append("../")  # enable importing from parent directory
from pattern_extraction import *

zerg_fields = ['baneling', 'banelingnest', 'broodlord', 'corruptor', 'drone', 'evolutionchamber', 'extractor',
               'greaterspire', 'hatchery', 'hive', 'hydralisk', 'hydraliskden', 'infestationpit', 'infestedterran',
               'infestor', 'lair', 'locust', 'lurker', 'lurkerden', 'mutalisk', 'nydusnetwork', 'nydusworm', 'overlord',
               'overseer', 'queen', 'ravager', 'roach', 'roachwarren', 'spawningpool', 'spinecrawler', 'spire',
               'sporecrawler', 'swarmhost', 'ultralisk', 'ultraliskcavern', 'viper', 'zergling']
protoss_fields = ['adept', 'archon', 'assimilator', 'carrier', 'colossus', 'cyberneticscore', 'darkshrine',
                  'darktemplar', 'disruptor', 'fleetbeacon', 'forge', 'gateway', 'hightemplar', 'immortal',
                  'interceptor', 'mothership', 'mothershipcore', 'nexus', 'observer', 'oracle', 'phoenix',
                  'photoncannon', 'probe', 'pylon', 'reactor', 'roboticsbay', 'roboticsfacility', 'sentry', 'stalker',
                  'stargate', 'tempest', 'templararchive', 'twilightcouncil', 'voidray', 'warpgate', 'warpprism',
                  'zealot']
terran_fields = ['armory', 'banshee', 'barracks', 'barrackstechlab', 'barracksreactor', 'battlecruiser',
                 'battlehellion', 'bunker', 'commandcenter', 'cyclone', 'engineeringbay', 'factory', 'factoryreactor',
                 'factorytechlab', 'fusioncore', 'ghost', 'ghostacademy', 'hellion', 'marauder', 'marine', 'medivac',
                 'missileturret', 'mule', 'orbitalcommand', 'planetaryfortress', 'raven', 'reactor', 'reaper', 'refinery', 'scv',
                 'sensortower', 'siegetank', 'starport', 'starportreactor', 'starporttechlab', 'supplydepot', 'techlab',
                 'thor', 'viking', 'warhound', 'widowmine']
aliases = {"supplydepotlowered": "supplydepot", "siegetanksieged": "siegetank", "widowmineburrowed": "widowmine"}
field_lookup = {"Protoss": protoss_fields, "Zerg": zerg_fields, "Terran": terran_fields}


def make_unit_series(replay: sc2reader.resources.Replay, pindex: int, binsize: int) -> np.ndarray:
    """

    :param replay:
    :param pindex:
    :return:
    """
    frame_to_events = {frame_bin: list(events) for frame_bin, events in
                       groupby(sorted(replay.events, key=lambda e: e.frame // binsize), lambda e: e.frame // binsize)}


    player = replay.players[pindex]
    fields = field_lookup[player.play_race]
    series = np.zeros((replay.frames // binsize, len(fields)))

    for frame_bin in range(len(series)):
        if frame_bin > 0:
            series[frame_bin] = series[frame_bin - 1]
        for e in frame_to_events.get(frame_bin, []):
            if not hasattr(e, "unit") or e.unit.owner != player:
                continue
            if isinstance(e, sc2reader.events.UnitBornEvent) or isinstance(e, sc2reader.events.UnitDoneEvent):
                unit = e.unit
                if (unit.is_army or unit.is_worker or unit.is_building) and not unit.hallucinated:
                    name = sorted(unit.type_history.items())[0][1].name.lower()
                    if name in aliases:
                        name = aliases[name]
                    assert name in fields, f"{player.play_race} {name} {unit.is_army} {unit.is_worker} {unit.is_building}, {unit._type_class.name}"
                    series[frame_bin][fields.index(name)] += 1
            if isinstance(e, sc2reader.events.UnitDiedEvent):
                unit = e.unit
                if unit.is_army or unit.is_worker or unit.is_building:
                    name = sorted(unit.type_history.items())[0][1].name.lower()
                    if name in aliases:
                        name = aliases[name]
                    assert name in fields
                    if unit.finished_at:  # ignore the death of an incomplete building
                        series[frame_bin][fields.index(name)] -= 1
    return series


def plot_unit_series(series: np.ndarray, fields: List[str], pattern_masks: Dict[str, np.ndarray], filename: str = "", figsize: Tuple[int, int] = (20, 10)) -> None:
    """

    :param series:
    :param pattern_masks:
    :param fields:
    :param filename:
    :param figsize:
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.get_cmap("tab20").colors + plt.cm.get_cmap("tab20b").colors
    x = np.arange(len(series))
    for j in range(series.shape[1]):
        ax.plot(x, series[:, j], color=colors[j], alpha=0.7)
    ax.set_xlim(-10, len(series) + 10)
    fig.tight_layout()
    plt.draw()

    if pattern_masks:
        mask_colors = plt.cm.get_cmap("viridis", len(pattern_masks)).colors
        fills = []
        pattern_labels = []
        for i, (pt, mask) in enumerate(pattern_masks.items()):
            if mask.any():
                fills.append(ax.fill_between(x, 0, series.max(), mask, color=mask_colors[i], alpha=0.4))
                pattern_labels.append(pt)
                if mask[0]:
                    ax.text(-2, series.max(), pt)
                for i, (a, b) in enumerate(zip(mask, mask[1:])):
                    if b and not a:
                        ax.text(i, series.max(), pt)
        if len(pattern_masks) > 0:
            ax.legend(fills, pattern_labels, bbox_to_anchor=(1.01, 1), fancybox=True, shadow=True)
        plt.draw()

    cols = {j for j in np.nonzero(series)[1]}
    for j in cols:
        field = fields[j]
        if os.path.exists(f"sprites/{field}.png"):
            im = plt.imread(f"sprites/{field}.png")
            im[np.where(np.all(im == np.array([0, 0, 0, 0]), axis=2))] = np.array(list(colors[j]) + [0.7])
            peak = series[:, j].argmax()
            # print(field, peak, series[peak, fields.index(field)])
            xo, yo = ax.transData.transform((peak, series[peak, fields.index(field)]))
            fig.figimage(im, xo=xo - im.shape[0] // 2, yo=yo - im.shape[1] // 2)
            plt.draw()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plot_model():
    # TODO prototype code from ipython
    k = 20
    for race in field_lookup:
        all_series = np.loadtxt(f"{results_path}/{race}/all_series.txt")
        with open(f"{results_path}/{race}/idx_lookup.pickle", 'rb') as fp:
            idx_lookup = pickle.load(fp)
        noise = np.array([-1] * len(field_lookup[race]))
        subseries_lookups = {}
        patterns = get_patterns(mrf_lookup[race][k], cluster_lookup[race][k], idx_lookup)
        subseries_lookups[k] = make_subseries_lookup(k, patterns, mrf_lookup[race][k], all_series, noise)
        pattern_lookups = get_pattern_lookups(krange, {20: {}}, {}, subseries_lookups,
                                             cluster_lookup[race], mrf_lookup[race], idx_lookup)
        for x, ((uid, gid, _), idx) in enumerate(idx_lookup.items()):
            print(f"plotting {x} out of {len(idx_lookup)} {race} series\r", end="")
            masks = get_pattern_masks(uid, gid, idx, [str(i) for i in pattern_lookups[20].keys() if i != "base"], {}, pattern_lookups[20], {})
            plot_unit_series(all_series[idx[0]: idx[1]], field_lookup[race], masks, f"{results_path}/{race}/{uid}_{gid}.png")
        plot_model(f"{results_path}/{race}", 20, [(i, 0) for i in pattern_lookups.keys() if i != "base"], all_series, pattern_lookup, field_lookup[race])


def replay_to_series(replay_file: str, binsize: int) -> Optional[Dict[Tuple[str, str], np.ndarray]]:
    """

    :param replay_file:
    :return:
    """
    try:
        replay = sc2reader.load_replay(replay_file)
    except Exception as e:
        print(f"error loading replay {replay_file}: {e}")
        return
    try:
        if len(replay.players) != 2:
            return
        game_id = replay_file.split("_")[1].split(".")[0]
        if replay_file.startswith("ggg"):
            game_id = "ggg-" + game_id
        elif replay_file.startswith("spawningtool"):
            game_id = "st-" + game_id
        return {(replay.players[i].detail_data['bnet']['uid'], game_id, replay.players[i].play_race):
                    make_unit_series(replay, i, binsize) for i in range(len(replay.players))}
    except Exception as e:
        print(f"error making series for {replay_file}: {e}")
        traceback.print_exc()
        return


def make_combined_series(race: str, series_lookup: Dict[Hashable, np.ndarray]) -> None:
    results_path = f"results/{race}"
    os.makedirs(results_path, exist_ok=True)

    print("combining series for", race)
    idx_lookup, all_series = combine_user_series(series_lookup, np.array([-1] * len(field_lookup[race])), 100)

    np.savetxt(f"{results_path}/all_series.txt", all_series)
    with open(f"{results_path}/idx_lookup.pickle", 'wb') as fp:
        pickle.dump(idx_lookup, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("model_unit_series.py")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--lookup")
    group.add_argument("--sample", nargs=1, type=int)
    args = parser.parse_args()

    results_path = "results/small"
    os.makedirs(results_path, exist_ok=True)

    if args.lookup:
        print(f"loading {args.lookup}")
        with open(args.lookup, "rb") as fp:
            race_lookup = pickle.load(fp)
    else:
        with open("valid_game_ids.txt") as fp:
            replay_files = [x.strip() for x in fp.readlines() if x.startswith("spawningtool")]
        if args.sample:
            matchups = []
            metas = {}
            rng = np.random.default_rng(42)
            for rf in replay_files:
                with open(f"replays/{rf.replace('.SC2Replay', '_meta.json')}")as fp:
                    meta = json.load(fp)
                    metas[rf.split("_")[1].split(".")[0]] = meta
                    matchups.extend(meta['tags'].get("Matchup", ["Missing"]))

            matchup_frac = {m: c / sum(matchups_count.values()) for m, c in matchups_count.items()}
            weights = np.array([matchup_frac[m["tags"]["Matchup"][0]] for m in metas.values()])
            gids = rng.choice(list(metas.keys()), args.sample, False, weights / weights.sum())
            replay_files = [f"spawningtool_{x}.SC2Replay" for x in gids]

        series = []
        with Pool() as pool:
            for i, s in enumerate(pool.imap_unordered(partial(replay_to_series, binsize=160), [f"replays/{x}" for x in replay_files])):
                print(f"done {i / len(replay_files):%}\r", end="")
                if s:
                    series.append(s)

        # sort series k-v pairs by race
        series = sorted([(k, v) for d in series for k, v in d.items()], key=lambda x: x[0][2])
        race_lookup = {race: dict(vs) for race, vs in groupby(series, lambda x: x[0][2])}
        with open(f"{results_path}/race_lookup.pickle", "wb") as fp:
            pickle.dump(race_lookup, fp)

    all_series_lookup = {}

    if not all(os.path.exists(f"{results_path}/{race}/all_series.txt") for race in race_lookup):
        with Pool() as pool:
            pool.starmap(make_combined_series, iter(race_lookup.items()))

    for race in race_lookup:
        print(f"loading {race} series")
        all_series_lookup[race] = np.loadtxt(f"{results_path}/{race}/all_series.txt")


    logging.getLogger().setLevel(logging.DEBUG)
    krange = range(3, 21)
    sub_krange = range(2, 11)
    try:
        print("attempting to laod existing ticc models... ", end="")
        cluster_lookup, mrf_lookup, model_lookup, bic_lookup = load_TICC_output(results_path, ["Zerg", "Terran", "Protoss"], krange)
        print("done")
    except:
        print("failed\n running ticc")
        run_TICC(all_series_lookup, results_path, krange, num_proc=8)
        cluster_lookup, mrf_lookup, model_lookup, bic_lookup = load_TICC_output(results_path, ["Zerg", "Terran", "Protoss"], krange)

    for race, all_series in all_series_lookup.items():
        results_path_race = f"{results_path}/{race}"
        with open(f"{results_path_race}/idx_lookup.pickle", 'rb') as fp:
            idx_lookup = pickle.load(fp)
        noise = np.array([-1] * len(field_lookup[race]))

        subseries_lookups = {}
        cd = cluster_lookup[race]
        md = mrf_lookup[race]
        ser = all_series_lookup[race]
        no_dups = {k for k in cd if
                   not any(any((m == md[k][ci]).all() for cj, m in md[k].items() if cj > ci) for ci in range(k))}
        print(f"{race} k with no dups: {no_dups}")
        null_clusters = {k: [ci for ci in md[k] if is_null_cluster(md[k][ci])] for k in md}
        handles_noise = {k for k in cd if len(null_clusters[k]) > 0 and all(
            cd[k][i] in null_clusters[k] for i in range(len(ser)) if (ser[i] == noise).all())}
        print(f"{race} k that handle noise: {handles_noise}")
        valid_k = no_dups.intersection(handles_noise)

        for k in valid_k:
            patterns = get_patterns(mrf_lookup[race][k], cluster_lookup[race][k], idx_lookup)
            subseries_lookups[k] = make_subseries_lookup(k, patterns, mrf_lookup[race][k], all_series, noise)

        try:
            print(f"Attempting to laod existing recursive ticc models for {race}... ", end="")
            sub_lookup = load_sub_lookup(results_path_race, subseries_lookups, sub_krange)
        except:
            print("failed\n Running recursive ticc")
            run_sub_TICC(subseries_lookups, results_path, race, sub_krange)
            sub_lookup = load_sub_lookup(results_path_race, subseries_lookups, sub_krange)

        print("saving final model output for", race)
        sub_clusters = sub_lookup.clusters
        pattern_lookup = get_pattern_lookups(valid_k, sub_clusters, sub_lookup.mrfs, subseries_lookups,
                                             cluster_lookup[race], mrf_lookup[race], idx_lookup)

        os.makedirs(f"{results_path_race}/eval", exist_ok=True)
        with open(f"{results_path_race}/eval/cluster_lookup.pickle", "wb") as fp:
            pickle.dump(cluster_lookup, fp)
        with open(f"{results_path_race}/eval/subseries_lookups.pickle", "wb") as fp:
            pickle.dump(subseries_lookups, fp)
        with open(f"{results_path_race}/eval/sub_clusters.pickle", "wb") as fp:
            pickle.dump(sub_clusters, fp)
        with open(f"{results_path_race}/eval/pattern_lookup.pickle", "wb") as fp:
            pickle.dump(pattern_lookup, fp)
