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
from typing import Iterable, List, Optional, Tuple, Dict
from itertools import groupby
from functools import partial
import matplotlib.pyplot as plt

sys.path.append("../")  # enable importing from parent directory
from pattern_extraction import combine_user_series, run_TICC

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
                if unit.is_army or unit.is_worker or unit.is_building:
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


def plot_unit_series(series: np.ndarray, fields: List[str], filename: str = "") -> None:
    """

    :param series:
    :param fields:
    :param filename:
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = plt.cm.get_cmap("tab20").colors + plt.cm.get_cmap("tab20b").colors
    for j in range(series.shape[1]):
        ax.plot(np.arange(len(series)), series[:, j], color=colors[j], alpha=0.7)
    fig.tight_layout()
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


def make_combined_series(race, series_lookup):
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

    if args.lookup:
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
                with open(f"{rf.replace('.SC2Replay', '_meta.json')}")as fp:
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
        with open("results/race_lookup.pickle", "wb") as fp:
            pickle.dump(race_lookup, fp)

    all_series_lookup = {}

    with Pool() as pool:
        pool.starmap(make_combined_series, iter(race_lookup.items()))

    for race in race_lookup:
        all_series_lookup[race] = np.loadtxt(f"results/{race}/all_series.txt")

    print("running ticc")
    run_TICC(all_series_lookup, "results", [20], window_size=5, num_proc=8)
