# Intended to be used for data visualization of when players scout
# during StarCraft 2

from __future__ import print_function
import csv
import sc2reader
import time
import file_locations
from functools import partial
from multiprocessing import Pool, cpu_count
from collections import Counter
from itertools import repeat
import scouting_stats
import unit_prediction
import scouting_detector
import file_locations
from load_map_path_data import load_path_data
from sc2reader.engine.plugins import SelectionTracker, APMTracker
from selection_plugin import ActiveSelection
from battle_detector import buildBattleList, remove_scouting_during_battle, \
    remove_scouting_during_battles_and_harassment
from base_plugins import BaseTracker
from generate_replay_info import group_replays_by_map
import numpy as np
import traceback
from modified_rank_plugin import ModifiedRank
from data_analysis_helper import run, save
from collections import namedtuple

scouting_instance_fields = namedtuple("scouting_instance_fields",
                                      ("GameId", "UID1", "UID2", "Rank1", "Rank2", "Race1", "Race2",
                                       "ScoutingStartTime", "ScoutingEndTime"))

try:
    from reprlib import repr
except ImportError:
    pass


def generateFields(filename, map_path_data):
    '''generateFields takes in a filename of a replay, loads it and gathers necessary
    statistics, and returns the statistics in a tuple. It is to be used to write
    these stats to a csv. It also takes in an integer (1 or 2), which indicates
    which statistics will be gathered. In this case, generateFields gathers
    each point in the game where a period of scouting is initiated. Inputting a
    1 will return times as a fraction of the total game time, whereas inputting
    a 2 will return absolute frames.'''
    # loading the replay
    try:
        t = time.time()
        # extracting the game id and adding the correct tag
        # pathname = "practice_replays/" + filename
        pathname = file_locations.REPLAY_FILE_DIRECTORY + "/" + filename
        game_id = filename.split("_")[1].split(".")[0]
        if filename.startswith("ggg"):
            game_id = "ggg-" + game_id
        elif filename.startswith("spawningtool"):
            game_id = "st-" + game_id
        # loading the replay
        try:
            r = sc2reader.load_replay(pathname)
            if any(v != (0, {}) for v in r.plugin_result.values()):
                print(pathname, r.plugin_result)
        except:
            print(filename, "cannot load using sc2reader due to an internal ValueError")
            raise
        scouting_instances = scouting_detector.get_scouting_instances(r, map_path_data)
        team1_times, team2_times = remove_scouting_during_battles_and_harassment(r, scouting_instances)
        team1_rank, team1_rel_rank, team2_rank, team2_rel_rank = scouting_stats.ranking_stats(r)
        team1_uid = r.players[0].detail_data['bnet']['uid']
        team2_uid = r.players[1].detail_data['bnet']['uid']
        team1_race = r.players[0].play_race
        team2_race = r.players[1].play_race
        results = []
        for instance in team1_times:
            results.append(
                scouting_instance_fields(game_id, team1_uid, team2_uid, team1_rank, team2_rank, team1_race, team2_race,
                                         instance.start_time, instance.end_time))
        for instance in team2_times:
            results.append(
                scouting_instance_fields(game_id, team2_uid, team1_uid, team2_rank, team1_rank, team2_race, team1_race,
                                         instance.start_time, instance.end_time))

        return results

    except KeyboardInterrupt:
        raise
    except:
        traceback.print_exc()
        return


def writeToCsv(which, filename):
    '''writeToCsv gathers information about all valid replays and writes
    that information to a csv for analysis in R. This file in particular
    contains information about when players initiate periods of scouting.

    It takes an integer (1 or 2), which indicates which statistics will be
    gathered. Inputting a 1 will gather times as a fraction of the total game
    time, whereas inputting a 2 will gather absolute frames.

    It also takes in the name of the csv file that the information will be
    written to.'''
    # valid_game_ids.txt must be produced first by running scouting_stats.py
    # with the command line argument -w
    results = []
    with Pool(min(cpu_count(), 60)) as pool:
        count = 0
        for map_name_group, replays in group_replays_by_map().items():
            map_path_data = load_path_data(map_name_group)
            if map_path_data is None:
                print("no path data for map", map_name_group)
                continue
            print("loaded path data for map", map_name_group, "with", len(replays), "replays")
            count += len(replays)
            map_time = time.time()
            new_results = pool.map(partial(generateFields, which=which, map_path_data=map_path_data), replays)
            print("analyzing", len(replays), "replays for map", map_name_group, "took", time.time() - map_time)
            for result in new_results:
                results.append(result)
    with open(filename, 'w', newline='') as my_csv:
        events_out = csv.DictWriter(my_csv,
                                    fieldnames=["GameID", "UID", "Rank", "Race", "ScoutStartTime", "ScoutEndTime",
                                                "ScoutType"])
        events_out.writeheader()
        for fields in results:
            if fields:
                game_id = fields[0]
                uid = fields[1]
                rank = fields[2]
                times = fields[3]
                race = fields[7]
                for scouting_time in times:
                    events_out.writerow(
                        {"GameID": game_id, "UID": uid, "Rank": rank, "Race": race,
                         "ScoutStartTime": scouting_time.start_time, "ScoutEndTime": scouting_time.end_time,
                         "ScoutType": scouting_time.scouting_type})
                uid = fields[4]
                rank = fields[5]
                times = fields[6]
                race = fields[8]
                for scouting_time in times:
                    events_out.writerow(
                        {"GameID": game_id, "UID": uid, "Rank": rank, "Race": race,
                         "ScoutStartTime": scouting_time.start_time, "ScoutEndTime": scouting_time.end_time,
                         "ScoutType": scouting_time.scouting_type})


if __name__ == "__main__":
    sc2reader.engine.register_plugin(APMTracker())
    sc2reader.engine.register_plugin(SelectionTracker())
    sc2reader.engine.register_plugin(ModifiedRank())
    sc2reader.engine.register_plugin(ActiveSelection())
    bt = BaseTracker()
    #     bt.logger.setLevel(logging.ERROR)
    #     bt.logger.addHandler(logging.StreamHandler(sys.stdout))
    sc2reader.engine.register_plugin(bt)
    #     sc2reader.log_utils.add_log_handler(logging.StreamHandler(sys.stdout), "INFO")

    results = run(generateFields)
    save(results, "scouting_instances_gm")
    # with open("missing_unit_speeds.txt", "r") as file:
    #     file.writelines(scouting_detector.missing_units)
    # with open("missing_unit_vision.txt", "r") as file:
    #     file.writelines(unit_prediction.missing_units)
