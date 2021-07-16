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
import scouting_stats
from itertools import repeat
import scouting_stats
import file_locations
from map_path_generation import load_path_data, get_all_possible_names
from sc2reader.engine.plugins import SelectionTracker, APMTracker
from selection_plugin import ActiveSelection
from base_plugins import BaseTracker
from replay_verification import group_replays_by_map, get_map_name_groups
import numpy as np
import traceback

try:
    from reprlib import repr
except ImportError:
    pass


def generateFields(filename, which, map_path_data):
    '''generateFields takes in a filename of a replay, loads it and gathers necessary
    statistics, and returns the statistics in a tuple. It is to be used to write
    these stats to a csv. It also takes in an integer (1 or 2), which indicates
    which statistics will be gathered. In this case, generateFields gathers
    each point in the game where a period of scouting is initiated. Inputting a
    1 will return times as a fraction of the total game time, whereas inputting
    a 2 will return absolute frames.'''
    # loading the replay
    try:
        print("analyzing replay", filename)
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
        print("loaded replay", filename, "len:", r.real_length)
        team1_times, team2_times = scouting_stats.scouting_times(r, which, map_path_data)
        team1_rank, team1_rel_rank, team2_rank, team2_rel_rank = scouting_stats.ranking_stats(r)
        print("finished scouting stats")
        team1_uid = r.players[0].detail_data['bnet']['uid']
        team2_uid = r.players[1].detail_data['bnet']['uid']
        team1_race = r.players[0].play_race
        team2_race = r.players[1].play_race
        fields = [game_id, team1_uid, team1_rank, team1_times, team2_uid, team2_rank, team2_times, team1_race,
                  team2_race]
        print("generated fields for replay", filename)
        print("time:", time.time() - t)
        return fields

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
    count = 0
    with Pool(min(cpu_count(), 10)) as pool:
        for map_name_group, replays in group_replays_by_map(file_locations.REPLAY_INFO_FILE).items():
            if "EternalEmpireLE" not in map_name_group:
                continue
            if count > 1000:
                break
            map_path_data = load_path_data(map_name_group)
            print("loaded path data for map", map_name_group, "with", len(replays), "replays")
            count += len(replays)
            map_time = time.time()
            new_results = map(partial(generateFields, which=which, map_path_data=map_path_data), replays[0:10])
            print("analyzing", len(replays), "replays for map", map_name_group, "took", time.time() - map_time)
            for result in new_results:
                results.append(result)
            break
    with open(filename, 'w', newline='') as my_csv:
        events_out = csv.DictWriter(my_csv, fieldnames=["GameID", "UID", "Rank", "Race", "ScoutTime"])
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
                        {"GameID": game_id, "UID": uid, "Rank": rank, "Race": race, "ScoutTime": scouting_time})
                uid = fields[4]
                rank = fields[5]
                times = fields[6]
                race = fields[8]
                for scouting_time in times:
                    events_out.writerow(
                        {"GameID": game_id, "UID": uid, "Rank": rank, "Race": race, "ScoutTime": scouting_time})


if __name__ == "__main__":
    sc2reader.engine.register_plugin(APMTracker())
    sc2reader.engine.register_plugin(SelectionTracker())
    sc2reader.engine.register_plugin(ActiveSelection())
    bt = BaseTracker()
    #     bt.logger.setLevel(logging.ERROR)
    #     bt.logger.addHandler(logging.StreamHandler(sys.stdout))
    sc2reader.engine.register_plugin(bt)
    #     sc2reader.log_utils.add_log_handler(logging.StreamHandler(sys.stdout), "INFO")

    t1 = time.time()
    #writeToCsv(1, "scouting_time_fraction.csv")
    writeToCsv(2, "scouting_time_frames1.csv")
    deltatime = time.time() - t1
    print("Run time: ", "{:2d}".format(int(deltatime // 60)), "minutes and", "{:05.2f}".format(deltatime % 60),
          "seconds")
