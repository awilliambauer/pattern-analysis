# Intended to be used for data visualization of when players scout
# during StarCraft 2

import csv
import sc2reader
import time
import traceback
from multiprocessing import Pool, cpu_count
from collections import Counter
import scouting_detector
import scouting_stats
from sc2reader.engine.plugins import SelectionTracker, APMTracker
from selection_plugin import ActiveSelection
from base_plugins import BaseTracker
from modified_rank_plugin import ModifiedRank
import numpy as np
from functools import partial
import battle_detector



def generateFields(filename):
    '''generateFields takes in a filename of a replay, loads it and gathers necessary
    statistics, and returns the statistics in a tuple. It is to be used to write
    these stats to a csv. It also takes in an integer (1 or 2), which indicates
    which statistics will be gathered. In this case, generateFields gathers
    each point in the game where a period of scouting is initiated. Inputting a
    1 will return times as a fraction of the total game time, whereas inputting
    a 2 will return absolute frames.'''
    # loading the replay
    try:
        # skipping non-replay files in the directory
        if filename[-9:] != "SC2Replay":
            raise RuntimeError()

        # extracting the game id and adding the correct tag
        # pathname = "practice_replays/" + filename
        pathname = "/Accounts/awb-data/replays/" + filename
        game_id = filename.split("_")[1].split(".")[0]
        if filename.startswith("ggg"):
            game_id = "ggg-" + game_id
        elif filename.startswith("spawningtool"):
            game_id = "st-" + game_id

        # loading the replay
        try:
            r = sc2reader.load_replay(pathname)
            # print(r.filename, r.players[0].highest_league, r.players[1].highest_league)
            if any(v != (0, {}) for v in r.plugin_result.values()):
                print(pathname, r.plugin_result)
        except:
            print(filename, "cannot load using sc2reader due to an internal ValueError")
            traceback.print_exc()
            raise

        # team1_times, team2_times = scouting_detector.scouting_times(r, which)
        team1_rank, team1_rel_rank, team2_rank, team2_rel_rank = scouting_stats.ranking_stats(r)
        battles, harassing = battle_detector.buildBattleList(r)

        frames = r.frames
        seconds = r.real_length.total_seconds()
        battle_time = []
        battle_duration = []
        harassing_time = []
        harassing_duration = []
        for battle in battles:
            starttime = (battle[0]/frames)*seconds
            endtime = (battle[1]/frames)*seconds
            duration = endtime - starttime
            battle_duration.append(duration)
            battle_time.append(starttime)
        
        for harass in harassing:
            starttime = (harass[0]/frames)*seconds
            endtime = (harass[1]/frames)*seconds
            duration = endtime - starttime
            harassing_duration.append(duration)
            harassing_time.append(starttime)

        team1_uid = r.players[0].detail_data['bnet']['uid']
        team2_uid = r.players[1].detail_data['bnet']['uid']

        fields = [game_id, team1_uid, team1_rank, team2_uid, team2_rank,\
                    battle_time, battle_duration, harassing_time, harassing_duration]
        return fields

    except KeyboardInterrupt:
        raise
    except:
        print("\nReturning none\n")
        #traceback.print_exc()
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
    files = []
    # valid_game_ids.txt musst be produced first by running scouting_stats.py
    # with the command line argument -w
    # games = open("valid_replay_filenames.txt", 'r')
    games = open("valid_replay_filenames.txt", 'r')
    for line in games:
        files.append(line.strip())
    games.close()

    pool = Pool(min(cpu_count(), 20))
    results = pool.map(generateFields, files)
    pool.close()
    pool.join()

    with open(filename, 'w', newline = '') as my_csv:
        if which == 1:
            events_out = csv.DictWriter(my_csv, fieldnames=["GameID", "UID", "Rank", \
                                                        "BattleTime", "BattleDuration"])                     
            events_out.writeheader()
            for fields in results:
                if fields:
                    game_id = fields[0]
                    uid_1 = fields[1]
                    rank_1 = fields[2]
                    uid_2 = fields[3]
                    rank_2 = fields[4]
                    battle_time = fields[5]
                    battle_duration = fields[6]
                    for i in range(0, len(battle_time)):
                        events_out.writerow({"GameID": game_id, "UID": uid_1, "Rank": rank_1, \
                            "BattleTime": battle_time[i], "BattleDuration": battle_duration[i]})
                        events_out.writerow({"GameID": game_id, "UID": uid_2, "Rank": rank_2, \
                            "BattleTime": battle_time[i], "BattleDuration": battle_duration[i]})
        else:
            events_out = csv.DictWriter(my_csv, fieldnames=["GameID", "UID", "Rank", \
                                                        "HarassingTime", "HarassingDuration"])                     
            events_out.writeheader()
            for fields in results:
                if fields:
                    game_id = fields[0]
                    uid_1 = fields[1]
                    rank_1 = fields[2]
                    uid_2 = fields[3]
                    rank_2 = fields[4]
                    harassing_time = fields[7]
                    harassing_duration = fields[8]
                    for i in range(0, len(harassing_time)):
                        events_out.writerow({"GameID": game_id, "UID": uid_1, "Rank": rank_1, \
                            "HarassingTime": harassing_time[i], "HarassingDuration": harassing_duration[i]})
                        events_out.writerow({"GameID": game_id, "UID": uid_2, "Rank": rank_2, \
                            "HarassingTime": harassing_time[i], "HarassingDuration": harassing_duration[i]})


if __name__ == "__main__":
    sc2reader.engine.register_plugin(APMTracker())
    sc2reader.engine.register_plugin(SelectionTracker())
    sc2reader.engine.register_plugin(ActiveSelection())
    sc2reader.engine.register_plugin(ModifiedRank())
    # bt = BaseTracker()
    # bt.logger.setLevel(logging.ERROR)
    # bt.logger.addHandler(logging.StreamHandler(sys.stdout))
    # sc2reader.engine.register_plugin(bt)
    # sc2reader.log_utils.add_log_handler(logging.StreamHandler(sys.stdout), "INFO")

    t1 = time.time()
    writeToCsv(1, "battle_time.csv")
    writeToCsv(2, "harassing_time.csv")
    deltatime = time.time()-t1
    print("Run time: ", "{:2d}".format(int(deltatime//60)), "minutes and", "{:05.2f}".format(deltatime%60), "seconds")
