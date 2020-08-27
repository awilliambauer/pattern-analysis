# Used to create a csv of information about the average time interval (in seconds)
# in between periods of scouting for players in StarCraft 2
# Alison Cameron
# July 2020

import csv
import sc2reader
import time
import math
from multiprocessing import Pool
from collections import Counter
import scouting_detector
import scouting_stats
from sc2reader.engine.plugins import SelectionTracker, APMTracker
from selection_plugin import ActiveSelection
from base_plugins import BaseTracker

def generateFields(filename):
    '''generateFields takes in a filename of a replay, loads it and gathers necessary
    statistics, and returns the statistics in a tuple. It is to be used to write
    these stats to a csv.'''
    # loading the replay
    try:
        # skipping non-replay files in the directory
        if filename[-9:] != "SC2Replay":
            raise RuntimeError()

        # extracting the game id and adding the correct tag
        # pathname = "practice_replays/" + filename
        pathname = "/Accounts/awb/pattern-analysis/starcraft/replays/" + filename
        game_id = filename.split("_")[1].split(".")[0]
        if filename.startswith("ggg"):
            game_id = "ggg-" + game_id
        elif filename.startswith("spawningtool"):
            game_id = "st-" + game_id

        # loading the replay
        try:
            r = sc2reader.load_replay(pathname)
        except:
            print(filename + " cannot load using sc2reader due to an internal ValueError")
            raise RuntimeError()

        team1_avg_int, team2_avg_int = scouting_detector.scouting_interval(r)
        team1_rank, team1_rel_rank, team2_rank, team2_rel_rank = scouting_stats.ranking_stats(r)

        team1_uid = r.players[0].detail_data['bnet']['uid']
        team2_uid = r.players[1].detail_data['bnet']['uid']

        fields = (game_id, team1_uid, team1_rank, team1_avg_int, team2_uid, team2_rank, team2_avg_int)
        return fields
    except:
        return

def writeToCsv():
    '''writeToCsv gathers information about all valid replays and writes
    that information to a csv for analysis in R. This file in particular
    contains information about the average time (in seconds) in between
    instances of scouting for each player.'''
    files = []
    # valid_game_ids.txt musst be produced first by running scouting_stats.py
    # with the command line argument -w
    games = open("valid_game_ids.txt", 'r')
    for line in games:
        files.append(line.strip())
    games.close()

    pool = Pool(20)
    results = pool.map(generateFields, files)
    pool.close()
    pool.join()

    with open("intervals.csv", 'w', newline='') as my_csv:
        events_out = csv.DictWriter(my_csv, fieldnames=["GameID", "UID", "Rank", "Interval"])
        events_out.writeheader()
        for fields in results:
            if fields:
                game_id = fields[0]
                team1_uid, team1_rank, team1_avg_int = fields[1], fields[2], fields[3]
                team2_uid, team2_rank, team2_avg_int = fields[4], fields[5], fields[6]
                if (team1_avg_int != -1) and not(math.isnan(team1_rank)):
                    events_out.writerow({"GameID": game_id, "UID": team1_uid, "Rank": team1_rank, "Interval": team1_avg_int})
                if (team2_avg_int != -1) and not(math.isnan(team2_rank)):
                    events_out.writerow({"GameID": game_id, "UID": team2_uid, "Rank": team2_rank, "Interval": team2_avg_int})



if __name__ == "__main__":
    t1 = time.time()

    sc2reader.engine.register_plugin(APMTracker())
    sc2reader.engine.register_plugin(SelectionTracker())
    sc2reader.engine.register_plugin(ActiveSelection())
    sc2reader.engine.register_plugin(BaseTracker())

    writeToCsv()
    deltatime = time.time()-t1
    print("Run time: ", "{:2d}".format(int(deltatime//60)), "minutes and", "{:05.2f}".format(deltatime%60), "seconds")
