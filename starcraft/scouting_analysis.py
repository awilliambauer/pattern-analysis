# A script to write information about players' scouting behavior is StarCraft 2
# to a CSV
# Alison Cameron
# August 2020

import sc2reader
import csv
import os
import sys
import scouting_detector
import scouting_stats
from multiprocessing import Pool, cpu_count
import argparse
import time
import math
from sc2reader.engine.plugins import SelectionTracker, APMTracker
from selection_plugin import ActiveSelection
from base_plugins import BaseTracker
import traceback


def generateFields(filename):
    '''generateFields takes in a filename of a replay, loads it and gathers necessary
    statistics, and returns the statistics in a tuple. It is to be used to write
    these stats to a csv.'''
    try:
        # skipping non-replay files in the directory
        if filename[-9:] != "SC2Replay":
            raise RuntimeError()

        # extracting the game id and adding the correct tag
        #pathname = "practice_replays/" + filename
        pathname = "/Accounts/awb/pattern-analysis/starcraft/replays/" + filename
        game_id = filename.split("_")[1].split(".")[0]
        if filename.startswith("ggg"):
            game_id = "ggg-" + game_id
        elif filename.startswith("spawningtool"):
            game_id = "st-" + game_id
        elif filename.startswith("dropsc"):
            game_id = "ds-" + game_id

        # loading the replay
        try:
            r = sc2reader.load_replay(pathname)
            if any(v != (0, {}) for v in r.plugin_result.values()):
                print(pathname, r.plugin_result)
        except:
            print(filename, "cannot load using sc2reader due to an internal ValueError")
            traceback.print_exc()
            raise RuntimeError()

        # collecting stats and values
        analysis_dict = scouting_detector.scouting_analysis(r)
        team1_rank, team1_rel_rank, team2_rank, team2_rel_rank = scouting_stats.ranking_stats(r)

        team1_uid = r.players[0].detail_data['bnet']['uid']
        team2_uid = r.players[1].detail_data['bnet']['uid']

        team1_list = [game_id, team1_uid, team1_rank] + analysis_dict[1]
        team2_list = [game_id, team2_uid, team2_rank] + analysis_dict[2]

        # creating the fields based on who won
        if r.winner.number == 1:
            fields = team1_list + [1] + team2_list + [0]
        elif r.winner.number == 2:
            fields = team1_list + [0] + team2_list + [1]
        return fields
    except:
        return

def writeToCsv():
    '''writeToCsv gathers information about all valid replays and writes
    that information to a csv for analysis in R. This file in particular
    contains information about each player's category of scouting, if they
    complete an initial scouting, if they consistently scout between battles,
    and other similar metrics.'''
    files = []
    # valid_game_ids.txt musst be produced first by running scouting_stats.py
    # with the command line argument -w
    games = open("valid_game_ids.txt", 'r')
    for line in games:
        files.append(line.strip())
    games.close()

    with open("scouting_analysis.csv", 'w', newline='') as my_csv:
        events_out = csv.DictWriter(my_csv, fieldnames=["GameID", "UID", "Rank", "Category",
                                         "InitialScouting", "BaseScouting", "NewAreas",
                                         "BetweenBattles", "Win"])
        events_out.writeheader()

        with Pool(min(cpu_count(), 20)) as pool:
            results = pool.map(generateFields, files)

        for fields in results:
            if fields: # generateFields will return None for invalid replays
                # writing 1 line to the csv for each player and their respective stats
                events_out.writerow({"GameID": fields[0], "UID": fields[1], "Rank": fields[2],
                                    "Category": fields[3], "InitialScouting": fields[4],
                                    "BaseScouting": fields[5], "NewAreas": fields[6],
                                    "BetweenBattles": fields[7], "Win": fields[8]})
                events_out.writerow({"GameID": fields[9], "UID": fields[10], "Rank": fields[11],
                                    "Category": fields[12], "InitialScouting": fields[13],
                                    "BaseScouting": fields[14], "NewAreas": fields[15],
                                    "BetweenBattles": fields[16], "Win": fields[17]})

if __name__ == "__main__":
    sc2reader.engine.register_plugin(APMTracker())
    sc2reader.engine.register_plugin(SelectionTracker())
    sc2reader.engine.register_plugin(ActiveSelection())
    sc2reader.engine.register_plugin(BaseTracker())

    t1 = time.time()
    writeToCsv()
    deltatime = time.time()-t1
    print("Run time: ", "{:2d}".format(int(deltatime//60)), "minutes and", "{:05.2f}".format(deltatime%60), "seconds")
