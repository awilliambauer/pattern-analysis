# Used to write statistics about player behavior in StarCraft 2 to a csv
# Alison Cameron
# July 2020

import sc2reader
import csv
import os
import sys
import scouting_detector
from multiprocessing import Pool, cpu_count
import argparse
import time
import math
import control_groups
from collections import Counter
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

        # checking the map
        if not(r.map_name in map_counter.keys()):
            print(filename + " is not played on an official Blizzard map")
            raise RuntimeError()

        # collecting stats and values
        team1_freq, team1_cat, team2_freq, team2_cat, winner = scouting_detector.scouting_freq_and_cat(r)
        team1_rank, team1_rel_rank, team2_rank, team2_rel_rank = ranking_stats(r)
        team1_cps, team1_peace_rate, team1_battle_rate, team2_cps, team2_peace_rate, team2_battle_rate = control_groups.control_group_stats(r)
        team1_rel_freq = team1_freq - team2_freq
        team2_rel_freq = team2_freq - team1_freq

        team1_rel_cps = team1_cps - team2_cps
        team2_rel_cps = team2_cps - team1_cps
        team1_rel_pr = team1_peace_rate - team2_peace_rate
        team2_rel_pr = team2_peace_rate - team1_peace_rate
        team1_rel_br = team1_battle_rate - team2_battle_rate
        team2_rel_br = team2_battle_rate - team1_battle_rate

        # changing actions per minute to actions per second to match other data
        team1_aps = (r.players[0].avg_apm) / 60
        team2_aps = (r.players[1].avg_apm) / 60
        team1_rel_aps = (r.players[0].avg_apm - r.players[1].avg_apm) / 60
        team2_rel_aps = (r.players[1].avg_apm - r.players[0].avg_apm) / 60
        team1_uid = r.players[0].detail_data['bnet']['uid']
        team2_uid = r.players[1].detail_data['bnet']['uid']
        # creating the fields based on who won
        if winner == 1:
            fields = (game_id, team1_uid, team1_cat, team1_rank, team1_rel_rank,
                        team1_freq, team1_rel_freq, team1_aps, team1_rel_aps,
                        team1_cps, team1_rel_cps, team1_peace_rate, team1_rel_pr,
                        team1_battle_rate, team1_rel_br, 1,
                      game_id, team2_uid, team2_cat, team2_rank, team2_rel_rank,
                        team2_freq, team2_rel_freq, team2_aps, team2_rel_aps,
                        team2_cps, team2_rel_cps, team2_peace_rate, team2_rel_pr,
                        team2_battle_rate, team2_rel_br, 0,
                      r.map_name)
        elif winner == 2:
            fields = (game_id, team1_uid, team1_cat, team1_rank, team1_rel_rank,
                        team1_freq, team1_rel_freq, team1_aps, team1_rel_aps,
                        team1_cps, team1_rel_cps, team1_peace_rate, team1_rel_pr,
                        team1_battle_rate, team1_rel_br, 0,
                      game_id, team2_uid, team1_cat, team2_rank, team2_rel_rank,
                        team2_freq, team2_rel_freq, team2_aps, team2_rel_aps,
                        team2_cps, team2_rel_cps, team2_peace_rate, team2_rel_pr,
                        team2_battle_rate, team2_rel_br, 1,
                      r.map_name)
        return fields
    except:
        return

def ranking_stats(replay):
    '''ranking_stats takes in a previously loaded replay and returns each player's
    rank and their rank relative to their opponent. If rankings don't exist,
    then the rank is NaN and so is the relative rank.'''
    p1_rank = replay.players[0].highest_league
    p2_rank = replay.players[1].highest_league

    # checking if rank exists for each player
    if (p1_rank == 0) or (p1_rank == 8):
        p1_rank = math.nan
    if (p2_rank == 0) or (p2_rank == 8):
        p2_rank = math.nan

    # if rankings exist for both players, then calculate relative ranks
    if not math.isnan(p1_rank) and not math.isnan(p2_rank):
        p1_rel = p1_rank - p2_rank
        p2_rel = p2_rank - p1_rank
    # if rankings don't exist for both players, then relative ranks are NaN
    else:
        p1_rel, p2_rel = math.nan, math.nan

    return p1_rank, p1_rel, p2_rank, p2_rel

def initializeCounter():
    '''Adds all valid blizzard maps to the map counter. Requires
    blizzard_maps.txt to be in the same directory as this file.'''
    bliz_maps = open("blizzard_maps.txt", 'r')
    for line in bliz_maps:
        map = line.strip()
        map_counter[map] = 0

map_counter = Counter()
initializeCounter()

def writeToCsv(write, debug, start, end):
    '''Write to csv takes in command line arguments write, debug, start, and end,
    and will create a csv with values and statistics for every replay
    in the replay directory.

    If write is true, writeToCsv will also create a text file containing
    the filename of all valid games. If write is false, writeToCsv will
    read a previously created text file with all valid game filenames,
    thus saving time.

    If debug is true, the replays will be processed in order and one at a time,
    from the start to end (input values intended to be indeces). If debug is
    false, the replays will be handled using multiprocessing.'''

    # obtain a list of filenames from either the directory or a text file
    if write:
        # files = os.listdir("practice_replays")
        files = os.listdir("/Accounts/awb/pattern-analysis/starcraft/replays")
        valid_games = []
    else:
        files = []
        games = open("valid_game_ids.txt", 'r')
        for line in games:
            files.append(line.strip())
        games.close()

    # open the csv and begin to write to it
    with open("scouting_stats.csv", 'w', newline = '') as fp:
        events_out = csv.DictWriter(fp, fieldnames=["GameID", "UID", "ScoutingCategory",
                                    "Rank", "RelRank", "ScoutingFrequency",
                                    "RelScoutingFrequency", "APS", "RelAPS",
                                    "CPS", "RelCPS", "PeaceRate", "RelPeaceRate",
                                    "BattleRate", "RelBattleRate", "Win"])
        events_out.writeheader()
        # debugging
        if debug:
            print("debugging!")
            i = start
            # processing replay files in order, from start to end
            for filename in files[start:end]:
                print("file #: ", i, "file name: ", filename)
                i += 1
                fields = generateFields(filename)
                if fields: # generateFields will return None for invalid replays
                    if write:
                        # formatting filenames to add to the text file
                        if fields[0].startswith("ggg-"):
                            filename = "gggreplays_{}.SC2Replay".format(fields[0][4:])
                        elif fields[0].startswith("st-"):
                            filename = "spawningtool_{}.SC2Replay".format(fields[0][3:])
                        elif fields[0].startswith("ds-"):
                            filename = "dropsc_{}.SC2Replay".format(fields[0][3:])
                        valid_games.append(filename)
                    # updating the map counter
                    map_counter[fields[32]] += 1
                    # writing 1 line to the csv for each player and their respective stats
                    events_out.writerow({"GameID":fields[0], "UID":fields[1],
                                        "ScoutingCategory":fields[2], "Rank":fields[3],
                                        "RelRank":fields[4], "ScoutingFrequency":fields[5],
                                        "RelScoutingFrequency":fields[6], "APS":fields[7],
                                        "RelAPS":fields[8], "CPS":fields[9], "RelCPS":fields[10],
                                        "PeaceRate":fields[11], "RelPeaceRate":fields[12],
                                        "BattleRate":fields[13], "RelBattleRate":fields[14],
                                        "Win":fields[15]})
                    events_out.writerow({"GameID":fields[16], "UID":fields[17],
                                        "ScoutingCategory":fields[18], "Rank":fields[19],
                                        "RelRank":fields[20], "ScoutingFrequency":fields[21],
                                        "RelScoutingFrequency":fields[22], "APS":fields[23],
                                        "RelAPS":fields[24], "CPS":fields[25], "RelCPS":fields[26],
                                        "PeaceRate":fields[27], "RelPeaceRate":fields[28],
                                        "BattleRate":fields[29], "RelBattleRate":fields[30],
                                        "Win":fields[31]})
        # running with multiprocessing
        else:
            pool = Pool(min(cpu_count(), 20))
            results = pool.map(generateFields, files)
            pool.close()
            pool.join()
            for fields in results:
                if fields: # generateFields will return None for invalid replays
                    if write:
                        # formatting filenames to add to the text file
                        if fields[0].startswith("ggg-"):
                            filename = "gggreplays_{}.SC2Replay".format(fields[0][4:])
                        elif fields[0].startswith("st-"):
                            filename = "spawningtool_{}.SC2Replay".format(fields[0][3:])
                        elif fields[0].startswith("ds-"):
                            filename = "dropsc_{}.SC2Replay".format(fields[0][3:])
                        valid_games.append(filename)
                    # updating the map counter
                    map_counter[fields[32]] += 1
                    # writing 1 line to the csv for each player and their respective stats
                    events_out.writerow({"GameID":fields[0], "UID":fields[1],
                                        "ScoutingCategory":fields[2], "Rank":fields[3],
                                        "RelRank":fields[4], "ScoutingFrequency":fields[5],
                                        "RelScoutingFrequency":fields[6], "APS":fields[7],
                                        "RelAPS":fields[8], "CPS":fields[9], "RelCPS":fields[10],
                                        "PeaceRate":fields[11], "RelPeaceRate":fields[12],
                                        "BattleRate":fields[13], "RelBattleRate":fields[14],
                                        "Win":fields[15]})
                    events_out.writerow({"GameID":fields[16], "UID":fields[17],
                                        "ScoutingCategory":fields[18], "Rank":fields[19],
                                        "RelRank":fields[20], "ScoutingFrequency":fields[21],
                                        "RelScoutingFrequency":fields[22], "APS":fields[23],
                                        "RelAPS":fields[24], "CPS":fields[25], "RelCPS":fields[26],
                                        "PeaceRate":fields[27], "RelPeaceRate":fields[28],
                                        "BattleRate":fields[29], "RelBattleRate":fields[30],
                                        "Win":fields[31]})

    # writing to a new text file if the command line arguments indicate to do so
    if write:
        with open("valid_game_ids.txt", 'w') as file:
            for game_id in valid_games:
                file.write(game_id + "\n")
    # print(map_counter)


if __name__ == "__main__":
    '''This main function parses command line arguments and calls
    writeToCsv, which will write statistics to a csv for each
    StarCraft 2 replay file in a directory.'''
    sc2reader.engine.register_plugin(APMTracker())
    sc2reader.engine.register_plugin(SelectionTracker())
    sc2reader.engine.register_plugin(ActiveSelection())
    sc2reader.engine.register_plugin(BaseTracker())

    t1 = time.time()
    # command line arguments for debugging and saving a list of valid replays
    parser = argparse.ArgumentParser()
    # debugging argument - if included, you must also include the start and end
    # index of which files in the directory you want to be processed
    # ex: python scouting_stats.py --d 0 50   will process the first 50
    # replays in the directory, print extra information, and process replays one by one
    # if not included, the program will use multiprocessing and will run much faster
    parser.add_argument('--d', nargs=2, type=int)
    # saving list of valid replays - if included, writeToCsv will create a text file
    # of all valid game filenames. If not included, writeToCsv will expect that
    # text file to exist and will read filenames from the text file.
    parser.add_argument('--w', action='store_true')
    args = parser.parse_args()

    if args.d:
        start = args.d[0]
        end = args.d[1]
    else:
        start = 0
        end = 0

    writeToCsv(args.w, args.d, start, end)
    deltatime = time.time()-t1
    print("Run time: ", "{:2d}".format(int(deltatime//60)), "minutes and", "{:05.2f}".format(deltatime%60), "seconds")
