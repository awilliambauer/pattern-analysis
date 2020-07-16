#Refined version of scouting_stats.py with multiprocessing

import sc2reader
import csv
import os
import sys
import scouting_detector
from multiprocessing import Pool
import argparse
import time
import math
import control_groups
from collections import Counter
from sc2reader.engine.plugins import SelectionTracker, APMTracker
from selection_plugin import ActiveSelection

sc2reader.engine.register_plugin(APMTracker())
sc2reader.engine.register_plugin(SelectionTracker())
sc2reader.engine.register_plugin(ActiveSelection())

def generateFields(filename):
    try:
        #skipping non-replay files in the directory
        if filename[-9:] != "SC2Replay":
            raise RuntimeError()

        #extracting the game id and adding the correct tag
        #pathname = "practice_replays/" + filename
        pathname = "/Accounts/awb/pattern-analysis/starcraft/replays/" + filename
        game_id = filename.split("_")[1].split(".")[0]
        if filename.startswith("ggg"):
            game_id = "ggg-" + game_id
        elif filename.startswith("spawningtool"):
            game_id = "st-" + game_id

        try:
            r = sc2reader.load_replay(pathname)
        except:
            print(filename + " cannot load using sc2reader due to an internal ValueError")
            raise RuntimeError()

        if not(r.map_name in map_counter.keys()):
            print(filename + " is not played on an official Blizzard map")
            raise RuntimeError()

        team1_nums, team1_fraction, team2_nums, team2_fraction, winner = scouting_detector.detect_scouting(r)
        team1_rank, team1_rel, team2_rank, team2_rel = ranking_stats(r)
        team1_cps, team1_peace_rate, team1_battle_rate, team2_cps, team2_peace_rate, team2_battle_rate = control_groups.control_group_stats(r)

        team1_apm = r.players[0].avg_apm - r.players[1].avg_apm
        team2_apm = r.players[1].avg_apm - r.players[0].avg_apm

        if winner == 1:
            fields = (game_id, team1_nums, team1_fraction, team1_apm, team1_rank, team1_rel, team1_cps, team1_peace_rate, team1_battle_rate, 1,
                        game_id, team2_nums, team2_fraction, team2_apm, team2_rank, team2_rel, team2_cps, team2_peace_rate, team2_battle_rate, 0,
                        r.map_name)
        elif winner == 2:
            fields = (game_id, team1_nums, team1_fraction, team1_apm, team1_rank, team1_rel, team1_cps, team1_peace_rate, team1_battle_rate, 0,
                        game_id, team2_nums, team2_fraction, team2_apm, team2_rank, team2_rel, team2_cps, team2_peace_rate, team2_battle_rate, 1,
                        r.map_name)
        return fields
    except:
        return

def ranking_stats(replay):
    p1_rank = replay.players[0].highest_league
    p2_rank = replay.players[1].highest_league

    #checking if rank exists for each player
    if (p1_rank == 0) or (p1_rank == 8):
        p1_rank = math.nan
    if (p2_rank == 0) or (p2_rank == 8):
        p2_rank = math.nan

    #if rankings exist for both players, then calculate relative ranks
    if not math.isnan(p1_rank) and not math.isnan(p2_rank):
        p1_rel = p1_rank - p2_rank
        p2_rel = p2_rank - p1_rank
    #if rankings don't exist for both players, then relative ranks are NaN
    else:
        p1_rel, p2_rel = math.nan, math.nan

    return p1_rank, p1_rel, p2_rank, p2_rel

def initializeCounter():
    bliz_maps = open("blizzard_maps.txt", 'r')
    for line in bliz_maps:
        map = line.strip()
        map_counter[map] = 0

map_counter = Counter()
initializeCounter()

if __name__ == "__main__":
    #command line argument for debugging
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', nargs=2, type=int) #debugging, input indeces
    parser.add_argument('--w', action='store_true')
    args = parser.parse_args()

    if args.d:
        startidx = args.d[0]
        endidx = args.d[1]

    if args.w:
        #files = os.listdir("practice_replays")
        files = os.listdir("/Accounts/awb/pattern-analysis/starcraft/replays")
        valid_games = []
    else:
        files = []
        games = open("valid_game_ids.txt", 'r')
        for line in games:
            files.append(line.strip())
        games.close()

    t1 = time.time()
    with open("scouting_stats.csv", 'w', newline = '') as fp:
        events_out = csv.DictWriter(fp, fieldnames=["GameID", "ScoutingFrequency", "ScoutingTime", "APM",
                                                    "Rank", "RelativeRank", "CPS", "PeaceRate",
                                                    "BattleRate", "Win"])
        events_out.writeheader()
        #debugging
        if args.d:
            print("debugging!")
            i = startidx
            for filename in files[startidx:endidx]:
                print("file #: ", i, "file name: ", filename)
                i += 1
                fields = generateFields(filename)
                if fields:
                    if args.w:
                        if fields[0].startswith("ggg-"):
                            filename = "gggreplays_{}.SC2Replay".format(fields[0][4:])
                        elif fields[0].startswith("st-"):
                            filename = "spawningtool_{}.SC2Replay".format(fields[0][3:])
                        valid_games.append(filename)
                    map_counter[fields[20]] += 1
                    events_out.writerow({"GameID": fields[0], "ScoutingFrequency": fields[1],
                                        "ScoutingTime": fields[2], "APM": fields[3], "Rank": fields[4],
                                        "RelativeRank": fields[5], "CPS": fields[6], "PeaceRate": fields[7],
                                        "BattleRate": fields[8], "Win": fields[9]})
                    events_out.writerow({"GameID": fields[10], "ScoutingFrequency": fields[11],
                                        "ScoutingTime": fields[12], "APM": fields[13], "Rank": fields[14],
                                        "RelativeRank": fields[15], "CPS": fields[16], "PeaceRate": fields[17],
                                        "BattleRate": fields[18], "Win": fields[19]})
        #running with multiprocessing
        else:
            pool = Pool(40)
            results = pool.map(generateFields, files)
            pool.close()
            pool.join()
            for fields in results:
                if fields:
                    if args.w:
                        if fields[0].startswith("ggg-"):
                            filename = "gggreplays_{}.SC2Replay".format(fields[0][4:])
                        elif fields[0].startswith("st-"):
                            filename = "spawningtool_{}.SC2Replay".format(fields[0][3:])
                        valid_games.append(filename)
                    map_counter[fields[20]] += 1
                    events_out.writerow({"GameID": fields[0], "ScoutingFrequency": fields[1],
                                        "ScoutingTime": fields[2], "APM": fields[3], "Rank": fields[4],
                                        "RelativeRank": fields[5], "CPS": fields[6], "PeaceRate": fields[7],
                                        "BattleRate": fields[8], "Win": fields[9]})
                    events_out.writerow({"GameID": fields[10], "ScoutingFrequency": fields[11],
                                        "ScoutingTime": fields[12], "APM": fields[13], "Rank": fields[14],
                                        "RelativeRank": fields[15], "CPS": fields[16], "PeaceRate": fields[17],
                                        "BattleRate": fields[18], "Win": fields[19]})
    #writing to a new file if it doesn't already exist and command line argument exists
    if args.w and not(os.path.exists("valid_game_ids.txt")):
        with open("valid_game_ids.txt", 'w') as file:
            for game_id in valid_games:
                file.write(game_id + "\n")
    #print(map_counter)
    print("Run time: ", (time.time()-t1)/60)
