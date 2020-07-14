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

def generateFields(filename):
    game_id = filename.split("_")[1].split(".")[0]
    pathname = "/tmp/replays/" + filename
    try:
        try:
            r = sc2reader.load_replay(pathname)
        except:
            print(filename + " cannot load using sc2reader due to an internal ValueError")
            raise RuntimeError()

        team1_nums, team1_fraction, team1_apm, team2_nums, team2_fraction, team2_apm, winner = scouting_detector.detect_scouting(r)
        team1_rank, team1_rel, team2_rank, team2_rel = ranking_stats(r)
        team1_cps, team1_ratio, team2_cps, team2_ratio = control_groups.control_group_stats(r)
        if winner == 1:
            fields = (game_id, team1_nums, team1_fraction, team1_apm, team1_rank, team1_rel, team1_cps, team1_ratio, 1,
                        game_id, team2_nums, team2_fraction, team2_apm, team2_rank, team2_rel, team2_cps, team2_ratio, 0)
        elif winner == 2:
            fields = (game_id, team1_nums, team1_fraction, team1_apm, team1_rank, team1_rel, team1_cps, team1_ratio, 0,
                        game_id, team2_nums, team2_fraction, team2_apm, team2_rank, team2_rel, team2_cps, team2_ratio, 1)
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
        files = os.listdir("/tmp/replays")
    else:
        files = []
        games = open("valid_game_ids.txt", 'r')
        for line in games:
            files.append(line.strip())
        games.close()

    valid_games = []
    t1 = time.time()
    with open("scouting_stats.csv", 'w', newline = '') as fp:
        events_out = csv.DictWriter(fp, fieldnames=["GameID", "ScoutingFrequency", "ScoutingTime", "APM",
                                                    "Rank", "Relative Rank", "CPS", "PvBRatio", "Win"])
        events_out.writeheader()
        #debugging
        if args.d:
            print("debugging!")
            i = startidx
            for filename in files[startidx:endidx]:
                print("file #: ", i, "file name: ", filename)
                i += 1
                fields = generateFields(filename)
                if args.w:
                    filename = "gggreplays_{}.SC2Replay".format(fields[0])
                    valid_games.append(filename)
                if fields:
                    events_out.writerow({"GameID": fields[0], "ScoutingFrequency": fields[1],
                                        "ScoutingTime": fields[2], "APM": fields[3], "Rank": fields[4],
                                        "Relative Rank": fields[5], "CPS": fields[6],
                                        "PvBRatio": fields[7], "Win": fields[8]})
                    events_out.writerow({"GameID": fields[9], "ScoutingFrequency": fields[10],
                                        "ScoutingTime": fields[11], "APM": fields[12], "Rank": fields[13],
                                        "Relative Rank": fields[14], "CPS": fields[15],
                                        "PvBRatio": fields[16], "Win": fields[17]})
        #running with multiprocessing
        else:
            pool = Pool(40)
            results = pool.map(generateFields, files)
            pool.close()
            pool.join()
            for fields in results:
                if fields:
                    if args.w:
                        filename = "gggreplays_{}.SC2Replay".format(fields[0])
                        valid_games.append(filename)
                    events_out.writerow({"GameID": fields[0], "ScoutingFrequency": fields[1],
                                        "ScoutingTime": fields[2], "APM": fields[3], "Rank": fields[4],
                                        "Relative Rank": fields[5], "CPS": fields[6],
                                        "PvBRatio": fields[7], "Win": fields[8]})
                    events_out.writerow({"GameID": fields[9], "ScoutingFrequency": fields[10],
                                        "ScoutingTime": fields[11], "APM": fields[12], "Rank": fields[13],
                                        "Relative Rank": fields[14], "CPS": fields[15],
                                        "PvBRatio": fields[16], "Win": fields[17]})
    #writing to a new file if it doesn't already exist and command line argument exists
    if args.w and not(os.path.exists("valid_game_ids.txt")):
        with open("valid_game_ids.txt", 'w') as file:
            for game_id in valid_games:
                file.write(game_id + "\n")
    print("Run time: ", (time.time()-t1)/60)
