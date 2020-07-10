#Refined version of scouting_stats.py with multiprocessing

import sc2reader
import csv
import os
import sys
import scouting_detector
from multiprocessing import Pool
import argparse
import time

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
        if winner == 1:
            fields = (game_id, team1_nums, team1_fraction, team1_apm, team1_rank, team1_rel, 1,
                        game_id, team2_nums, team2_fraction, team2_apm, team2_rank, team2_rel, 0)
        elif winner == 2:
            fields = (game_id, team1_nums, team1_fraction, team1_apm, team1_rank, team1_rel, 0,
                        game_id, team2_nums, team2_fraction, team2_apm, team2_rank, team2_rel, 1)
        return fields
    except:
        return

def ranking_stats(replay):
    p1_rank = replay.players[0].highest_league
    p2_rank = replay.players[1].highest_league
    p1_rel = p1_rank - p2_rank
    p2_rel = p2_rank - p1_rank
    return p1_rank, p1_rel, p2_rank, p2_rel

if __name__ == "__main__":
    #command line argument for debugging
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', nargs=2, type=int) #debugging, input indeces
    parser.add_argument('-w', action='store_true')
    parser.add_argument('-r', action='store_true')
    args = parser.parse_args()

    if args.d:
        startidx = args.d[0]
        endidx = args.d[1]

    if args.w:
        files = os.listdir("/tmp/replays")
    elif args.r:
        files = []
        games = open("valid_game_ids.txt", 'r')
        for line in games:
            files.append(line.strip())
        games.close()

    valid_games = []
    t1 = time.time()
    with open("scouting_stats.csv", 'w', newline = '') as fp:
        events_out = csv.DictWriter(fp, fieldnames=["GameID", "ScoutingFrequency", "ScoutingTime",
                                                    "APM", "Rank", "Relative Rank", "Win"])
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
                    events_out.writerow({"GameID": fields[0], "ScoutingFrequency": fields[1],
                                        "ScoutingTime": fields[2], "APM": fields[3], "Rank": fields[4],
                                        "Relative Rank": fields[5], "Win": fields[6]})
                    events_out.writerow({"GameID": fields[7], "ScoutingFrequency": fields[8],
                                        "ScoutingTime": fields[9], "APM": fields[10], "Rank": fields[11],
                                        "Relative Rank": fields[12], "Win": fields[13]})
        #running with multiprocessing            
        else:
            pool = Pool(40)
            results = pool.map(generateFields, files)
            pool.close()
            pool.join()
            for fields in results:
                if fields:
                    filename = "gggreplays_{}.SC2Replay".format(fields[0])
                    valid_games.append(filename)
                    events_out.writerow({"GameID": fields[0], "ScoutingFrequency": fields[1],
                                        "ScoutingTime": fields[2], "APM": fields[3], "Rank": fields[4],
                                        "Relative Rank": fields[5], "Win": fields[6]})
                    events_out.writerow({"GameID": fields[7], "ScoutingFrequency": fields[8],
                                        "ScoutingTime": fields[9], "APM": fields[10], "Rank": fields[11],
                                        "Relative Rank": fields[12], "Win": fields[13]})
    if args.w:
        with open("valid_game_ids.txt", 'w') as file:
            for game_id in valid_games:
                file.write(game_id + "\n")
    print("Run time: ", (time.time()-t1)/60)
