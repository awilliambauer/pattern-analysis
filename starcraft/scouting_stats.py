#Refined version of scouting_stats.py with multiprocessing

import sc2reader
import csv
import os
import sys
import scouting_detector
from multiprocessing import Pool
import argparse

def generateFields(filename):
    game_id = filename.split("_")[1].split(".")[0]
    pathname = "replays/" + filename
    try:
        team1_nums, team1_fraction, team2_nums, team2_fraction, winner = scouting_detector.detect_scouting(pathname)
        if winner == 1:
            fields = (game_id, team1_nums, team1_fraction, 1, game_id, team2_nums, team2_fraction, 0)
        elif winner == 2:
            fields = (game_id, team1_nums, team1_fraction, 0, game_id, team2_nums, team2_fraction, 1)
        return fields
    except RuntimeError:
        return

if __name__ == "__main__":
    #command line argument for debugging
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', action='store_true')
    parser.add_argument('index', nargs=2, type=int)
    args = parser.parse_args()
    startidx = args.index[0]
    endidx = args.index[1]

    files = os.listdir("replays")

    with open("scouting_stats.csv", 'w', newline = '') as fp:
        events_out = csv.DictWriter(fp, fieldnames=["GameID", "ScoutingFrequency",
                                                    "ScoutingTime", "Win"])
        events_out.writeheader()
        #debugging
        if args.d:
            print("debugging!")
            for filename in files[startidx:endidx]:
                game_id = filename.split("_")[1].split(".")[0]
                pathname = "replays/" + filename
                try:
                    team1_nums, team1_fraction, team2_nums, team2_fraction, winner = scouting_detector.detect_scouting(pathname)
                    if winner == 1:
                        events_out.writerow({"GameID": game_id, "ScoutingFrequency": team1_nums,
                                     "ScoutingTime": team1_fraction, "Win": 1})
                        events_out.writerow({"GameID": game_id, "ScoutingFrequency": team2_nums,
                                     "ScoutingTime": team2_fraction, "Win": 0})
                    elif winner == 2:
                        events_out.writerow({"GameID": game_id, "ScoutingFrequency": team1_nums,
                                     "ScoutingTime": team1_fraction, "Win": 0})
                        events_out.writerow({"GameID": game_id, "ScoutingFrequency": team2_nums,
                                     "ScoutingTime": team2_fraction, "Win": 1})
                except RuntimeError:
                    continue
        else:
            pool = Pool()
            results = pool.map(generateFields, files[startidx:endidx])
            pool.close()
            pool.join()
            for fields in results:
                if fields:
                    events_out.writerow({"GameID": fields[0], "ScoutingFrequency": fields[1],
                                        "ScoutingTime": fields[2], "Win": fields[3]})
                    events_out.writerow({"GameID": fields[4], "ScoutingFrequency": fields[5],
                                        "ScoutingTime": fields[6], "Win": fields[7]})
