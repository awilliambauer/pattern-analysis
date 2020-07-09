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
    pathname = "/tmp/replays/" + filename
    try:
        team1_nums, team1_fraction, team1_apm, team2_nums, team2_fraction, team2_apm, winner = scouting_detector.detect_scouting(pathname)
        if winner == 1:
            fields = (game_id, team1_nums, team1_fraction, team1_apm, 1, game_id, team2_nums, team2_fraction, team2_apm, 0)
        elif winner == 2:
            fields = (game_id, team1_nums, team1_fraction, team1_apm, 0, game_id, team2_nums, team2_fraction, team2_apm, 1)
        return fields
    except RuntimeError:
        return

if __name__ == "__main__":
    #command line argument for debugging
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', action='store_true')
    #parser.add_argument('--index', nargs=2, type=int)
    args = parser.parse_args()
    #startidx = args.index[0]
    #endidx = args.index[1]

    files = os.listdir("/tmp/replays")
    with open("scouting_stats.csv", 'w', newline = '') as fp:
        events_out = csv.DictWriter(fp, fieldnames=["GameID", "ScoutingFrequency",
                                                    "ScoutingTime", "APM", "Win"])
        events_out.writeheader()
        #debugging
        if args.d:
            print("debugging!")
            i = startidx
            for filename in files:
                print("file #: ", i, "file name: ", filename)
                game_id = filename.split("_")[1].split(".")[0]
                pathname = "/tmp/replays/" + filename
                i += 1
                try:
                    team1_nums, team1_fraction, team1_apm, team2_nums, team2_fraction, team2_apm, winner = scouting_detector.detect_scouting(pathname)
                    if winner == 1:
                        events_out.writerow({"GameID": game_id, "ScoutingFrequency": team1_nums,
                                     "ScoutingTime": team1_fraction, "APM": team1_apm, "Win": 1})
                        events_out.writerow({"GameID": game_id, "ScoutingFrequency": team2_nums,
                                     "ScoutingTime": team2_fraction, "APM": team2_apm, "Win": 0})
                    elif winner == 2:
                        events_out.writerow({"GameID": game_id, "ScoutingFrequency": team1_nums,
                                     "ScoutingTime": team1_fraction, "APM": team1_apm, "Win": 0})
                        events_out.writerow({"GameID": game_id, "ScoutingFrequency": team2_nums,
                                     "ScoutingTime": team2_fraction, "APM": team2_apm, "Win": 1})
                except RuntimeError:
                    continue
        else:
            pool = Pool(40)
            results = pool.map(generateFields, files)
            pool.close()
            pool.join()
            for fields in results:
                if fields:
                    events_out.writerow({"GameID": fields[0], "ScoutingFrequency": fields[1],
                                        "ScoutingTime": fields[2], "APM": fields[3], "Win": fields[4]})
                    events_out.writerow({"GameID": fields[5], "ScoutingFrequency": fields[6],
                                        "ScoutingTime": fields[7], "APM": fields[8], "Win": fields[9]})
