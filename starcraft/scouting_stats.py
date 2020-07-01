#Refined version of scouting_stats.py with multiprocessing

import sc2reader
import csv
import os
import sys
import scouting_detector
from multiprocessing import Pool

def generateFields(filename):
    game_id = filename.split("_")[1].split(".")[0]
    pathname = "replays/" + filename
    try:
        team1_nums, team1_fraction, team2_nums, team2_fraction, winner = scouting_detector.detect_scouting(pathname)
        fields = (game_id, team1_nums, team1_fraction, team2_nums, team2_fraction, winner)
        return fields
    except RuntimeError:
        return

if __name__ == "__main__":
    files = os.listdir("replays")
    pool = Pool()
    results = pool.imap_unordered(generateFields, files)
    pool.close()
    pool.join()
    with open("scouting_stats2.csv", 'w', newline = '') as fp:
        events_out = csv.DictWriter(fp, fieldnames=["Game ID", "Team 1 Scouting Frequency", "Team 1 Scouting Time",
                                                    "Team 2 Scouting Frequency", "Team 2 Scouting Time", "Winner (team #)"])
        events_out.writeheader()
        for fields in results:
            events_out.writerow({"Game ID": fields[0], "Team 1 Scouting Frequency": fields[1], "Team 1 Scouting Time": fields[2],
                                "Team 2 Scouting Frequency": fields[3], "Team 2 Scouting Time": fields[4],
                                "Winner (team #)": fields[5]})
