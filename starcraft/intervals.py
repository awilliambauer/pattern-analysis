# Intended to be used for data visualization of time in between periods
# of scouting for players in StarCraft 2

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

sc2reader.engine.register_plugin(APMTracker())
sc2reader.engine.register_plugin(SelectionTracker())
sc2reader.engine.register_plugin(ActiveSelection())

def generateFields(filename):
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

        fields = (game_id, team1_rank, team1_avg_int, team2_rank, team2_avg_int)
        return fields
    except:
        return

def writeToCsv():
    files = []
    games = open("valid_game_ids.txt", 'r')
    for line in games:
        files.append(line.strip())
    games.close()

    pool = Pool(20)
    results = pool.map(generateFields, files)
    pool.close()
    pool.join()

    with open("intervals.csv", 'w', newline='') as my_csv:
        events_out = csv.DictWriter(my_csv, fieldnames=["GameID", "Rank", "Interval"])
        events_out.writeheader()
        for fields in results:
            if fields:
                game_id = fields[0]
                team1_rank, team1_avg_int, team2_rank, team2_avg_int = fields[1], fields[2], fields[3], fields[4]
                if (team1_avg_int != -1) and not(math.isnan(team1_rank)):
                    events_out.writerow({"GameID": game_id, "Rank": team1_rank, "Interval": team1_avg_int})
                if (team2_avg_int != -1) and not(math.isnan(team2_rank)):
                    events_out.writerow({"GameID": game_id, "Rank": team2_rank, "Interval": team2_avg_int})



if __name__ == "__main__":
    t1 = time.time()
    writeToCsv()
    deltatime = time.time()-t1
    print("Run time: ", "{:2d}".format(int(deltatime//60)), "minutes and", "{:05.2f}".format(deltatime%60), "seconds")
