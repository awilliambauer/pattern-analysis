# Intended to be used for data visualization of when players scout
# during StarCraft 2

import csv
import sc2reader
import time
from multiprocessing import Pool, cpu_count
from collections import Counter
import scouting_detector
import scouting_stats
from sc2reader.engine.plugins import SelectionTracker, APMTracker
from selection_plugin import ActiveSelection
from base_plugins import BaseTracker
import numpy as np
import traceback
from functools import partial
import logging
import sys


def generateFields(filename, which):
    # loading the replay
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

        # loading the replay
        try:
            r = sc2reader.load_replay(pathname)
            if any(v != (0, {}) for v in r.plugin_result.values()):
                print(pathname, r.plugin_result)
        except:
            print(filename, "cannot load using sc2reader due to an internal ValueError")
            raise

        team1_times, team2_times = scouting_detector.scouting_times(r, which)
        team1_rank, team1_rel_rank, team2_rank, team2_rel_rank = scouting_stats.ranking_stats(r)

        fields = [game_id, team1_rank, team1_times, team2_rank, team2_times]
        return fields

    except KeyboardInterrupt:
        raise
    except:
        return


def writeToCsv(which, filename):
    files = []
    games = open("valid_game_ids.txt", 'r')
    for line in games:
        files.append(line.strip())
    games.close()

    pool = Pool(min(cpu_count(), 20))
    results = pool.map(partial(generateFields, which=which), files)
    pool.close()
    pool.join()

    with open(filename, 'w', newline = '') as my_csv:
        events_out = csv.DictWriter(my_csv, fieldnames=["GameID", "Rank", "ScoutTime"])
        events_out.writeheader()
        for fields in results:
            if fields:
                game_id = fields[0]
                rank = fields[1]
                times = fields[2]
                for time in times:
                    events_out.writerow({"GameID": game_id, "Rank": rank, "ScoutTime": time})
                rank = fields[3]
                times = fields[4]
                for time in times:
                    events_out.writerow({"GameID": game_id, "Rank": rank, "ScoutTime": time})


if __name__ == "__main__":
    sc2reader.engine.register_plugin(APMTracker())
    sc2reader.engine.register_plugin(SelectionTracker())
    sc2reader.engine.register_plugin(ActiveSelection())
    bt = BaseTracker()
#     bt.logger.setLevel(logging.ERROR)
#     bt.logger.addHandler(logging.StreamHandler(sys.stdout))
    sc2reader.engine.register_plugin(bt)
#     sc2reader.log_utils.add_log_handler(logging.StreamHandler(sys.stdout), "INFO")

    t1 = time.time()
    writeToCsv(1, "scouting_time_fraction.csv")
    writeToCsv(2, "scouting_time_frames1.csv")
    deltatime = time.time()-t1
    print("Run time: ", "{:2d}".format(int(deltatime//60)), "minutes and", "{:05.2f}".format(deltatime%60), "seconds")
