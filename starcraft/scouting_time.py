#Intended to be used for data visualization of when players scout
#during StarCraft 2

import csv
import sc2reader
import time
from multiprocessing import Pool
import scouting_detector
from sc2reader.engine.plugins import SelectionTracker, APMTracker
from selection_plugin import ActiveSelection

sc2reader.engine.register_plugin(APMTracker())
sc2reader.engine.register_plugin(SelectionTracker())
sc2reader.engine.register_plugin(ActiveSelection())

def generateFields(filename):
    #loading the replay
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

        #loading the replay
        try:
            r = sc2reader.load_replay(pathname)
        except:
            print(filename + " cannot load using sc2reader due to an internal ValueError")
            raise RuntimeError()

        team1_times, team2_times = scouting_detector.scouting_times(r)
        all_times = team1_times + team2_times

        fields = [game_id, all_times]
        return fields

    except:
        return

def writeToCsv():
    files = []
    games = open("valid_game_ids.txt", 'r')
    for line in games:
        files.append(line.strip())
    games.close()

    pool = Pool(40)
    results = pool.map(generateFields, files)
    pool.close()
    pool.join()

    bad_files = 0

    with open("scouting_time.csv", 'w', newline = '') as my_csv:
        events_out = csv.DictWriter(my_csv, fieldnames=["GameID", "ScoutTime"])
        events_out.writeheader()
        for fields in results:
            if fields:
                game_id = fields[0]
                times = fields[1]
                for time in times:
                    events_out.writerow({"GameID": game_id, "ScoutTime": time})
            else:
                bad_files += 1
    print(bad_files)

if __name__ == "__main__":
    t1 = time.time()
    writeToCsv()
    deltatime = time.time()-t1
    print("Run time: ", "{:2d}".format(int(deltatime//60)), "minutes and", "{:05.2f}".format(deltatime%60), "seconds")
