#A CSV of event counts for StarCraft 2 replays

import sc2reader
import csv
from multiprocessing import Pool

def event_counts(replay):
    sets = {1: 0, 2: 0}
    adds = {1: 0, 2: 0}
    gets = {1: 0, 2: 0}
    totals = {1: 0, 2: 0}

    for event in replay.game_events:
        if isinstance(event, sc2reader.events.game.ControlGroupEvent):
            team = event.player.pid
            totals[team] += 1
            if isinstance(event, sc2reader.events.game.SetControlGroupEvent):
                sets[team] += 1
            elif isinstance(event, sc2reader.events.game.AddToControlGroupEvent):
                adds[team] += 1
            elif isinstance(event, sc2reader.events.game.GetControlGroupEvent):
                gets[team] += 1

    team1_set, team2_set = sets[1], sets[2]
    team1_add, team2_add = adds[1], adds[2]
    team1_get, team2_get = gets[1], gets[2]
    team1_all, team2_all = totals[1], totals[2]
    return team1_set, team1_add, team1_get, team1_all, team2_set, team2_add, team2_get, team2_all


def generateFields(filename):
    #loading the replay
    try:
        if filename[-9:] != "SC2Replay":
            raise RuntimeError()

        #pathname = "practice_replays/" + filename
        pathname = "/Accounts/awb/pattern-analysis/starcraft/replays/" + filename
        r = sc2reader.load_replay(pathname)

        game_id = filename.split("_")[1].split(".")[0]
        if filename.startswith("ggg"):
            game_id = "ggg-" + game_id
        elif filename.startswith("spawningtool"):
            game_id = "st-" + game_id

        team1_uid = r.players[0].detail_data['bnet']['uid']
        team2_uid = r.players[1].detail_data['bnet']['uid']

        team1_rank = r.players[0].highest_league
        team2_rank = r.players[1].highest_league

        team1_set, team1_add, team1_get, team1_all, team2_set, team2_add, team2_get, team2_all = event_counts(r)

        if r.winner.number == 1:
            fields = (game_id, team1_uid, team1_rank, team1_set, team1_add, team1_get, team1_all, 1,
                    game_id, team2_uid, team2_rank, team2_set, team2_add, team2_get, team2_all, 0)
        elif r.winner.number == 2:
            fields = (game_id, team1_uid, team1_rank, team1_set, team1_add, team1_get, team1_all, 0,
                    game_id, team2_uid, team2_rank, team2_set, team2_add, team2_get, team2_all, 1)

        return fields
    except:
        return

def writeToCsv():
    files = []
    games = open("valid_game_ids.txt", 'r')
    for line in games:
        files.append(line.strip())
    games.close()

    with open("event_counts.csv", 'w', newline = '') as file:
        events_out = csv.DictWriter(file, fieldnames=["GameID", "UID", "Rank",
                                    "SetCGCount", "AddCGCount", "GetCGCount",
                                    "TotalCount", "Win"])
        events_out.writeheader()
        pool = Pool(40)
        results = pool.map(generateFields, files)
        pool.close()
        pool.join()
        for fields in results:
            if fields:
                events_out.writerow({"GameID": fields[0], "UID": fields[1], "Rank": fields[2],
                                    "SetCGCount": fields[3], "AddCGCount": fields[4],
                                    "GetCGCount": fields[5], "TotalCount": fields[6],
                                    "Win": fields[7]})
                events_out.writerow({"GameID": fields[8], "UID": fields[9], "Rank": fields[10],
                                    "SetCGCount": fields[11], "AddCGCount": fields[12],
                                    "GetCGCount": fields[13], "TotalCount": fields[14],
                                    "Win": fields[15]})

if __name__ == "__main__":
    writeToCsv()
