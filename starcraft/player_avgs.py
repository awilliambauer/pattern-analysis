# A script to read in a csv with data from StarCraft 2 and sort players
# into novice, proficient, and expert categories as well as average all
# available statistics for each player
# Alison Cameron
# July 2020

import csv
import time
from collections import defaultdict

def readCSV():
    '''readCSV reads scouting_stats.csv and returns a dictionary where the keys
    are each player's unique user ID, and the values are lists containing
    all of their stats held in scouting_stats.csv.'''
    uid_dict = defaultdict()
    # for each uid in uid_dict, there is a list of 6 lists
    # list 0: rank, list 1: scouting frequency,
    # list 2: aps, list 3: cps,
    # list 4: peace rate, list 5: battle rate
    with open("scouting_stats.csv", 'r') as my_csv:
        reader = csv.DictReader(my_csv)
        for row in reader:
            uid, rank, scoutingF, aps = row["UID"], row["Rank"], float(row["ScoutingFrequency"]), float(row["APS"])
            cps, peaceRate, battleRate = float(row["CPS"]), float(row["PeaceRate"]), float(row["BattleRate"])

            if rank == "nan":
                continue

            if not (uid in uid_dict.keys()):
                uid_dict[uid] = [[], [], [], [], [], []]

            uid_dict[uid][0].append(int(rank))
            uid_dict[uid][1].append(scoutingF)
            uid_dict[uid][2].append(aps)
            uid_dict[uid][3].append(cps)
            uid_dict[uid][4].append(peaceRate)
            uid_dict[uid][5].append(battleRate)

    return uid_dict

def writeToCsv(uid_dict):
    '''writeToCsv takes in the user id dictionary returned by readCSV,
    averages all statistics for each player, and writes their averages
    to a new csv.'''
    with open("player_avgs.csv", 'w', newline = '') as my_csv:
        avgs_out = csv.DictWriter(my_csv, fieldnames = ["UID", "Expertise",
                                    "ScoutingFrequency", "APS", "CPS",
                                    "PeaceRate", "BattleRate"])
        avgs_out.writeheader()
        for uid in uid_dict.keys():
            avgs = []
            stats_lists = uid_dict[uid]
            for list in stats_lists:
                total = 0
                num = len(list)
                for i in range(num):
                    total += list[i]
                avg = total/num
                avgs.append(avg)
            rank = int(avgs[0])
            if rank <= 3:
                expertise = 1
            elif rank <= 5:
                expertise = 2
            elif rank <= 7:
                expertise = 3
            avgs_out.writerow({"UID": uid, "Expertise": expertise,
                                "ScoutingFrequency": avgs[1], "APS": avgs[2],
                                "CPS": avgs[3], "PeaceRate": avgs[4], "BattleRate": avgs[5]})


if __name__ == "__main__":
    t1 = time.time()
    uid_dict = readCSV()
    writeToCsv(uid_dict)
    deltatime = time.time()-t1
    print("Run time: ", "{:2d}".format(int(deltatime//60)), "minutes and", "{:05.2f}".format(deltatime%60), "seconds")
