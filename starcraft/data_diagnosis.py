# A script to gather summary statistics of StarCraft 2 data
# Alison Cameron
# July 2020

import csv
import time
from collections import Counter
import matplotlib.pyplot as plt
from itertools import groupby
import sys
sys.path.append("../") # to enable importing from plot_util.py

from plot_util import make_boxplot

def read_scouting_stats():
    '''read scouting_stats creates boxplots by rank and by loss/win,
    and calculates statistics for each value in scouting_stats.py It returns
    a dictionary of the number of unique user ids by rank as well.'''
    rank_uid_counter = {1: Counter(), 2: Counter(), 3: Counter(), 4: Counter(),
                        5: Counter(), 6: Counter(), 7: Counter()}
    total_rows = 0

    SF_inv = 0
    ST_inv = 0
    APS_inv = 0
    Rank_inv = 0
    CPS_inv = 0
    PR_inv = 0
    BR_inv = 0

    SF_set = {0}
    ST_set = {0}
    APS_set = {0}
    Rank_set = {0}
    CPS_set = {0}
    PR_set = {0}
    BR_set = {0}

    SF_rank_data = [[], [], [], [], [], [], []]
    APS_rank_data = [[], [], [], [], [], [], []]
    CPS_rank_data = [[], [], [], [], [], [], []]
    PR_rank_data = [[], [], [], [], [], [], []]
    BR_rank_data = [[], [], [], [], [], [], []]

    rank_categories = ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Master", "Grandmaster"]

    SF_win_data = [[], []]
    APS_win_data = [[], []]
    CPS_win_data = [[], []]
    PR_win_data = [[], []]
    BR_win_data = [[], []]
    win_categories = ["Loss", "Win"]

    pos_APS_wins = 0
    pos_APS_loss = 0
    neg_APS_wins = 0
    neg_APS_loss = 0

    with open("scouting_stats.csv", 'r') as my_csv:
        reader = csv.DictReader(my_csv)
        i = 2
        valid_rank = True
        for row in reader:
            total_rows += 1

            uid, ScoutingFrequency, APS, Rank, CPS, PeaceRate, BattleRate, Win = row["UID"], row["ScoutingFrequency"], row["APS"], row["Rank"], row["CPS"], row["PeaceRate"], row["BattleRate"], row["Win"]

            # checking if Rank is valid and setting a flag if it is not
            if Rank == "nan":
                Rank_inv += 1
                Rank_set.add(i)
                valid_rank = False
            else:
                valid_rank = True
                intRank = int(Rank)
                intWin = int(Win)
                rank_uid_counter[intRank][uid] += 1

            # Checking scouting frequency
            if float(ScoutingFrequency) == 0:
                SF_inv += 1
                SF_set.add(i)
            elif valid_rank == True:
                SF_rank_data[intRank-1].append(float(ScoutingFrequency))
                SF_win_data[intWin].append(float(ScoutingFrequency))

            # Checking APS
            if float(APS) == 0:
                APS_inv += 1
                APS_set.add(i)
            elif valid_rank == True:
                APS_rank_data[intRank-1].append(float(APS))
                APS_win_data[intWin].append(float(APS))

            if int(Win) == 1:
                if float(APS) > 0:
                    pos_APS_wins += 1
                elif float(APS) < 0:
                    neg_APS_wins += 1
            elif int(Win) == 0:
                if float(APS) > 0:
                    pos_APS_loss += 1
                elif float(APS) < 0:
                    neg_APS_loss += 1

            # Checking CPS
            if float(CPS) == 0:
                CPS_inv += 1
                CPS_set.add(i)
            elif valid_rank == True:
                CPS_rank_data[intRank-1].append(float(CPS))
                CPS_win_data[intWin].append(float(CPS))

            # Checking peace rate
            if float(PeaceRate) == 0:
                PR_inv += 1
                PR_set.add(i)
            elif valid_rank == True:
                PR_rank_data[intRank-1].append(float(PeaceRate))
                PR_win_data[intWin].append(float(PeaceRate))

            # Checking battle rate
            if float(BattleRate) == 0:
                BR_inv += 1
                BR_set.add(i)
            elif valid_rank == True:
                BR_rank_data[intRank-1].append(float(BattleRate))
                BR_win_data[intWin].append(float(BattleRate))

            i += 1


    make_boxplot(SF_rank_data, rank_categories, "Scouting Frequency", "ScoutingFrequencyByRank.png")
    make_boxplot(APS_rank_data, rank_categories, "Actions per Second", "APSByRank.png", ylims=(0, 20))
    make_boxplot(CPS_rank_data, rank_categories, "Commands per Second", "CPSByRank.png")
    make_boxplot(PR_rank_data, rank_categories, "Macro Selection Rate during Peace Time", "PeaceRateByRank.png")
    make_boxplot(BR_rank_data, rank_categories, "Macro Selection Rate during Battle Time", "BattleRateByRank.png")

    make_boxplot(SF_win_data, win_categories, "Scouting Frequency", "ScoutingFrequencyByWin.png")
    make_boxplot(APS_win_data, win_categories, "Actions per Second", "APSByWin.png", ylims=(0, 20))
    make_boxplot(CPS_win_data, win_categories, "Commands per Second", "CPSByWin.png")
    make_boxplot(PR_win_data, win_categories, "Macro Selection Rate during Peace Time", "PeaceRateByWin.png")
    make_boxplot(BR_win_data, win_categories, "Macro Selection Rate during Battle Time", "BattleRateByWin.png")

    with open("scouting_stats.csv", 'r') as my_csv:
        reader = csv.DictReader(my_csv)
        rows = [r for r in reader]


    make_boxplot([[float(r["APS"]) for r in rows if r["Win"] == "1"], [float(r["APS"]) for r in rows if r["Win"] == "0"]],
                 ["Win", "Loss"], "Relative APS", "Rel_APS_WL.png", (-1000, 1000))

    binned_pos_apms = [a for a in sorted({int(float(r["APS"])) for r in rows}) if a > 0]
    apm_to_rows = {apm: list(rs) for apm, rs in groupby(sorted(rows, key=lambda r: int(float(r["APS"]))), lambda r: int(float(r["APS"])))}

    fig, ax = plt.subplots(figsize=(10,5))
    # plot those apm values where we have at least 10 data points to reduce noise
    ax.plot([apm for apm in binned_pos_apms if len(apm_to_rows[apm]) > 10],
            [len([r for r in apm_to_rows[apm] if r["Win"] == "1"]) / len(apm_to_rows[apm]) * 100 for apm in binned_pos_apms if len(apm_to_rows[apm]) > 10])
    plt.xlabel("Relative APS")
    plt.ylabel("Win Percentage")
    fig.tight_layout()
    fig.savefig("Rel_APS_vs_Win_Rate.png")

    SF_perc = (SF_inv/total_rows)
    ST_perc = (ST_inv/total_rows)
    APS_perc = (APS_inv/total_rows)
    Rank_perc = (Rank_inv/total_rows)
    CPS_perc = (CPS_inv/total_rows)
    PR_perc = (PR_inv/total_rows)
    BR_perc = (BR_inv/total_rows)

    all_inv_idx = SF_set.union(ST_set, APS_set, Rank_set, CPS_set, PR_set, BR_set)
    all_perc = (len(all_inv_idx)-1)/total_rows

    print("Total rows/datapoints: ", total_rows)

    print(f"Number of invalid Scouting Frequencies: {SF_inv}, or {SF_perc:.2%} of the data")
    print(f"Number of invalid Scouting Times: {ST_inv}, or {ST_perc:.2%} of the data")
    print(f"Number of invalid APSs: {APS_inv}, or {APS_perc:.2%} of the data")
    print(f"Number of invalid Ranks: {Rank_inv}, or {Rank_perc:.2%} of the data")
    print(f"Number of invalid CPS's: {CPS_inv}, or {CPS_perc:.2%} of the data")
    print(f"Number of invalid Peace Rates: {PR_inv}, or {PR_perc:.2%} of the data")
    print(f"Number of invalid Battle Rates: {BR_inv}, or {BR_perc:.2%} of the data")

    print("Total number of invalid datapoints:", len(all_inv_idx)-1, ", or {:.2%} of the data".format(all_perc))

    print("Positive APS wins:", pos_APS_wins, ", and losses:", pos_APS_loss)
    print("Negative APS wins:", neg_APS_wins, ", and losses:", neg_APS_loss)

    return rank_uid_counter

def read_event_counts():
    '''Creates boxplots for control group selections for gold and grandmaster ranks,
        as well as boxplots for control group selection ratio for each rank'''
    total_rows = 0

    gold_bxplt_data = [[], []]
    gmaster_bxplt_data = [[], []]
    rank_cg_categories = ["Set and Add CG Rate", "Get CG Rate"]

    ratio_bxplt_data = [[], [], [], [], [], [], []]
    rank_categories = ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Master", "Grandmaster"]

    with open("event_counts.csv", 'r') as my_csv:
        reader = csv.DictReader(my_csv)
        for row in reader:
            rank, set_ct, add_ct, get_ct, total_ct, win = row["Rank"], float(row["SetCGCount"]), float(row["AddCGCount"]), float(row["GetCGCount"]), float(row["TotalCount"]), row["Win"]
            if total_ct == 0 or get_ct == 0:
                continue

            ratio = (set_ct + add_ct)/get_ct

            if rank == "nan":
                valid_rank = False
            else:
                valid_rank = True
                intRank = int(rank)

            if valid_rank:
                ratio_bxplt_data[intRank-1].append(ratio)

            if valid_rank and intRank == 3:
                gold_bxplt_data[0].append(((set_ct + add_ct)/total_ct))
                gold_bxplt_data[1].append((get_ct/total_ct))

            elif valid_rank and intRank == 7:
                gmaster_bxplt_data[0].append(((set_ct + add_ct)/total_ct))
                gmaster_bxplt_data[1].append((get_ct/total_ct))

    make_boxplot(gold_bxplt_data, rank_cg_categories, "Gold League Control Group Selection", "GoldCGSelection.png")
    make_boxplot(gmaster_bxplt_data, rank_cg_categories, "Grandmaster League Control Group Selection", "GrandmasterCGSelection.png")
    make_boxplot(ratio_bxplt_data, rank_categories, "Ratio of Set and Add Counts to Get Counts", "CGRatioByRank.png")

def uid_stats(counter):
    '''Collects and prints statistics about unique player ids,
    as well as information about how many players had only one game or
    multiple games.'''
    num_unique = len(counter)
    keys = counter.keys()
    total_games = 0
    one_game = 0
    mult_games = 0

    for uid in keys:
        games = counter[uid]
        total_games += games
        if games == 1:
            one_game += 1
        elif games > 1:
            mult_games += 1

    if num_unique != 0:
        avg_games = total_games/num_unique
        one_perc = int((one_game/num_unique)*100)
        mult_perc = int((mult_games/num_unique)*100)

        print("Number of unique uid's:", num_unique)
        print("Average number of games per player:", avg_games)
        print("Number of players with only one game:", one_game, "or {:2d}% of players".format(one_perc))
        print("Number of players with multiple games:", mult_games, "or {:2d}% of players".format(mult_perc))

def match_counter(filename):
    '''Collects and prints statistics about the number of matches
    at each rank'''
    unique_ids = [set(), set(), set(), set(), set(), set(), set()]
    with open(filename, 'r') as my_csv:
        reader = csv.DictReader(my_csv)
        for row in reader:
            game_id, uid, rank = row["GameID"], row["UID"], row["Rank"]
            if rank == "nan":
                continue
            else:
                intRank = int(rank)

            game_and_user = game_id + uid
            unique_ids[intRank-1].add(game_and_user)
    print("---{}---".format(filename))
    print("Bronze unique ids:", len(unique_ids[0]))
    print("Silver unique ids:", len(unique_ids[1]))
    print("Gold unique ids:", len(unique_ids[2]))
    print("Platinum unique ids:", len(unique_ids[3]))
    print("Diamond unique ids:", len(unique_ids[4]))
    print("Master unique ids:", len(unique_ids[5]))
    print("Grandmaster unique ids:", len(unique_ids[6]))

def read_intervals():
    '''Reads statistics and creates boxplots for information about the average
    length in between periods of scouting'''
    interval_data = [[], [], [], [], [], [], []]
    rank_categories = ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Master", "Grandmaster"]
    with open("intervals.csv", 'r') as my_csv:
        reader = csv.DictReader(my_csv)
        for row in reader:
            rank, interval = row["Rank"], float(row["Interval"])
            if rank == "nan":
                continue
            else:
                intRank = int(rank)
            interval_data[intRank-1].append(interval)

    make_boxplot(interval_data, rank_categories, "Average Length (in seconds) Between Scouting Periods", "Intervals.png", yscale = "log")
    make_boxplot([interval_data[6]], ["Grandmaster"], "Average Length (in seconds) Between Scouting Periods", "IntervalsGrandmaster.png")

def main1():
    '''Collects and prints statistics about the number of players
    at each rank'''
    t1 = time.time()
    uid_counter = read_scouting_stats()
    read_event_counts()
    print("---Bronze---")
    uid_stats(uid_counter[1])
    print("---Silver---")
    uid_stats(uid_counter[2])
    print("---Gold---")
    uid_stats(uid_counter[3])
    print("---Platinum---")
    uid_stats(uid_counter[4])
    print("--Diamond---")
    uid_stats(uid_counter[5])
    print("---Master---")
    uid_stats(uid_counter[6])
    print("---Grandmaster---")
    uid_stats(uid_counter[7])
    deltatime = time.time()-t1
    print("Run time: ", "{:2d}".format(int(deltatime//60)), "minutes and", "{:05.2f}".format(deltatime%60), "seconds")

def main2():
    '''Counts the number of matches for scouting_time_fraction.csv
    and scouting_time_frames1.csv'''
    match_counter("scouting_time_fraction.csv")
    match_counter("scouting_time_frames1.csv")

def scouting_analysis():
    '''scouting_analysis reads and prints information to be used by ScoutingAnalysis.Rmd
    to create histograms. It calculates the percentage of players that exhibit a
    certain trait for each rank and level of expertise.'''
    matches_rank = [0, 0, 0, 0, 0, 0, 0]
    stats_dict_rank = {"initial": [0, 0, 0, 0, 0, 0, 0], "base": [0, 0, 0, 0, 0, 0, 0],
                    "new": [0, 0, 0, 0, 0, 0, 0], "between": [0, 0, 0, 0, 0, 0, 0]}
    matches_expertise = [0, 0, 0]
    stats_dict_expertise = {"initial": [0, 0, 0], "base": [0, 0, 0],
                    "new": [0, 0, 0], "between": [0, 0, 0]}
    with open("scouting_analysis.csv") as my_csv:
        reader = csv.DictReader(my_csv)
        for row in reader:
            rank, initial, base, new, between = row["Rank"], int(row["InitialScouting"]), int(row["BaseScouting"]), int(row["NewAreas"]), int(row["BetweenBattles"])
            if rank == "nan":
                continue
            else:
                intRank = int(rank)
                matches_rank[intRank-1] += 1
                if intRank == 1 or intRank == 2:
                    expertise = 0
                elif intRank == 4 or intRank == 5:
                    expertise = 1
                elif intRank == 7:
                    expertise = 2
                matches_expertise[expertise] += 1

            if initial == 1:
                stats_dict_rank["initial"][intRank-1] += 1
                stats_dict_expertise["initial"][expertise] +=1

            if base == 0:
                stats_dict_rank["base"][intRank-1] += 1
                stats_dict_expertise["base"][expertise] +=1

            if new == 1:
                stats_dict_rank["new"][intRank-1] += 1
                stats_dict_expertise["new"][expertise] +=1

            if between == 1:
                stats_dict_rank["between"][intRank-1] += 1
                stats_dict_expertise["between"][expertise] +=1

    stats = stats_dict_rank.keys()
    for stat in stats:
        for i in range(7):
            stats_dict_rank[stat][i] = stats_dict_rank[stat][i]/matches_rank[i]
        for i in range(3):
            stats_dict_expertise[stat][i] = stats_dict_expertise[stat][i]/matches_expertise[i]

    print("RANK")
    print("---Initial Scouting---")
    print(stats_dict_rank["initial"])
    print("---Scouting Expansions---")
    print(stats_dict_rank["base"])
    print("---Scouting New Areas---")
    print(stats_dict_rank["new"])
    print("---Scouting Between Battles")
    print(stats_dict_rank["between"])
    print("\n\nEXPERTISE")
    print("---Initial Scouting---")
    print(stats_dict_expertise["initial"])
    print("---Scouting Expansions---")
    print(stats_dict_expertise["base"])
    print("---Scouting New Areas---")
    print(stats_dict_expertise["new"])
    print("---Scouting Between Battles")
    print(stats_dict_expertise["between"])
