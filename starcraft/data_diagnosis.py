#A script to gather summary statistics of StarCraft 2 data

import csv
import matplotlib.pyplot as plt
from itertools import groupby
import sys
sys.path.append("../") # to enable importing from plot_util.py

from plot_util import make_boxplot


def data_summary():
    total_rows = 0

    SF_inv = 0
    ST_inv = 0
    APM_inv = 0
    Rank_inv = 0
    CPS_inv = 0
    PR_inv = 0
    BR_inv = 0

    SF_set = {0}
    ST_set = {0}
    APM_set = {0}
    Rank_set = {0}
    CPS_set = {0}
    PR_set = {0}
    BR_set = {0}

    SF_bxplt_data = [[], [], [], [], [], [], []]
    ST_bxplt_data = [[], [], [], [], [], [], []]
    APM_bxplt_data = [[], [], [], [], [], [], []]
    CPS_bxplt_data = [[], [], [], [], [], [], []]
    PR_bxplt_data = [[], [], [], [], [], [], []]
    BR_bxplt_data = [[], [], [], [], [], [], []]

    categories = ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Master", "Grandmaster"]

    pos_APM_wins = 0
    pos_APM_loss = 0
    neg_APM_wins = 0
    neg_APM_loss = 0

    with open("scouting_stats.csv", 'r') as my_csv:
        reader = csv.DictReader(my_csv)
        i = 2
        valid_rank = True
        for row in reader:
            total_rows += 1
            ScoutingFrequency, ScoutingTime, APM, Rank, CPS, PeaceRate, BattleRate, Win = row["ScoutingFrequency"], row["ScoutingTime"], row["APM"], row["Rank"], row["CPS"], row["PeaceRate"], row["BattleRate"], row["Win"]
            #checking if Rank is valid and setting a flag if it is not
            if Rank == "nan":
                Rank_inv += 1
                Rank_set.add(i)
                valid_rank = False
            else:
                valid_rank = True
                intRank = int(Rank)

            #Checking scouting frequency
            if float(ScoutingFrequency) == 0:
                SF_inv += 1
                SF_set.add(i)
            elif valid_rank == True:
                SF_bxplt_data[intRank-1].append(float(ScoutingFrequency))

            #Checking scouting time
            if float(ScoutingTime) == 0:
                ST_inv += 1
                ST_set.add(i)
            elif valid_rank == True:
                ST_bxplt_data[intRank-1].append(float(ScoutingTime))

            #Checking APM
            if float(APM) == 0:
                APM_inv += 1
                APM_set.add(i)
            elif valid_rank == True:
                APM_bxplt_data[intRank-1].append(float(APM))

            if int(Win) == 1:
                if float(APM) > 0:
                    pos_APM_wins += 1
                elif float(APM) < 0:
                    neg_APM_wins += 1
            elif int(Win) == 0:
                if float(APM) > 0:
                    pos_APM_loss += 1
                elif float(APM) < 0:
                    neg_APM_loss += 1

            #Checking CPS
            if float(CPS) == 0:
                CPS_inv += 1
                CPS_set.add(i)
            elif valid_rank == True:
                CPS_bxplt_data[intRank-1].append(float(CPS))

            #Checking peace rate
            if float(PeaceRate) == 0:
                PR_inv += 1
                PR_set.add(i)
            elif valid_rank == True:
                PR_bxplt_data[intRank-1].append(float(PeaceRate))

            #Checking battle rate
            if float(BattleRate) == 0:
                BR_inv += 1
                BR_set.add(i)
            elif valid_rank == True:
                BR_bxplt_data[intRank-1].append(float(BattleRate))

            i += 1


    make_boxplot(SF_bxplt_data, categories, "Scouting Frequency", "ScoutingFrequencyBoxplot.png")
    make_boxplot(ST_bxplt_data, categories, "Scouting Time", "ScoutingTimeBoxplot.png")
    make_boxplot(APM_bxplt_data, categories, "Relative Actions per Minute", "APMBoxplot.png")
    make_boxplot(CPS_bxplt_data, categories, "Commands per Second", "CPSBoxplot.png")
    make_boxplot(PR_bxplt_data, categories, "Macro Selection Rate during Peace Time", "PeaceRateBoxplot.png")
    make_boxplot(BR_bxplt_data, categories, "Macro Selection Rate during Battle Time", "BattleRateBoxplot.png")

    with open("scouting_stats.csv", 'r') as my_csv:
        reader = csv.DictReader(my_csv)
        rows = [r for r in reader]

    
    make_boxplot([[float(r["APM"]) for r in rows if r["Win"] == "1"], [float(r["APM"]) for r in rows if r["Win"] == "0"]], 
                 ["Win", "Loss"], "Relative APM", "Rel_APM_WL.png", (-1000, 1000))
    
    binned_pos_apms = [a for a in sorted({int(float(r["APM"])) for r in rows}) if a > 0]
    apm_to_rows = {apm: list(rs) for apm, rs in groupby(sorted(rows, key=lambda r: int(float(r["APM"]))), lambda r: int(float(r["APM"])))}
    
    fig, ax = plt.subplots(figsize=(10,5))
    # plot those apm values where we have at least 10 data points to reduce noise
    ax.plot([apm for apm in binned_pos_apms if len(apm_to_rows[apm]) > 10], 
            [len([r for r in apm_to_rows[apm] if r["Win"] == "1"]) / len(apm_to_rows[apm]) * 100 for apm in binned_pos_apms if len(apm_to_rows[apm]) > 10])
    plt.xlabel("Relative APM")
    plt.ylabel("Win Percentage")
    fig.tight_layout()
    fig.savefig("Rel_APM_vs_Win_Rate.png")

    SF_perc = int((SF_inv/total_rows)*100)
    ST_perc = int((ST_inv/total_rows)*100)
    APM_perc = int((APM_inv/total_rows)*100)
    Rank_perc = int((Rank_inv/total_rows)*100)
    CPS_perc = int((CPS_inv/total_rows)*100)
    PR_perc = int((PR_inv/total_rows)*100)
    BR_perc = int((BR_inv/total_rows)*100)

    all_inv_idx = SF_set.union(ST_set, APM_set, Rank_set, CPS_set, PR_set, BR_set)
    all_perc = int((len(all_inv_idx)-1/total_rows)*100)

    print("Total rows/datapoints: ", total_rows)

    print("Number of invalid Scouting Frequencies:", SF_inv, ", or {:2d}% of the data".format(SF_perc))
    print("Number of invalid Scouting Times:", ST_inv, ", or {:2d}% of the data".format(ST_perc))
    print("Number of invalid APMs:", APM_inv, ", or {:2d}% of the data".format(APM_perc))
    print("Number of invalid Ranks:", Rank_inv, ", or {:2d}% of the data".format(Rank_perc))
    print("Number of invalid CPS's:", CPS_inv, ", or {:2d}% of the data".format(CPS_perc))
    print("Number of invalid Peace Rates:", PR_inv, ", or {:2d}% of the data".format(PR_perc))
    print("Number of invalid Battle Rates:", BR_inv, ", or {:2d}% of the data".format(BR_perc))

    print("Total number of invalid datapoints:", len(all_inv_idx)-1, ", or {:2d}% of the data".format(all_perc))

    print("Positive APM wins:", pos_APM_wins, ", and losses:", pos_APM_loss)
    print("Negative APM wins:", neg_APM_wins, ", and losses:", neg_APM_loss)

data_summary()
