# used for generating the replays_info.csv which is in turn used to group replays by their map
# replays_info.csv stores several useful bits of information about each replay: rank, race, map, player id, etc.
# requires there to be a valid replay filename list in file_locations.VALID_REPLAY_FILENAMES_FILE
# ZL June 2021

import os
import sc2reader
from multiprocessing import Pool, cpu_count
import csv
import traceback
import random
import file_locations
from replay_verification import map_pretty_name_to_file
from load_map_path_data import get_all_possible_names
import csv


def get_map_name_groups():
    """
    This function is used for getting around the issue of localized map names.
    :return: a list of groups of map names. Each group consists of only map names that have been associated with
    each other through their hashes. Thus, barring strange cases of misnamed maps, each group should only consist
    of one map name but in several different languages.
    """
    map_names = set()
    with open(file_locations.REPLAY_INFO_FILE, "r") as replays_info:
        reader = csv.DictReader(replays_info)
        for row in reader:
            map_names.add(row["Map"])
    map_names_groups = [(name, get_all_possible_names(name)) for name in map_names]
    map_name_clusters = []
    for map_name, possible_names in map_names_groups:
        matching_cluster = None
        # find a cluster which has an intersection with this possible name group
        for cluster in map_name_clusters:
            if any(filter(lambda x: x in cluster, possible_names)):
                matching_cluster = cluster
                break
        if matching_cluster is None:
            # if there is no matching cluster start a new one
            map_name_clusters.append(possible_names)
            continue
        # otherwise, coalesce clusters
        for possible_name in possible_names:
            if possible_name not in matching_cluster:
                matching_cluster.append(possible_name)
    return map_name_clusters


def group_replays_by_map(replay_filter=lambda x: True):
    """
    :return: a dictionary of map name groups to replays that took place on that map. See get_map_name_groups for more
    info.
    """
    map_name_groups = get_map_name_groups()
    maps = {}
    with open(file_locations.REPLAY_INFO_FILE, "r") as replays_info:
        reader = csv.DictReader(replays_info)
        rows = random.choices(list(reader), k=10000)
        for row in rows:
            if not replay_filter(row):
                continue
            matching_map_name_group = None
            for map_name_group in map_name_groups:
                if row["Map"] in map_name_group:
                    matching_map_name_group = tuple(map_name_group)
                    break
            if matching_map_name_group is None:
                print("no map name group for replay", row)  # i think this should be impossible...
                continue
            if matching_map_name_group not in maps:
                maps[matching_map_name_group] = []
            maps[matching_map_name_group].append(row["ReplayID"])
    return maps


def _generate_replay_entry(file):
    if not file.endswith("SC2Replay"):
        return None
    try:
        loaded_replay = sc2reader.load_replay(file_locations.REPLAY_FILE_DIRECTORY + "/" + file)
        player_1_info = _generate_player_specific_entry(loaded_replay.players[0], 1)
        player_2_info = _generate_player_specific_entry(loaded_replay.players[1], 2)
        other_info = {"ReplayID": file,
                      "Map": map_pretty_name_to_file(loaded_replay.map_name)}
        other_info.update(player_1_info)
        other_info.update(player_2_info)
        return other_info
    except:
        print("error loading replay")
        traceback.print_exc()
        return None


def _generate_player_specific_entry(player, number):
    return {"UID" + str(number): player.detail_data['bnet']['uid'], "Race" + str(number): player.play_race,
            "Rank" + str(number): player.highest_league}


def generate_replay_info_csv():
    """
    Generates a replay info CSV file for all replays in the valid replay filenames file. This replay info is
    used in grouping replays by map for analysis. It also has several other useful values which can be used to filter
    the replays as needed. The values are: ReplayID (the filename of the replay), Map (the localized map name of the
    replay), and for each player in the replay, their UID, race and rank.
    """
    valid_games = None
    with open(file_locations.VALID_REPLAY_FILENAMES_FILE, "r") as f:
        valid_games = [line[:-1] for line in f.readlines()]
    with open(file_locations.REPLAY_INFO_FILE, 'w', newline='') as fp:
        events_out = csv.DictWriter(fp,
                                    fieldnames=["ReplayID", "Map", "UID1", "UID2", "Race1", "Race2", "Rank1", "Rank2"])
        events_out.writeheader()
        pool = Pool(min(cpu_count(), 60))
        entries = pool.map(_generate_replay_entry,
                           filter(lambda file: file in valid_games, os.listdir(file_locations.REPLAY_FILE_DIRECTORY)))
        for entry in entries:
            if entry is not None:
                events_out.writerow(entry)


def test():
    generate_replay_info_csv()


if __name__ == "__main__":
    test()
