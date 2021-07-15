import os
import sc2reader
from multiprocessing import Pool, cpu_count
import csv
import file_locations
from replay_verification import map_pretty_name_to_file


def generate_replay_entry(file):
    if not file.endswith("SC2Replay"):
        return None
    try:
        loaded_replay = sc2reader.load_replay(file_locations.REPLAY_FILE_DIRECTORY + "/" + file)
        player_1_info = generate_player_specific_entry(loaded_replay.players[0], 1)
        player_2_info = generate_player_specific_entry(loaded_replay.players[1], 2)
        other_info = {"ReplayID": file,
                      "Map": map_pretty_name_to_file(loaded_replay.map_name)}
        other_info.update(player_1_info)
        other_info.update(player_2_info)
        return other_info
    except:
        print("error loading replay")
        return None


def generate_player_specific_entry(player, number):
    return {"UID" + str(number): player.detail_data['bnet']['uid'], "Race" + str(number): player.play_race,
            "Rank" + str(number): player.highest_league}


def generate_replay_info_csv(output_file):
    valid_games = None
    with open("valid_replay_filenames.txt", "r") as f:
        valid_games = [line[:-1] for line in f.readlines()]
    with open(output_file, 'w', newline='') as fp:
        events_out = csv.DictWriter(fp,
                                    fieldnames=["ReplayID", "Map", "UID1", "UID2", "Race1", "Race2", "Rank1", "Rank2"])
        events_out.writeheader()
        pool = Pool(min(cpu_count(), 60))
        entries = pool.map(generate_replay_entry,
                           filter(lambda file: file in valid_games, os.listdir(file_locations.REPLAY_FILE_DIRECTORY)))
        for entry in entries:
            if entry is not None:
                events_out.writerow(entry)


def test():
    generate_replay_info_csv("replays_info.csv")


if __name__ == "__main__":
    test()
