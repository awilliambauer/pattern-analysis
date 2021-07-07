import os
import sc2reader

import csv
from scouting_stats import map_pretty_name_to_file


def generate_player_specific_entry(player, number):
    return {"UID" + str(number): player.detail_data['bnet']['uid'], "Race" + str(number): player.play_race,
            "Rank" + str(number): player.highest_league}


def generate_replay_info_csv(replay_location_dir, output_file):
    valid_games = None
    with open("valid_game_ids.txt", "r") as f:
        valid_games = [line[:-1] for line in f.readlines()]
    with open(output_file, 'w', newline='') as fp:
        events_out = csv.DictWriter(fp,
                                    fieldnames=["ReplayID", "Map", "UID1", "UID2", "Race1", "Race2", "Rank1", "Rank2"])
        events_out.writeheader()
        valid = 0
        invalid = 0
        for file in os.listdir(replay_location_dir):
            if not file.endswith("SC2Replay"):
                continue
            if file not in valid_games:
                invalid += 1
                continue
            replay_id = file.split("_")[1].split(".")[0]
            valid += 1
            if valid % 500 == 0:
                print("working...")
            loaded_replay = sc2reader.load_replay(replay_location_dir + "/" + file)
            player_1_info = generate_player_specific_entry(loaded_replay.players[0], 1)
            player_2_info = generate_player_specific_entry(loaded_replay.players[1], 2)
            other_info = {"ReplayID": replay_id,
                          "Map": map_pretty_name_to_file(loaded_replay.map_name)}
            other_info.update(player_1_info)
            other_info.update(player_2_info)
            events_out.writerow(other_info)
        print("valid",valid, "invalid", invalid)


def test():
    generate_replay_info_csv("/Accounts/awb/pattern-analysis/starcraft/replays", "replays_info.csv")


if __name__ == "__main__":
    test()
