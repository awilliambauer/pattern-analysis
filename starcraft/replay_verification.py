import os
from multiprocessing import Pool, cpu_count
import sc2reader
import traceback
import csv
from itertools import repeat

import starcraft

valid_maps = []
with open(starcraft.VALID_MAP_LIST_FILE, "r") as valid_maps_file:
    for line in valid_maps_file:
        valid_maps.append(line[:-1])

maps_with_paths_generated = []
for path_data in os.listdir(starcraft.MAP_PATH_DATA_DIRECTORY):
    if path_data[:-5] not in maps_with_paths_generated:
        maps_with_paths_generated.append(path_data[:-5].strip())


def group_replays_by_map(replay_info_csv):
    maps = {}
    with open(replay_info_csv, "r") as replays_info:
        reader = csv.DictReader(replays_info)
        for row in reader:
            if row["Map"] not in maps.keys():
                maps[row["Map"]] = []
            maps[row["Map"]].append(row["ReplayID"])
    return maps


def map_pretty_name_to_file(map_name: str):
    return map_name.replace(" ", "").replace("'", "").strip()


# i have to make a separate function because multiprocessing is stupid
def _validate_replay(replay_file, replays_folder,
                     one_replay_per_player=False,
                     allow_ai_games=False,
                     allow_short_games=False,
                     allow_0_camera_events=False):
    if is_valid_replay(replays_folder + "/" + replay_file,
                       one_replay_per_player,
                       allow_ai_games,
                       allow_short_games,
                       allow_0_camera_events):
        return replay_file
    return None


def get_valid_replay_filenames(replays_folder, one_replay_per_player=False, allow_ai_games=False,
                               allow_short_games=False,
                               allow_0_camera_events=False):
    """
    Goes through all files in the given folder and returns a list of filenames which are able to be analyzed by our code.
    Note that unlike previous implementations of the valid_game_ids.txt generator, this does not check whether the
    scouting analysis code crashes when running on each replay, it merely checks that it matches all the requirements we
    have for replays
    """

    pool = Pool(min(cpu_count(), 60))
    valid_replay_filenames = pool.starmap(_validate_replay,
                                          zip(os.listdir(replays_folder), repeat(replays_folder),
                                              repeat(one_replay_per_player), repeat(allow_ai_games),
                                              repeat(allow_short_games), repeat(allow_0_camera_events)))
    pool.close()
    pool.join()
    return filter(lambda filename: filename is not None, valid_replay_filenames)


def save_valid_replay_filenames(replays_folder, output_file, one_replay_per_player=False, allow_ai_games=False,
                                allow_short_games=False,
                                allow_0_camera_events=False):
    replay_filenames = get_valid_replay_filenames(replays_folder, one_replay_per_player, allow_ai_games,
                                                  allow_short_games, allow_0_camera_events)
    with open(output_file, "w") as output:
        for filename in replay_filenames:
            output.write(filename + "\n")


def is_valid_replay(replay_file_path, one_replay_per_player=False, allow_ai_games=False, allow_short_games=False,
                    allow_0_camera_events=False):
    if replay_file_path[-9:] != "SC2Replay":
        return False

    # loading the replay
    try:
        replay = sc2reader.load_replay(replay_file_path)
        if any(v != (0, {}) for v in replay.plugin_result.values()):
            print(replay_file_path, replay.plugin_result)
    except:
        print(replay_file_path, "cannot load using sc2reader due to an internal error:")
        traceback.print_exc()
        return False
    # if replay.map.filename not in maps_with_paths_generated:
    #     print("Map", map_pretty_name_to_file(replay.map_name), "has not been pickled")
    #     return False
    if not replay.is_ladder:
        print(replay_file_path + " is not a ladder game")
    # if replay.map_name not in valid_maps:
    #     print(replay_file_path + " is not played on an official Blizzard map")
    #     return False
    if replay.winner is None:
        print(replay.filename, "has no winner information")
        return False
    try:
        # some datafiles did not have a 'Controller' attribute
        if replay.attributes[1]["Controller"] == "Computer" or replay.attributes[2]["Controller"] == "Computer" \
                and not allow_ai_games:
            print(replay.filename, "is a player vs. AI game")
            return False
    except:
        traceback.print_exc()
        return False

    if replay.length.seconds < 300 and not allow_short_games:
        print(replay.filename, "is shorter than 5 minutes")
        return False

    if len(replay.players) != 2:
        print(replay.filename, "is not a 1v1 game")
        return False
    print("replay verified")
    return True


def test():
    save_valid_replay_filenames("/Accounts/awb/pattern-analysis/starcraft/replays", "valid_replay_filenames.txt")

if __name__ == "__main__":
    test()