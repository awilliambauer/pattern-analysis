# used to generate a valid replay filenames list from a directory of sc2 replays
# this should then be used to generate a replays_info.csv with generate_replay_info.py
# ZL June 2021

import os
from multiprocessing import Pool, cpu_count
import sc2reader
import traceback
from itertools import repeat
import file_locations


def map_pretty_name_to_file(map_name: str):
    """Converts an English map name to a file name"""
    return map_name.replace(" ", "").replace("'", "").strip()


# i have to make a separate function because multiprocessing is stupid
def _validate_replay(replay_file, replays_folder):
    if is_valid_replay(replays_folder + "/" + replay_file):
        return replay_file
    return None


def get_valid_replay_filenames(replays_folder):
    """
    Goes through all files in the given folder and returns a list of filenames which are able to be analyzed by our code.
    Note that unlike previous implementations of the valid_replay_filenames.txt generator, this does not check whether the
    scouting analysis code crashes when running on each replay, it merely checks that it matches all the requirements we
    have for replays (game length, player count, etc)
    """
    files = os.listdir(replays_folder)
    pool = Pool(min(cpu_count(), 60))
    valid_replay_filenames = pool.starmap(_validate_replay,
                                          zip(files, repeat(replays_folder)))
    pool.close()
    pool.join()
    return filter(lambda x: x is not None, valid_replay_filenames)


def save_valid_replay_filenames(replays_folder, output_file):
    """
    Saves the results of get_valid_replay_filenames
    """
    replay_filenames = get_valid_replay_filenames(replays_folder)
    with open(output_file, "w") as output:
        for filename in replay_filenames:
            output.write(filename + "\n")


def is_valid_replay(replay_file_path):
    """
    Return true if the given replay file is a valid replay for our purposes, false otherwise
    """
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
    if not replay.is_ladder and "spawningtool" not in replay_file_path:
        print(replay_file_path + " is not a ladder game")
        return False
    if replay.winner is None:
        print(replay.filename, "has no winner information")
        return False
    try:
        # some datafiles did not have a 'Controller' attribute
        if replay.attributes[1]["Controller"] == "Computer" or replay.attributes[2]["Controller"] == "Computer":
            print(replay.filename, "is a player vs. AI game")
            return False
    except:
        traceback.print_exc()
        return False

    if replay.length.seconds < 300:
        print(replay.filename, "is shorter than 5 minutes")
        return False

    if len(replay.players) != 2:
        print(replay.filename, "is not a 1v1 game")
        return False
    print("replay verified")
    return True


def test():
    save_valid_replay_filenames(file_locations.REPLAY_FILE_DIRECTORY, file_locations.VALID_REPLAY_FILENAMES_FILE)


if __name__ == "__main__":
    test()
