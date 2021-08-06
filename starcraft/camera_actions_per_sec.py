from data_analysis_helper import run, save
from collections import namedtuple
import traceback
from sc2reader.events.game import CameraEvent
import sc2reader
import file_locations
from generate_replay_info import group_replays_by_map
from load_map_path_data import load_path_data
import time
from multiprocessing import Pool, cpu_count
import csv, datetime
from functools import partial
from collections import defaultdict


def stupid_default_dict_lambda():
    return defaultdict(int)


def get_cps(filename):
    try:
        pathname = file_locations.REPLAY_FILE_DIRECTORY + "/" + filename
        game_id = filename.split("_")[1].split(".")[0]
        if filename.startswith("ggg"):
            game_id = "ggg-" + game_id
        elif filename.startswith("spawningtool"):
            game_id = "st-" + game_id
        # loading the replay
        try:
            replay = sc2reader.load_replay(pathname)
            if any(v != (0, {}) for v in replay.plugin_result.values()):
                print(pathname, replay.plugin_result)
        except:
            print(filename, "cannot load using sc2reader due to an internal ValueError")
            raise
        camera_events = defaultdict(stupid_default_dict_lambda)
        for event in replay.events:
            if isinstance(event, sc2reader.events.game.CameraEvent):
                if event.player.pid in [1, 2]:
                    nearest_loc = (round(event.location[0]), round(event.location[1]))
                    camera_events[event.player.highest_league][nearest_loc] += 1
        print("got camera events for replay", filename)
        return camera_events
    except:
        traceback.print_exc()
        return None


if __name__ == "__main__":
    map_name = "BlackburnLE"
    results = defaultdict(lambda: defaultdict(int))
    replay_map_groups = group_replays_by_map()
    with Pool(min(60, cpu_count())) as pool:
        for map_name_group, replays in replay_map_groups.items():
            if map_name not in map_name_group:
                continue
            if len(replays) == 0:
                continue
            new_results = pool.map(get_cps, replays)
            for result in new_results:
                if result is not None:
                    for rank, camera_events in result.items():
                        for camera_loc, c in camera_events.items():
                            results[rank][camera_loc] += c

            break
    output_file = "camera_locations"
    output_file += datetime.datetime.now().date().__str__()
    output_file += ".txt"

    # assume that every result has the same fields
    with open(output_file, 'w', newline='') as output:
        for rank, camera_events in results.items():
            output.write(str(rank) + "\n")
            for camera_loc, c in camera_events.items():
                output.write(str(camera_loc) + ":" + str(c) + "\n")
