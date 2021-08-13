import sc2reader
from data_analysis_helper import run, save
from collections import defaultdict, namedtuple
from math import dist
import file_locations
import traceback
from modified_rank_plugin import ModifiedRank

CameraHotkeyResult = namedtuple("CameraHotkeyResult", (
    "GameID", "UID1", "UID2", "Rank1", "Rank2", "Race1", "Race2", "HotkeyUsageCount1", "HotkeyUsageCount2", "Winner"))


def get_camera_hotkey_counts(replay):
    camera_event_locations = {1: defaultdict(int), 2: defaultdict(int)}
    last_camera_event_location = {1: (-1000, -1000), 2: (-1000, -1000)}
    hotkey_locations = {1: [], 2: []}
    for event in replay.events:
        if isinstance(event, sc2reader.events.game.CameraEvent) and event.player.pid in [1, 2, ]:
            if not dist(event.location[:2], last_camera_event_location[event.player.pid]) < 8:
                if camera_event_locations[event.player.pid][event.location[:2]] > 0:
                    # we have gotten to this location without scrolling before
                    if event.location[:2] not in hotkey_locations[event.player.pid]:
                        hotkey_locations[event.player.pid].append(event.location[:2])
                camera_event_locations[event.player.pid][event.location[:2]] += 1

            last_camera_event_location[event.player.pid] = event.location[:2]
    return sum(map(lambda hotkey: camera_event_locations[1][hotkey], hotkey_locations[1])), \
           sum(map(lambda hotkey: camera_event_locations[2][hotkey], hotkey_locations[2]))


def get_camera_hotkey_counts_data(filename, map_path_data):
    try:
        # extracting the game id and adding the correct tag
        # pathname = "practice_replays/" + filename
        pathname = file_locations.REPLAY_FILE_DIRECTORY + "/" + filename
        game_id = filename.split("_")[1].split(".")[0]
        if filename.startswith("ggg"):
            game_id = "ggg-" + game_id
        elif filename.startswith("spawningtool"):
            game_id = "st-" + game_id
        # loading the replay
        try:
            r = sc2reader.load_replay(pathname)
            if any(v != (0, {}) for v in r.plugin_result.values()):
                print(pathname, r.plugin_result)
        except:
            print(filename, "cannot load using sc2reader due to an internal ValueError")
            raise
        uid1 = r.players[0].detail_data['bnet']['uid']
        uid2 = r.players[1].detail_data['bnet']['uid']
        race1 = r.players[0].play_race
        race2 = r.players[1].play_race
        rank1 = r.players[0].highest_league
        rank2 = r.players[1].highest_league
        hotkey_usage_1, hotkey_usage_2 = get_camera_hotkey_counts(r)
        return CameraHotkeyResult(game_id, uid1, uid2, rank1, rank2, race1, race2, hotkey_usage_2,
                                  hotkey_usage_1, r.winner.players[0].pid)
    except KeyboardInterrupt:
        raise
    except:
        traceback.print_exc()
        return


if __name__ == '__main__':
    sc2reader.engine.register_plugin(ModifiedRank())
    results = run(get_camera_hotkey_counts_data)
    save(results, "camera_hotkeys_usage")
