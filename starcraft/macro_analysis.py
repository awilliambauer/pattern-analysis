from collections import namedtuple
from data_analysis_helper import run, save
import file_locations
import sc2reader
import traceback
from sc2reader.engine.plugins import SelectionTracker, APMTracker
from selection_plugin import ActiveSelection
from base_plugins import BaseTracker
from modified_rank_plugin import ModifiedRank
from ggpyjobs_plugins import ZergMacroTracker, ProtossTerranMacroTracker
MacroStatsResult = namedtuple("MacroStatsResult",
                              ["GameID", "UID1", "UID2", "Race1", "Race2", "Rank1", "Rank2", "MacroUtilization1",
                               "MacroUtilization2", "Winner"])


def get_macro_stats(filename, map_path_data):
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
        player1 = r.players[0] if r.players[0].pid == 1 else r.players[1]
        uid1 = r.players[0].detail_data['bnet']['uid']
        uid2 = r.players[1].detail_data['bnet']['uid']
        race1 = r.players[0].play_race
        race2 = r.players[1].play_race
        rank1 = r.players[0].highest_league
        rank2 = r.players[1].highest_league
        ZergMacroTracker(r)
        ProtossTerranMacroTracker(r)
        return MacroStatsResult(game_id, uid1, uid2, race1, race2, rank1, rank2, r.players[0].race_macro_count,
                                r.players[1].race_macro_count, r.winner.players[0].pid)
    except KeyboardInterrupt:
        raise
    except:
        traceback.print_exc()
        return


if __name__ == '__main__':
    sc2reader.engine.register_plugin(APMTracker())
    sc2reader.engine.register_plugin(SelectionTracker())
    sc2reader.engine.register_plugin(ModifiedRank())
    sc2reader.engine.register_plugin(ActiveSelection())
    bt = BaseTracker()
    #     bt.logger.setLevel(logging.ERROR)
    #     bt.logger.addHandler(logging.StreamHandler(sys.stdout))
    sc2reader.engine.register_plugin(bt)
    results = run(get_macro_stats)
    save(results, "macro_count")
