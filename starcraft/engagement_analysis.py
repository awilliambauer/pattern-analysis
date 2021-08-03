from engagement_detector import get_engagements
from collections import namedtuple
import file_locations
import sc2reader
import traceback
from sc2reader.engine.plugins import SelectionTracker, APMTracker
from selection_plugin import ActiveSelection
from base_plugins import BaseTracker
from modified_rank_plugin import ModifiedRank
from data_analysis_helper import run, save

EngagementResult = namedtuple("EngagementResult", (
    "GameID", "StartTimeSeconds", "EndTimeSeconds", "BaseClusterType", "BaseClusterPlayer",
    "AverageDeathPosX1", "AverageDeathPosY1",
    "TotalArmyValue1", "TotalArmySupply1", "ArmySupplyLost1", "ArmyValueLost1", "TotalWorkerSupply1",
    "WorkerSupplyLost1", "TotalBuildingCount1", "BuildingCountLost1", "AverageDeathPosX2", "AverageDeathPosY2",
    "TotalArmyValue2", "TotalArmySupply2", "ArmySupplyLost2", "ArmyValueLost2", "TotalWorkerSupply2",
    "WorkerSupplyLost2", "TotalBuildingCount2", "BuildingCountLost2"))


def get_engagement_results(filename):
    '''generateFields takes in a filename of a replay, loads it and gathers necessary
    statistics, and returns the statistics in a tuple. It is to be used to write
    these stats to a csv. It also takes in an integer (1 or 2), which indicates
    which statistics will be gathered. In this case, generateFields gathers
    each point in the game where a period of scouting is initiated. Inputting a
    1 will return times as a fraction of the total game time, whereas inputting
    a 2 will return absolute frames.'''
    # loading the replay
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
        results = []
        for engagement in get_engagements(r):
            deaths_1 = engagement.sides[0].unit_death_positions
            deaths_2 = engagement.sides[1].unit_death_positions
            average_death_x_1 = (sum(map(lambda pos: pos[0], deaths_1)) / len(deaths_1)) if len(deaths_1) > 0 else -1
            average_death_y_1 = (sum(map(lambda pos: pos[1], deaths_1)) / len(deaths_1)) if len(deaths_1) > 0 else -1
            average_death_x_2 = (sum(map(lambda pos: pos[0], deaths_2)) / len(deaths_2)) if len(deaths_2) > 0 else -1
            average_death_y_2 = (sum(map(lambda pos: pos[1], deaths_2)) / len(deaths_2)) if len(deaths_2) > 0 else -1
            side_1 = engagement.sides[0]
            side_2 = engagement.sides[1]
            results.append(
                EngagementResult(game_id, engagement.start_time / 22.4, engagement.end_time / 22.4,
                                 engagement.base_cluster.base_type,
                                 engagement.base_cluster.player_id, average_death_x_1, average_death_y_1,
                                 side_1.total_army_value,
                                 side_1.total_army_supply, side_1.army_supply_lost, side_1.army_value_lost,
                                 side_1.total_worker_supply,
                                 side_1.worker_supply_lost, side_1.total_building_count, side_1.building_count_lost,
                                 average_death_x_2, average_death_y_2, side_2.total_army_value,
                                 side_2.total_army_supply,
                                 side_2.army_supply_lost, side_2.army_value_lost, side_2.total_worker_supply,
                                 side_2.worker_supply_lost, side_2.total_building_count, side_2.building_count_lost)
            )
        return results

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
    results = run(get_engagement_results, n=10000)
    save(results, "engagements")
