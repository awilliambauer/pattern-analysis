# Used to write statistics about player's scouting behavior in StarCraft 2 to a csv
# Alison Cameron, David Chu, Zimri Leisher
# July 2021

import scouting_detector
import math
import sc2reader
from sc2reader.engine.plugins import SelectionTracker, APMTracker
import control_groups
import scouting_detector
from file_locations import REPLAY_FILE_DIRECTORY
from base_plugins import BaseTracker, BaseType
from modified_rank_plugin import ModifiedRank
from selection_plugin import ActiveSelection
import traceback
import battle_detector
from collections import namedtuple
from data_analysis_helper import run, save

# creating the fields based on who won as a named tuple
fields_tuple = namedtuple('fields_tuple', ['game_id',
                                           'uid', 'rank', 'scout_freq',
                                           'scout_freq_fb', 'scout_mb', 'scout_first',
                                           'apm', 'rel_apm',
                                           'cps', 'peace_rate',
                                           'battle_rate', "win",])


def get_scouting_frequency(replay, map_path_data):
    '''get_scouting_frequency takes in a previously loaded replay
    from sc2reader and returns the scouting frequency (instances per second),
    the scouting_frequnecy after the first battle, the ratio of the number of 
    scouting instances of the opponent's mainbase to the total number of scouting 
    instances, and the frame of the first scouting instance for each player.'''
    r = replay
    frames = r.frames
    seconds = r.real_length.total_seconds()
    battles, harassing = battle_detector.buildBattleList(r)
    try:
        first_battle_start_frame = battles[0][0]
    except:
        print("Battle time or peace time is zero")
        raise RuntimeError()

    try:
        scouting_instances_1, scouting_instances_2 = scouting_detector.get_scouting_instances(r, map_path_data)
        check_1, check_2 = scouting_instances_1[0], scouting_instances_2[0]
    except:
        print("No scouting instance was detected")
        raise RuntimeError()
    scouting_count_1, scouting_count_2 = 0, 0
    scouting_count_after_first_battle_1, scouting_count_after_first_battle_2 = 0, 0
    first_scouting_time_1, first_scouting_time_2 = 0, 0
    is_first_scouting_1, is_first_scouting_2 = True, True
    scouting_mb_1_count, scouting_mb_2_count = 0, 0

    # player 1's scouting instances
    for scouting_instance in scouting_instances_1:
        if scouting_instance.start_time > first_battle_start_frame:
            scouting_count_after_first_battle_1 += 1
        if is_first_scouting_1:
            first_scouting_time_1 = scouting_instance.start_time / 22.4
            is_first_scouting_1 = False
        if scouting_instance.scouting_type == BaseType.MAIN:
            scouting_mb_1_count += 1
        scouting_count_1 += 1

    # player 2's scouting instances
    for scouting_instance in scouting_instances_2:
        if scouting_instance.start_time > first_battle_start_frame:
            scouting_count_after_first_battle_2 += 1
        if is_first_scouting_2:
            first_scouting_time_2 = scouting_instance.start_time / 22.4
            is_first_scouting_2 = False
        if scouting_instance.scouting_type == BaseType.MAIN:
            scouting_mb_2_count += 1
        scouting_count_2 += 1


    # scouting rate (instances/seconds)
    scouting_freq_1 = scouting_count_1 / seconds
    scouting_freq_2 = scouting_count_2 / seconds

    # scouting rate after the first battle (instances/seconds)
    seconds_after_fb = (frames - first_battle_start_frame) / 22.4
    scouting_freq_fb_1 = scouting_count_after_first_battle_1 / seconds_after_fb
    scouting_freq_fb_2 = scouting_count_after_first_battle_2 / seconds_after_fb

    # scouting opponent's main base ratio to total scouting instances
    scouting_mb_ratio_1 = scouting_mb_1_count / scouting_count_1
    scouting_mb_ratio_2 = scouting_mb_2_count / scouting_count_2

    return  scouting_freq_1, scouting_freq_fb_1, scouting_mb_ratio_1, first_scouting_time_1, \
            scouting_freq_2, scouting_freq_fb_2, scouting_mb_ratio_2, first_scouting_time_2



def generate_fields(replay_file, map_path_data):
    """generate_fields takes in a filename of a replay, loads it and gathers necessary
    statistics, and returns the statistics in a named tuple. It is to be used to write
    these stats to a csv."""
    try:
        # convert replay file name to path
        replay_file_path = REPLAY_FILE_DIRECTORY + "/" + replay_file
        # assume that replays that are passed in are valid
        # if they are valid, sc2reader should not crash
        replay = sc2reader.load_replay(replay_file_path)
        # KEEP IN MIND: due to a small mistake generating replay files, some of them have an underscore before the
        # .SC2Replay I think this code still works...
        game_id = replay_file.split("_")[1].split(".")[0]
        if replay_file.startswith("ggg"):
            game_id = "ggg-" + game_id
        elif replay_file.startswith("spawningtool"):
            game_id = "st-" + game_id
        elif replay_file.startswith("dropsc"):
            game_id = "ds-" + game_id

        # Scouting stats
        team1_freq, team1_freq_fb, team1_scout_mb, team1_first_scouting, \
        team2_freq, team2_freq_fb, team2_scout_mb, team2_first_scouting, \
            = get_scouting_frequency(replay, map_path_data)
        winner = replay.winner.number
        team1_rank, team1_rel_rank, team2_rank, team2_rel_rank = ranking_stats(replay)

        # Control group stats
        team1_cps, team1_peace_rate, team1_battle_rate, team2_cps, team2_peace_rate, team2_battle_rate, \
        team1_peace_rb, team2_peace_rb, team1_battle_rb, team2_battle_rb, \
        team1_cps_cg, team2_cps_cg = control_groups.control_group_stats(replay)

        # APM stats
        team1_apm = replay.players[0].avg_apm
        team2_apm = replay.players[1].avg_apm
        team1_rel_apm = team1_apm - team2_apm
        team2_rel_apm = team2_apm - team1_apm

        # Player ID
        team1_uid = replay.players[0].detail_data['bnet']['uid']
        team2_uid = replay.players[1].detail_data['bnet']['uid']

        if winner == 1:
            fields_1 = fields_tuple(game_id,
                                  team1_uid, team1_rank, team1_freq,
                                  team1_freq_fb, team1_scout_mb, team1_first_scouting,
                                  team1_apm, team1_rel_apm,
                                  team1_cps, team1_peace_rate,
                                  team1_battle_rate, 1)
            fields_2 = fields_tuple(game_id,
                                  team2_uid, team2_rank, team2_freq,
                                  team2_freq_fb, team2_scout_mb, team2_first_scouting,
                                  team2_apm, team2_rel_apm,
                                  team2_cps, team2_peace_rate,
                                  team2_battle_rate, 0)
        elif winner == 2:
            fields_1 = fields_tuple(game_id,
                                  team1_uid, team1_rank, team1_freq,
                                  team1_freq_fb, team1_scout_mb, team1_first_scouting,
                                  team1_apm, team1_rel_apm,
                                  team1_cps, team1_peace_rate,
                                  team1_battle_rate, 0)
            fields_2 = fields_tuple(game_id,
                                  team2_uid, team2_rank, team2_freq,
                                  team2_freq_fb, team2_scout_mb, team2_first_scouting,
                                  team2_apm, team2_rel_apm,
                                  team2_cps, team2_peace_rate,
                                  team2_battle_rate, 1)
        return [fields_1, fields_2]
    except:
        print("exception while generating scouting stats for replay", replay_file)
        # traceback.print_exc()
        return None


def ranking_stats(replay):
    '''ranking_stats takes in a previously loaded replay and returns each player's
    rank and their rank relative to their opponent. If rankings don't exist,
    then the rank is NaN and so is the relative rank.'''
    p1_rank = replay.players[0].highest_league
    p2_rank = replay.players[1].highest_league

    # checking if rank exists for each player
    if (p1_rank == 0) or (p1_rank == 8):
        p1_rank = math.nan
    if (p2_rank == 0) or (p2_rank == 8):
        p2_rank = math.nan

    # if rankings exist for both players, then calculate relative ranks
    if not math.isnan(p1_rank) and not math.isnan(p2_rank):
        p1_rel = p1_rank - p2_rank
        p2_rel = p2_rank - p1_rank
    # if rankings don't exist for both players, then relative ranks are NaN
    else:
        p1_rel, p2_rel = math.nan, math.nan

    return p1_rank, p1_rel, p2_rank, p2_rel


if __name__ == "__main__":
    '''This main function parses command line arguments and calls
    writeToCsv, which will write statistics to a csv for each
    StarCraft 2 replay file in a directory.'''
    sc2reader.engine.register_plugin(APMTracker())
    sc2reader.engine.register_plugin(SelectionTracker())
    sc2reader.engine.register_plugin(ActiveSelection())
    sc2reader.engine.register_plugin(BaseTracker())
    sc2reader.engine.register_plugin(ModifiedRank())
    results = run(generate_fields, threads = 10)
    save(results, "sc2_prediction_data")
