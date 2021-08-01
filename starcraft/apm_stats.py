# Used to write statistics about player's APM behavior in StarCraft 2 to a csv
# David Chu
# July 2021

import math
import traceback

from numpy.lib.arraysetops import isin
import sc2reader
from sc2reader.engine.plugins import SelectionTracker, APMTracker
from file_locations import REPLAY_FILE_DIRECTORY
from modified_rank_plugin import ModifiedRank
from selection_plugin import ActiveSelection
from collections import namedtuple
from data_analysis_helper import run, save
import numpy as np
from control_groups import isMacro
import battle_detector


# creating the fields based on who won as a named tuple
fields_tuple = namedtuple('fields_tuple', ['game_id',
                                           'uid', 'rank',
                                           'apm', 'rel_apm',
                                           'apm_warmup', 'apm_non_warmup',
                                           'apm_battle_macro', 'apm_peace_macro',
                                           "win"])
    
def get_apm_stats(replay, battles):
    """get_apm_stats takes in a replay and returns the macro apm in battle 
    and peace time, the apm in warmup and non-warmup time, and 
    the average apm of a player."""
    team1_avg_apm = replay.players[0].avg_apm/1.4
    team2_avg_apm = replay.players[1].avg_apm/1.4
    
    frames = replay.frames
    seconds = replay.real_length.total_seconds()
    game_events = replay.game_events
    
    team1_seconds = replay.humans[0].seconds_played/1.4
    team2_seconds = replay.humans[1].seconds_played/1.4
    
    team_apm_count = {1 : 0, 2: 0}
    aps_battle_count = {1 : 0, 2 : 0}
    aps_peace_count = {1 : 0, 2 : 0}
    aps_wmup_count = {1 : 0, 2 : 0}
    aps_non_wmup_count = {1 : 0, 2 : 0}
    
    battle_time = 0
    
    for event in game_events:
        if  isinstance(event, sc2reader.events.game.ControlGroupEvent) or \
            isinstance(event, sc2reader.events.game.SelectionEvent) or \
            isinstance(event, sc2reader.events.game.CommandEvent):
            second = event.second
            frame = event.frame
            team = event.player.pid
            if isMacro(event.active_selection):
                if battle_detector.duringBattle(frame, battles):
                    aps_battle_count[team] += 1
                else:
                    aps_peace_count[team] += 1
            if second < 120:
                aps_wmup_count[team] += 1
            else:
                aps_non_wmup_count[team] += 1
                
            team_apm_count[team] += 1
                
    # Calculating total peacetime and total battletime (in seconds) for the game
    for battle in battles:
        starttime = (battle[0]/frames)*seconds
        endtime = (battle[1]/frames)*seconds
        duration = endtime - starttime
        battle_time += duration
    peace_time = seconds - battle_time
    
    team1_apm_battle = aps_battle_count[1] / (battle_time / 60)
    team2_apm_battle = aps_battle_count[2] / (battle_time / 60)
    team1_apm_peace = aps_peace_count[1] / (peace_time / 60)
    team2_apm_peace = aps_peace_count[2] / (peace_time / 60)
    
    team1_apm_wmup = aps_wmup_count[1] / (86 / 60)
    team1_apm_non_wmup = aps_non_wmup_count[1] / ((team1_seconds - 86) / 60)
    team2_apm_wmup = aps_wmup_count[2] / (86 / 60)
    team2_apm_non_wmup = aps_non_wmup_count[2] / ((team2_seconds - 86) / 60)
    
    # print("default", team_apm_count[1]/team1_seconds * 60, "calculated:", team1_avg_apm)
    # print("default", team_apm_count[2]/team2_seconds * 60, "calculated:", team2_avg_apm)
    
    # sum_1 = aps_wmup_count[1] + aps_non_wmup_count[1]
    # sum_2 = aps_wmup_count[2] + aps_non_wmup_count[2]
    
    # print("total ", team_apm_count[1], "count 1 ", aps_wmup_count[1], "count 2", aps_non_wmup_count[1], "seconds", team1_seconds)
    # print("total ", team_apm_count[2], "count 1 ", aps_wmup_count[2], "count 2", aps_non_wmup_count[2], "seconds", team2_seconds)
    
    return  team1_apm_battle, team1_apm_peace, team1_avg_apm, team1_apm_wmup, team1_apm_non_wmup, \
            team2_apm_battle, team2_apm_peace, team2_avg_apm, team2_apm_wmup, team2_apm_non_wmup   
            
                
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

        battles, harassing = battle_detector.buildBattleList(replay)
        winner = replay.winner.number
        team1_rank, team1_rel_rank, team2_rank, team2_rel_rank = ranking_stats(replay)
        
        # APM stats
        team1_apm_battle, team1_apm_peace, team1_avg_apm, team1_apm_wmup, team1_apm_non_wmup, \
        team2_apm_battle, team2_apm_peace, team2_avg_apm, team2_apm_wmup, team2_apm_non_wmup = get_apm_stats(replay, battles)
        team1_rel_apm = team1_avg_apm - team2_avg_apm
        team2_rel_apm = team2_avg_apm - team1_avg_apm

        # Player ID
        team1_uid = replay.players[0].detail_data['bnet']['uid']
        team2_uid = replay.players[1].detail_data['bnet']['uid']

        if winner == 1:
            fields_1 = fields_tuple(game_id,
                                  team1_uid, team1_rank,
                                  team1_avg_apm, team1_rel_apm,
                                  team1_apm_wmup, team1_apm_non_wmup,
                                  team1_apm_battle, team1_apm_peace, 1)
            fields_2 = fields_tuple(game_id,
                                  team2_uid, team2_rank,
                                  team2_avg_apm, team2_rel_apm,
                                  team2_apm_wmup, team2_apm_non_wmup,
                                  team2_apm_battle, team2_apm_peace, 0)
        elif winner == 2:
            fields_1 = fields_tuple(game_id,
                                  team1_uid, team1_rank,
                                  team1_avg_apm, team1_rel_apm,
                                  team1_apm_wmup, team1_apm_non_wmup,
                                  team1_apm_battle, team1_apm_peace, 0)
            fields_2 = fields_tuple(game_id,
                                  team2_uid, team2_rank,
                                  team2_avg_apm, team2_rel_apm,
                                  team2_apm_wmup, team2_apm_non_wmup,
                                  team2_apm_battle, team2_apm_peace, 1)
        return [fields_1, fields_2]
    except:
        print("exception while generating scouting stats for replay", replay_file)
        traceback.print_exc()
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
    sc2reader.engine.register_plugin(ModifiedRank())
    # results = run(generate_fields, threads = 15, replay_filter = lambda replay: replay['ReplayID'] in \
        # ['gggreplays_261224.SC2Replay', 'gggreplays_308021.SC2Replay', 'gggreplays_301713.SC2Replay'])
    results = run(generate_fields, threads = 15)
    save(results, "apm_stats_data_test")
