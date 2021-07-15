# Used to write statistics about player behavior in StarCraft 2 to a csv
# Alison Cameron
# July 2020
from scouting_detector import is_scouting, final_scouting_states, to_time, print_time
import argparse
import csv
import time
from itertools import repeat
from multiprocessing import Pool, cpu_count

import math
import sc2reader
from sc2reader.engine.plugins import SelectionTracker, APMTracker

import control_groups
import scouting_detector
from base_plugins import BaseTracker
from map_path_generation import load_path_data
from replay_verification import group_replays_by_map
from selection_plugin import ActiveSelection
from sc2.position import Point2

import statistics
import traceback
import battle_detector
from unit_prediction import get_position_estimate_along_path, get_movement_speed, is_flying_unit


def scouting_stats(scouting_dict):
    '''scouting_stats calculates the number of times a player initiates scouting
    and the total number of frames they spend scouting their opponent.
    It takes in a scouting dictionary returned by final_scouting_states
    and returns the number of instances and the total number of frames.'''
    num_times = 0
    total_frames = 0
    scouting_frames = 0
    cur_scouting = False

    length = len(scouting_dict.keys())
    if is_scouting(scouting_dict[1]):
        num_times += 1
        scouting_frames += 1
        cur_scouting = True
    total_frames += 1
    frame = 2
    while (frame < length):
        total_frames += 1
        if is_scouting(scouting_dict[frame]):
            # if the player is in a streak of scouting
            if cur_scouting == True:
                scouting_frames += 1
            # if the player just switched states from something else
            # to scouting their opponent
            else:
                num_times += 1
                scouting_frames += 1
                cur_scouting = True
        else:
            cur_scouting = False
        frame += 1

    # calculating rates based on counts, currently not useful information so we don't return
    scouting_fraction = scouting_frames / total_frames
    scouting_rate = num_times / total_frames

    return num_times, scouting_frames


def categorize_player(scouting_frames, battles, harassing, total_frames):
    '''categorize_player is used to sort players based on their scouting
    behavior. It takes in the frames in which scouting begins (returned by
    scouting_timeframe_list1), a list of battles and instances of
    harassing (returned by battle_detector.buildBattleList), and the total
    frames in the game. categorize_player incorporates all contact with the
    opponent, because a player can learn about their opponent through battles
    and harassing, and not just scouting. The categories are numerical (1-4)
    and are returned by this function. The following is a summary of each category.

    1. No scouting - there are no scouting tags in a player's dictionary

    2. Only scouts in the beginning - the only existing scouting tags happen
    within the first 25% of the game

    3. Sporadic scouters - a player scouts past the first 25% of the game,
    but the time intervals between instances of scouting are inconsistent.
    Intervals are considered inconsistent if the standard deviation is
    greater than or equal to a third of the mean of all intervals.

    4. Consistent scouters - a player scouts past the first 25% of the game,
    and the time intervals between instances of scouting are consistent.
    Intervals are considered consistent if the standard deviation is less
    than a third of the mean.'''

    # No scouting, return the 1st category
    if len(scouting_frames) == 0:
        category = 1
        return category

    unsorted_engagements = battles + harassing
    engagements = sorted(unsorted_engagements, key=lambda e: e[0])
    cur_engagement = 0
    cur_scout = 0

    no_scouting = True
    beginning_scouting = True
    intervals = []

    frame = 1
    interval = 0
    after_first = False
    while frame <= total_frames:
        engagement = engagements[cur_engagement]
        scout = scouting_frames[cur_scout]

        # handling the scouting frames
        if frame == scout:
            after_first = True
            if interval:
                intervals.append(interval)
            interval = 0

            if frame / total_frames >= 0.25:
                beginning_scouting = False

            if cur_scout + 1 < len(scouting_frames):
                cur_scout += 1

        # handling the engagements
        elif frame == engagement[0]:
            after_first = True
            if interval:
                intervals.append(interval)
            interval = 0

            if cur_engagement + 1 < len(engagements):
                cur_engagement += 1

            frame = engagement[1]

        else:
            if after_first:
                interval += 1

        frame += 1

    # only scouts in the beginning or there are 2 or less instances of scouting
    if beginning_scouting or (len(intervals) >= 0 and len(intervals) <= 2):
        category = 2
        return category

    mean_interval = statistics.mean(intervals)
    stdev = statistics.pstdev(intervals)

    # Sporadic scouter
    if stdev / mean_interval >= 0.3:
        category = 3
    # Consistent scouter
    elif stdev / mean_interval < 0.3:
        category = 4

    return category


def avg_interval(scouting_dict, scale):
    '''avg_interval returns the average time interval (in seconds) between
        periods of scouting. It takes in a completes scouting dictionary
        returned by final_scouting_states(replay) as well as the scale,
        which is in frames per seconds.'''
    new_scale = 16 * scale
    intervals = []
    keys = scouting_dict.keys()
    after_first = False
    any_scouting = False
    start_interval = 0
    interval = 0

    for key in keys:
        state = scouting_dict[key]
        if is_scouting(state):
            after_first = True
            any_scouting = True
            if interval:
                intervals.append(interval)
            interval = 0
        else:
            if after_first:
                interval += 1
            else:
                start_interval += 1

    if len(intervals) == 0 and any_scouting:
        # only one instance of scouting, return the time it took to get
        # to that first instance
        return start_interval / new_scale
    elif len(intervals) == 0 and not (any_scouting):
        # no scouting ocurred, return a flag to indicate this
        return -1
    elif len(intervals) > 0:
        # 2 or more instances of scouting ocurred, find the average interval between
        mean_interval = (statistics.mean(intervals)) / new_scale
        return mean_interval


def scouting_timefrac_list(scouting_dict, frames):
    '''scouting_timefrac_list returns a list of instances where scouting
        begins as fractions of the total gametime. It takes in a completed
        scouting dictionary returned by final_scouting_states(replay), as
        well as the total number of frames in the game.'''
    time_fracs = []
    keys = scouting_dict.keys()
    cur_scouting = False
    for key in keys:
        state = scouting_dict[key]
        if is_scouting(state):
            if not (cur_scouting):
                frac = key / frames
                time_fracs.append(frac)
            cur_scouting = True
        else:
            cur_scouting = False

    return time_fracs


def scouting_timeframe_list1(scouting_dict):
    '''scouting_timeframe_list1 returns a list of frames where a period of
        scouting began. It takes in a completed scouting dictionary returned
        by final_scouting_states(replay).'''
    time_frames = []
    keys = scouting_dict.keys()
    cur_scouting = False
    for key in keys:
        state = scouting_dict[key]
        if is_scouting(state):
            if not (cur_scouting):
                time_frames.append(key)
            cur_scouting = True
        else:
            cur_scouting = False
    return time_frames


def has_initial_scouting(scouting_dict, frames, battles):
    '''hasInitialScouting returns 1 (indicating True) if a player scouts
    for the first time in the first 25% of the game and before the first
    battle, and 0 (indicating False) otherwise. It takes in a scouting
    dictionary returned by final_scouting_states, the total frames in a
    game, and a list of battles returned by battle_detector.buildBattleList.'''
    keys = scouting_dict.keys()
    if len(battles) == 0:
        first_battle_start = frames
    else:
        first_battle_start = battles[0][0]
    for key in keys:
        state = scouting_dict[key]
        if is_scouting(state) and key / frames <= 0.25 and key < first_battle_start:
            # True
            return 1
    # False
    return 0


def scoutsMainBase(scouting_dict):
    '''scoutsMainBase returns 1 (indicating True) if a player scouts their
    opponent's main base more than 50% of the times they scout, and
    0 (indicating False) if a player scouts expansion bases more than 50%
    of the time. It takes in a complete scouting dictionary returned
    by final_scouting_states.'''
    keys = scouting_dict.keys()
    mainbase_ct = 0
    scouting_ct = 0
    cur_scouting = False
    for key in keys:
        state = scouting_dict[key]
        if is_scouting(state) and not (cur_scouting):
            scouting_ct += 1
            cur_scouting = True

            if "main base" in state[0]:
                mainbase_ct += 1

        else:
            cur_scouting = False

    if scouting_ct == 0:
        # No scouting, return a flag
        return -1

    if mainbase_ct / scouting_ct >= 0.5:
        # True
        return 1
    else:
        # False
        return 0


def scout_new_areas(scouting_dict):
    '''scoutNewAreas returns 1 (indicating True) if a player consistently
    scouts new locations on the map, and 0 (indicating False) otherwise.
    Scouting new areas is considered consistent if the average distance between
    areas of scouting is greater than or equal to 20 units. The typical map is
    about 200 by 200 units. scoutNewAreas takes in a complete scouting dictionary
    returned by final_scouting_states.'''
    locations = []
    keys = scouting_dict.keys()
    new_instance = True
    for key in sorted(keys):
        state = scouting_dict[key]
        if is_scouting(state) and "Scouting opponent" in state[0] and len(state) == 3:
            new_instance = False
            locations.append(state[2])

    # 1 or less instances of scouting, return a flag indicating that this measure is invalid
    if len(locations) <= 1:
        return -1

    distances_apart = []
    for i in range(len(locations)):
        for j in range(i):
            first_loc = locations[i]
            second_loc = locations[j]
            first_x, first_y = first_loc[0], first_loc[1]
            second_x, second_y = second_loc[0], second_loc[1]
            x_diff, y_diff = abs(first_x - second_x), abs(first_y - second_y)
            distance_apart = math.sqrt(x_diff ** 2 + y_diff ** 2)
            distances_apart.append(distance_apart)

    avg_distance = statistics.mean(distances_apart)

    if avg_distance > 20:
        # True
        return 1
    else:
        # False
        return 0


def scout_between_battles(scouting_dict, battles, frames):
    '''scoutBetweenBattles returns 1 (indicating True) if a player has
    at least 1 instance of scouting for 70% of the time periods between
    battles, and 0 (indicating False) if otherwise. It takes in a complete
    scouting dictionary returned by final_scouting_states, a list of
    battles returned by battle_detector.buildBattleList, and the total
    number of frames in a game.'''
    num_battles = len(battles)

    # not enough battles for a player to scout between them, return a flag
    if num_battles <= 1:
        return -1

    # creating a dictionary of frames of peacetime between battles
    peacetime_dict = {}
    peacetime_list = []
    for i in range(num_battles - 1):
        battle = battles[i]
        next_battle = battles[i + 1]
        peace_start = battle[1] + 1
        peace_end = next_battle[0] - 1
        peacetime_dict[(peace_start, peace_end)] = 0
        peacetime_list.append((peace_start, peace_end))

    # checking for scouting in between battles
    scouting_keys = scouting_dict.keys()
    peacetime_keys = peacetime_dict.keys()
    new_instance = True
    for s_key in scouting_keys:
        state = scouting_dict[s_key]
        # avoiding counting one instance of scouting more than once
        if new_instance and is_scouting(state):
            new_instance = False
            # checking which period of peacetime the scouting occurs
            for p_key in peacetime_keys:
                if s_key >= p_key[0] and s_key <= p_key[1]:
                    # adding to the count of the correct peacetime
                    peacetime_dict[p_key] += 1
                    break
        else:
            # If the scouting state changes or it is no longer during peacetime, then reset to a new instance
            if not (is_scouting(state)) or not (battle_detector.duringBattle(s_key, peacetime_list)):
                new_instance = True

    # determine if there are enough scouting instances between battles
    nums_between = []
    for (pstart, pend), count in peacetime_dict.items():
        if pend - pstart > 22.4 * 20:  # only consider peacetime longer than 20s
            # we care whether they scouted, during each peacetime, but not how many times
            nums_between.append(min(1, count))

    # at least one instance of scouting for 70% of peacetime periods
    return 1 if len(nums_between) > 0 and statistics.mean(nums_between) >= 0.7 else 0


def avg_interval_before_battle(scouting_frames, battles, scale):
    '''avg_interval_before_battle returns the average length (in seconds)
    between instances of scouting and a battle, if they are within 1 minute
    of each other. This function is a start to the exploration of
    responses to scouting. It takes in a list of frames where scouting started,
    returned by scouting_time_frames1, a list of battles returned by
    battle_detector.buildBattleList, and the scale as game speed in frames per
    second.'''
    frame_jump60 = 60 * int(scale)
    num_battles = len(battles)
    intervals = []

    # no battles or no scouting, return a flag
    if num_battles == 0 or len(scouting_frames) == 0:
        return -1

    # one interval for every battle
    for i in range(num_battles):
        intervals.append(0)

    cur_battle = 0
    prev_battle_end = 1
    for frame in scouting_frames:
        battle_start = battles[cur_battle][0]
        # if the frame is within a minute of the start of the battle
        if frame >= battle_start - frame_jump60 and frame < battle_start and frame > prev_battle_end:
            frame_interval = battle_start - frame
            sec_interval = frame_interval / scale
            # reset the battle-specific interval to the smallest interval
            intervals[cur_battle] = sec_interval

        elif (frame > battle_start) and (cur_battle + 1 < num_battles):
            prev_battle_end = battles[cur_battle][1]
            cur_battle += 1
            # re-check scouting instance
            battle_start = battles[cur_battle][0]
            # if the frame is within a minute of the start of the battle
            if frame >= battle_start - frame_jump60 and frame < battle_start and frame > prev_battle_end:
                frame_interval = battle_start - frame
                sec_interval = frame_interval / scale
                # reset the battle-specific interval to the smallest interval
                intervals[cur_battle] = sec_interval
        elif cur_battle + 1 >= num_battles:
            break

    # filtering out initialized zeros so they don't skew the average
    non_zero_intervals = []
    for interval in intervals:
        if interval != 0:
            non_zero_intervals.append(interval)

    # no battles happened in response to scouting, return a flag
    if len(non_zero_intervals) == 0:
        return -1

    avg = statistics.mean(non_zero_intervals)
    return avg


def scouting_freq_and_cat(replay, current_map_path_data):
    '''scouting_freq_and_cat takes in a previously loaded replay
    from sc2reader and returns the scouting frequency (instances per second)
    for each player, how their scouting behavior is categorized, as well as
    the winner of the game.'''
    r = replay

    try:
        frames = r.frames
        seconds = r.real_length.total_seconds()

        team1_scouting_states, team2_scouting_states = final_scouting_states(r, current_map_path_data)

        team1_num_times, team1_time = scouting_stats(team1_scouting_states)
        team2_num_times, team2_time = scouting_stats(team2_scouting_states)

        team1_freq = team1_num_times / seconds
        team2_freq = team2_num_times / seconds

        team1_frames = scouting_timeframe_list1(team1_scouting_states)
        team2_frames = scouting_timeframe_list1(team2_scouting_states)

        battles, harassing = battle_detector.buildBattleList(r)

        team1_cat = categorize_player(team1_frames, battles, harassing, frames)
        team2_cat = categorize_player(team2_frames, battles, harassing, frames)

        return team1_freq, team1_cat, team2_freq, team2_cat, r.winner.number

    except:
        print(replay.filename, "contains errors within scouting_detector")
        raise


def scouting_analysis(replay):
    '''scouting_analysis takes in a previously loaded replay and returns a
    dictionary that contains verious metrics of scouting for each player.
    These metrics include the scouting category, whether or not they
    execute an initial scouting, whether they mostly scout their opponent's
    main base, whether they consistently scout new areas of the map, and
    whether they consistently scout between battles.'''
    r = replay
    try:
        team1_scouting_states, team2_scouting_states = final_scouting_states(r)
        battles, harassing = battle_detector.buildBattleList(r)

        frames = r.frames
        team1_initial = has_initial_scouting(team1_scouting_states, frames, battles)
        team2_initial = has_initial_scouting(team2_scouting_states, frames, battles)

        team1_frames = scouting_timeframe_list1(team1_scouting_states)
        team2_frames = scouting_timeframe_list1(team2_scouting_states)

        team1_cat = categorize_player(team1_frames, battles, harassing, frames)
        team2_cat = categorize_player(team2_frames, battles, harassing, frames)

        team1_base = scoutsMainBase(team1_scouting_states)
        team2_base = scoutsMainBase(team2_scouting_states)

        team1_newAreas = scout_new_areas(team1_scouting_states)
        team2_newAreas = scout_new_areas(team2_scouting_states)

        team1_betweenBattles = scout_between_battles(team1_scouting_states, battles, frames)
        team2_betweenBattles = scout_between_battles(team2_scouting_states, battles, frames)

        return {1: [team1_cat, team1_initial, team1_base, team1_newAreas, team1_betweenBattles],
                2: [team2_cat, team2_initial, team2_base, team2_newAreas, team2_betweenBattles]}
    except:
        traceback.print_exc()
        print(replay.filename, "contains errors within scouting_detector")
        raise


def scouting_times(replay, which, current_map_path_data):
    '''scouting_times takes in a previously loaded replay from sc2reader as
        well as an integer (either 1 or 2) indicating which type of time list
        will be returned. 1 indicates a list of when scouting occurs as fractions
        of gametime, whereas 2 indicates a list of absolute frames.'''
    r = replay

    try:
        frames = r.frames
        team1_scouting_states, team2_scouting_states = final_scouting_states(r, current_map_path_data)

        # times normalized by the length of the game
        if which == 1:
            team1_time_list = scouting_timefrac_list(team1_scouting_states, frames)
            team2_time_list = scouting_timefrac_list(team2_scouting_states, frames)
        # absolute frames
        elif which == 2:
            team1_time_list = scouting_timeframe_list1(team1_scouting_states)
            team2_time_list = scouting_timeframe_list1(team2_scouting_states)

        return team1_time_list, team2_time_list

    except:
        traceback.print_exc()
        print(replay.filename, "contains errors within scouting_detector")
        raise


def scouting_interval(replay):
    '''scouting_interval takes in a previously loaded replay from sc2reader
    and returns the average time (in seconds) between periods of scouting
    for each player.'''
    r = replay
    try:
        factors = sc2reader.constants.GAME_SPEED_FACTOR
        # scale = factors[r.expansion][r.speed]
        # it appears the scale is always 22.4 in our dataset, despite documentation to the contrary
        scale = 22.4

        team1_scouting_states, team2_scouting_states = final_scouting_states(r)

        team1_avg_int = avg_interval(team1_scouting_states, scale)
        team2_avg_int = avg_interval(team2_scouting_states, scale)

        return team1_avg_int, team2_avg_int

    except:
        print(replay.filename, "contains errors within scouting_detector")
        raise


def scouting_response(replay):
    '''scouting_response takes in a previously loaded replay and returns the
    average interval in between scouting and battles for each player. This is
    a start on exploring responses to scouting.'''
    r = replay
    try:
        factors = sc2reader.constants.GAME_SPEED_FACTOR
        # scale = 16*factors[r.expansion][r.speed]
        # it appears the scale is always 22.4 in our dataset, despite documentation to the contrary
        scale = 22.4

        team1_scouting_states, team2_scouting_states = final_scouting_states(r)
        battles, harassing = battle_detector.buildBattleList(r)

        team1_scouting_frames = scouting_timeframe_list1(team1_scouting_states)
        team2_scouting_frames = scouting_timeframe_list1(team2_scouting_states)

        team1_avg = avg_interval_before_battle(team1_scouting_frames, battles, scale)
        team2_avg = avg_interval_before_battle(team2_scouting_frames, battles, scale)

        return team1_avg, team2_avg

    except:
        raise


def print_verification(replay):
    '''print_verification takes in a previously loaded replay from sc2reader
    and prints out information useful for verification. More specifically,
    it prints out the game time at which scouting states switch for each
    team/player'''
    r = replay

    try:
        team1_scouting_states, team2_scouting_states = final_scouting_states(r)

        frames = r.frames
        seconds = r.length.seconds
        team1_time_dict = to_time(team1_scouting_states, frames, seconds)
        team2_time_dict = to_time(team2_scouting_states, frames, seconds)

        print("---Team 1---")
        print_time(team1_time_dict)
        print("\n\n---Team 2---")
        print_time(team2_time_dict)

    except:
        print(replay.filename, "contains errors within scouting_detector")
        raise


def generate_fields(replay_file, map_path_data):
    """generate_fields takes in a filename of a replay, loads it and gathers necessary
    statistics, and returns the statistics in a tuple. It is to be used to write
    these stats to a csv."""
    try:
        # convert replay file name to path
        replay_file_path = "/Accounts/awb/pattern-analysis/starcraft/replays" + "/" + replay_file
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

        team1_freq, team1_cat, team2_freq, team2_cat, winner = scouting_detector.scouting_freq_and_cat(replay,
                                                                                                       map_path_data)
        team1_rank, team1_rel_rank, team2_rank, team2_rel_rank = ranking_stats(replay)
        team1_cps, team1_peace_rate, team1_battle_rate, team2_cps, team2_peace_rate, team2_battle_rate = \
            control_groups.control_group_stats(replay)
        team1_rel_freq = team1_freq - team2_freq
        team2_rel_freq = team2_freq - team1_freq

        team1_rel_cps = team1_cps - team2_cps
        team2_rel_cps = team2_cps - team1_cps
        team1_rel_pr = team1_peace_rate - team2_peace_rate
        team2_rel_pr = team2_peace_rate - team1_peace_rate
        team1_rel_br = team1_battle_rate - team2_battle_rate
        team2_rel_br = team2_battle_rate - team1_battle_rate

        # changing actions per minute to actions per second to match other data
        team1_aps = replay.players[0].avg_apm / 60
        team2_aps = replay.players[1].avg_apm / 60
        team1_rel_aps = (replay.players[0].avg_apm - replay.players[1].avg_apm) / 60
        team2_rel_aps = (replay.players[1].avg_apm - replay.players[0].avg_apm) / 60
        team1_uid = replay.players[0].detail_data['bnet']['uid']
        team2_uid = replay.players[1].detail_data['bnet']['uid']
        # creating the fields based on who won
        if winner == 1:
            fields = (game_id, team1_uid, team1_cat, team1_rank, team1_rel_rank,
                      team1_freq, team1_rel_freq, team1_aps, team1_rel_aps,
                      team1_cps, team1_rel_cps, team1_peace_rate, team1_rel_pr,
                      team1_battle_rate, team1_rel_br, 1,
                      game_id, team2_uid, team2_cat, team2_rank, team2_rel_rank,
                      team2_freq, team2_rel_freq, team2_aps, team2_rel_aps,
                      team2_cps, team2_rel_cps, team2_peace_rate, team2_rel_pr,
                      team2_battle_rate, team2_rel_br, 0,
                      replay.map_name)
        elif winner == 2:
            fields = (game_id, team1_uid, team1_cat, team1_rank, team1_rel_rank,
                      team1_freq, team1_rel_freq, team1_aps, team1_rel_aps,
                      team1_cps, team1_rel_cps, team1_peace_rate, team1_rel_pr,
                      team1_battle_rate, team1_rel_br, 0,
                      game_id, team2_uid, team1_cat, team2_rank, team2_rel_rank,
                      team2_freq, team2_rel_freq, team2_aps, team2_rel_aps,
                      team2_cps, team2_rel_cps, team2_peace_rate, team2_rel_pr,
                      team2_battle_rate, team2_rel_br, 1,
                      replay.map_name)
        return fields
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


def writeToCsv():
    with open("scouting_stats.csv", 'w', newline='') as fp:
        events_out = csv.DictWriter(fp, fieldnames=["GameID", "UID", "ScoutingCategory",
                                                    "Rank", "RelRank", "ScoutingFrequency",
                                                    "RelScoutingFrequency", "APS", "RelAPS",
                                                    "CPS", "RelCPS", "PeaceRate", "RelPeaceRate",
                                                    "BattleRate", "RelBattleRate", "Win"])
        events_out.writeheader()
        for map_name, replays in maps.items():
            print("loading path data for map", map_name, "which has", len(replays), "replays")
            pool = Pool(min(cpu_count(), 10))
            map_path_data = load_path_data(map_name)
            results = pool.starmap(generate_fields, zip(replays, repeat(map_path_data)))
            pool.close()
            pool.join()
            for fields in results:
                if fields:  # generateFields will return None for invalid replays
                    # writing 1 line to the csv for each player and their respective stats
                    events_out.writerow({"GameID": fields[0], "UID": fields[1],
                                         "ScoutingCategory": fields[2], "Rank": fields[3],
                                         "RelRank": fields[4], "ScoutingFrequency": fields[5],
                                         "RelScoutingFrequency": fields[6], "APS": fields[7],
                                         "RelAPS": fields[8], "CPS": fields[9], "RelCPS": fields[10],
                                         "PeaceRate": fields[11], "RelPeaceRate": fields[12],
                                         "BattleRate": fields[13], "RelBattleRate": fields[14],
                                         "Win": fields[15]})
                    events_out.writerow({"GameID": fields[16], "UID": fields[17],
                                         "ScoutingCategory": fields[18], "Rank": fields[19],
                                         "RelRank": fields[20], "ScoutingFrequency": fields[21],
                                         "RelScoutingFrequency": fields[22], "APS": fields[23],
                                         "RelAPS": fields[24], "CPS": fields[25], "RelCPS": fields[26],
                                         "PeaceRate": fields[27], "RelPeaceRate": fields[28],
                                         "BattleRate": fields[29], "RelBattleRate": fields[30],
                                         "Win": fields[31]})


def test():
    import time
    from sc2reader.engine.plugins import SelectionTracker, APMTracker
    from selection_plugin import ActiveSelection
    from base_plugins import BaseTracker
    from map_path_generation import load_path_data, get_all_possible_names
    from replay_verification import map_pretty_name_to_file
    sc2reader.engine.register_plugin(APMTracker())
    sc2reader.engine.register_plugin(SelectionTracker())
    sc2reader.engine.register_plugin(ActiveSelection())
    sc2reader.engine.register_plugin(BaseTracker())
    r = sc2reader.load_replay("/Accounts/awb/pattern-analysis/starcraft/replays/spawningtool_58796.SC2Replay")
    print(r.map_name)
    map_data = load_path_data(get_all_possible_names(map_pretty_name_to_file(r.map_name)))
    if not map_data:
        print("no map path data for map", r.map_name)
        return
    print("path data loaded")
    ts = time.time()
    times = scouting_times(r, 2, map_data)
    print(r.players[0].play_race, r.players[1].play_race)
    print("team 1:", [time / 22.4 for time in times[0]])
    print("team 2:", [time / 22.4 for time in times[1]])
    print("processing", "replay", "took", time.time() - ts, "sec")
    print("replay was",r.real_length)


if __name__ == "__main__":
    test2()
