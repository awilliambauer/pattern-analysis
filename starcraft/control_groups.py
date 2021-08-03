# A program to calculate statistics on control group usage for players
# in StarCraft 2
# Alison Cameron
# July 2020

import sc2reader
import battle_detector
import os
import numpy as np

# the main function, control_group_stats(replay) takes in a replay previously
# loaded by sc2reader. Wherever this replay is loaded, the following plugins
# must be registered in order for these functions to operate correctly
# from sc2reader.engine.plugins import SelectionTracker
# from selection_plugin import ActiveSelection
# sc2reader.engine.register_plugin(SelectionTracker())
# sc2reader.engine.register_plugin(ActiveSelection())

def commandsPerSecond(game_events, seconds_1, seconds_2):
    '''commandsPerSecond calculates the average commands per second in regards
    to control group usage for each player in a replay. commandsPerSecond takes
    in a list of all game events (accessed by replay.game_events) and the length
    of the game in seconds. commandsPerSecond returns this rate for each player.'''
    p1_count = 0
    p2_count = 0
    for event in game_events:
        if isinstance(event, sc2reader.events.game.ControlGroupEvent):
            if event.player.pid == 1:
                p1_count += 1
            elif event.player.pid == 2:
                p2_count += 1
    p1_cps = p1_count/seconds_1
    p2_cps = p2_count/seconds_2
    return p1_cps, p2_cps

def macroRates(game_events, battles, frames, seconds_1, seconds_2):
    '''macroRates computes the Macro (economic) control group selection rate for
    peacetime and battletime for each player. macroRates takes in the replay's
    game events, a list of battles returned by battle_detector.buildBattleList,
    the total frames of the game, and the length of the game in seconds.
    macroRates returns the peacetime and battletime macro selection rate for
    each player'''
    # list of units in each control group for each player
    cgrps = {1: {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]},
             2: {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}}
    peace_ct = {1: 0, 2: 0}
    battle_ct = {1: 0, 2: 0}
    peace_rb = {1: 0, 2: 0}
    battle_rb = {1: 0, 2: 0}
    battle_time_1, battle_time_2 = 0, 0
    peace_time_1, peace_time_2 = 0, 0
    

    cgrps_set = {1: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0},
             2: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}}
    cgrps_add = {1: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0},
             2: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}}
    cgrps_get = {1: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0},
             2: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}}


    # building up peace and battle macro command counts for each player
    for event in game_events:
        if isinstance(event, sc2reader.events.game.ControlGroupEvent):
            control_group = event.control_group
            team = event.player.pid
            # if not players
            if team != 1 and team != 2:
                continue
            # new control group
            if isinstance(event, sc2reader.events.game.SetControlGroupEvent):
                list1 = sorted(cgrps[team][control_group])
                list2 = sorted(event.active_selection)
                if len(cgrps[team][control_group]) != 0 and list1 != list2:
                    if battle_detector.duringBattle(event.frame, battles, margin = 0):
                        battle_rb[team] += 1
                    else:
                        peace_rb[team] += 1
                cgrps[team][control_group] = event.active_selection
                cgrps_set[team][control_group] += 1
              

            # adding units to an existing control group
            elif isinstance(event, sc2reader.events.game.AddToControlGroupEvent):
                if len(cgrps[team][control_group]) != 0:
                    if battle_detector.duringBattle(event.frame, battles, margin = 0):
                        battle_rb[team] += 1
                    else:
                        peace_rb[team] += 1
                for unit in event.active_selection:
                    if not(unit in cgrps[team][control_group]):
                        cgrps[team][control_group].append(unit)
                cgrps_add[team][control_group] += 1                 

            # Selecting a pre-set control group
            elif isinstance(event, sc2reader.events.game.GetControlGroupEvent):
                if isMacro(cgrps[team][control_group]):
                    if battle_detector.duringBattle(event.frame, battles, margin = 0):
                        battle_ct[team] += 1
                    else:
                        peace_ct[team] += 1
                cgrps_get[team][control_group] += 1 

    # Calculating total peacetime and total battletime (in seconds) for the game
    for battle in battles:
        # player 1
        starttime_1 = (battle[0]/frames)*seconds_1
        endtime_1 = (battle[1]/frames)*seconds_1
        duration_1 = endtime_1 - starttime_1
        battle_time_1 += duration_1
        # player 2
        starttime_2 = (battle[0]/frames)*seconds_2
        endtime_2 = (battle[1]/frames)*seconds_2
        duration_2 = endtime_2 - starttime_2
        battle_time_2 += duration_2
    peace_time_1 = seconds_1 - battle_time_1
    peace_time_2 = seconds_2 - battle_time_2

    if battle_time_1 == 0 or battle_time_2 == 0 or \
        peace_time_1 == 0 or peace_time_2 == 0:
        print("battle time or peace time is zero")
        # replay is not valid to calculate proper rates
        # raise RuntimeError()

    # Calculating rates
    p1_peace_rate = peace_ct[1]/peace_time_1
    p2_peace_rate = peace_ct[2]/peace_time_2
    p1_battle_rate = battle_ct[1]/battle_time_1
    p2_battle_rate = battle_ct[2]/battle_time_2

    p1_peace_rb = peace_rb[1]/peace_time_1
    p2_peace_rb = peace_rb[2]/peace_time_2
    p1_battle_rb = battle_rb[1]/battle_time_1
    p2_battle_rb = battle_rb[2]/battle_time_2

    p1_cps_cg = list(np.array(list(cgrps_set[1].values()))/seconds_1) + \
                list(np.array(list(cgrps_add[1].values()))/seconds_1) + \
                list(np.array(list(cgrps_get[1].values()))/seconds_1)
    p2_cps_cg = list(np.array(list(cgrps_set[2].values()))/seconds_2) + \
                list(np.array(list(cgrps_add[2].values()))/seconds_2) + \
                list(np.array(list(cgrps_get[2].values()))/seconds_2)

    # print(p1_cps_cg, "\n", p2_cps_cg, "\n")

    return p1_peace_rate, p1_battle_rate, p2_peace_rate, p2_battle_rate, \
    p1_peace_rb, p2_peace_rb, p1_battle_rb, p2_battle_rb, p1_cps_cg, p2_cps_cg

def isMacro(cgrp):
    '''isMacro returns true if all of the units in a control group
    are economic units. Returns false if any units in a control group
    are military units. isMacro takes in a list of units, known as
    a control group.'''
    for unit in cgrp:
        if unit.is_army:
            return False
    return True

def control_group_stats(replay):
    '''control_group_stats is the main function for this script. It takes in
    a pre-loaded replay using sc2reader and returns the commands per second,
    peacetime macro selection rate, and battletime macro selection rate
    for each player.'''
    r = replay

    p1_cps, p2_cps = commandsPerSecond(r.game_events, \
        replay.humans[0].seconds_played/1.4, replay.humans[1].seconds_played/1.4)
    battles, harassing = battle_detector.buildBattleList(r)
    p1_peace_rate, p1_battle_rate, p2_peace_rate, p2_battle_rate, \
    p1_peace_rb, p2_peace_rb, p1_battle_rb, p2_battle_rb, \
    p1_cps_cg, p2_cps_cg = macroRates(r.game_events, battles, r.frames, \
        replay.humans[0].seconds_played/1.4, replay.humans[1].seconds_played/1.4)

    return p1_cps, p1_peace_rate, p1_battle_rate, p2_cps, p2_peace_rate, p2_battle_rate, \
    p1_peace_rb, p2_peace_rb, p1_battle_rb, p2_battle_rb, p1_cps_cg, p2_cps_cg
