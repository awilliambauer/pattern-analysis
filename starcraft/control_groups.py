# A program to calculate statistics on control group usage for players
# in StarCraft 2
# Alison Cameron
# July 2020

import sc2reader
import battle_detector
import os

# the main function, control_group_stats(replay) takes in a replay previously
# loaded by sc2reader. Wherever this replay is loaded, the following plugins
# must be registered in order for these functions to operate correctly
# from sc2reader.engine.plugins import SelectionTracker
# from selection_plugin import ActiveSelection
# sc2reader.engine.register_plugin(SelectionTracker())
# sc2reader.engine.register_plugin(ActiveSelection())

def commandsPerSecond(game_events, seconds):
    '''commandsPerSecond calculates the average commands per second in regards
    to control group usage for each player in a replay. commandsPerSecond takes
    in a list of all game events (accessed by replay.game_events) and the length
    of the game in seconds. commandsPerSecond returns this rate for each player.'''
    length = seconds
    p1_count = 0
    p2_count = 0
    for event in game_events:
        if isinstance(event, sc2reader.events.game.ControlGroupEvent):
            if event.player.pid == 1:
                p1_count += 1
            elif event.player.pid == 2:
                p2_count += 1
    p1_cps = p1_count/length
    p2_cps = p2_count/length
    return p1_cps, p2_cps

def macroRates(game_events, battles, frames, seconds):
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
    peace_time, battle_time = 0, 0

    # building up peace and battle macro command counts for each player
    for event in game_events:
        if isinstance(event, sc2reader.events.game.ControlGroupEvent):
            control_group = event.control_group
            team = event.player.pid
            # new control group
            if isinstance(event, sc2reader.events.game.SetControlGroupEvent):
                cgrps[team][control_group] = event.player.selection[10]
                # if isMacro(cgrps[team][control_group]):
                #     if battle_detector.duringBattle(event.frame, battles):
                #         battle_ct[team] += 1
                #     else:
                #         peace_ct[team] += 1

            # adding units to an existing control group
            elif isinstance(event, sc2reader.events.game.AddToControlGroupEvent):
                units = event.player.selection[10]
                for unit in units:
                    if not(unit in cgrps[team][control_group]):
                        cgrps[team][control_group].append(unit)
                # if isMacro(cgrps[team][control_group]):
                #     if battle_detector.duringBattle(event.frame, battles):
                #         battle_ct[team] += 1
                #     else:
                #         peace_ct[team] += 1

            # Selecting a pre-set control group
            elif isinstance(event, sc2reader.events.game.GetControlGroupEvent):
                if isMacro(cgrps[team][control_group]):
                    if battle_detector.duringBattle(event.frame, battles):
                        battle_ct[team] += 1
                    else:
                        peace_ct[team] += 1

    # Calculating total peacetime and total battletime (in seconds) for the game
    for battle in battles:
        starttime = (battle[0]/frames)*seconds
        endtime = (battle[1]/frames)*seconds
        duration = endtime - starttime
        battle_time += duration
    peace_time = seconds - battle_time

    if battle_time == 0 or peace_time == 0:
        print("battle time or peace time is zero")
        # replay is not valid to calculate proper rates
        raise RuntimeError()

    # Calculating rates
    p1_peace_rate = peace_ct[1]/peace_time
    p2_peace_rate = peace_ct[2]/peace_time
    p1_battle_rate = battle_ct[1]/battle_time
    p2_battle_rate = battle_ct[2]/battle_time

    return p1_peace_rate, p1_battle_rate, p2_peace_rate, p2_battle_rate

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

    p1_cps, p2_cps = commandsPerSecond(r.game_events, r.real_length.total_seconds())
    battles, harassing = battle_detector.buildBattleList(r)
    p1_peace_rate, p1_battle_rate, p2_peace_rate, p2_battle_rate = macroRates(r.game_events, battles, r.frames, r.real_length.total_seconds())

    return p1_cps, p1_peace_rate, p1_battle_rate, p2_cps, p2_peace_rate, p2_battle_rate
