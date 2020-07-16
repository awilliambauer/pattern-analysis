#A program to calculate statistics on control group usage for players
#in StarCraft 2

import sc2reader
import battle_detector
import os

#the main function, control_group_stats(replay) takes in a replay previously
#loaded by sc2reader. Wherever this replay is loaded, the following plugins
#must be registered in order for these functions to operate correctly
from sc2reader.engine.plugins import SelectionTracker
from selection_plugin import ActiveSelection
sc2reader.engine.register_plugin(SelectionTracker())
sc2reader.engine.register_plugin(ActiveSelection())

def commandsPerSecond(game_events, seconds):
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

def macroRatio(game_events, battles, frames, seconds):
    cgrps = {1: {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]},
             2: {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}}
    peace_ct = {1: 0, 2: 0}
    battle_ct = {1: 0, 2: 0}
    peace_time, battle_time = 0, 0

    #building up peace and battle macro command counts for each player
    for event in game_events:
        if isinstance(event, sc2reader.events.game.ControlGroupEvent):
            control_group = event.control_group
            team = event.player.pid
            #new control group
            if isinstance(event, sc2reader.events.game.SetControlGroupEvent):
                cgrps[team][control_group] = event.player.selection[10]
                # if isMacro(cgrps[team][control_group]):
                #     if battle_detector.duringBattle(event.frame, battles):
                #         battle_ct[team] += 1
                #     else:
                #         peace_ct[team] += 1

            #adding units to an existing control group
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

            #Selecting a pre-set control group
            elif isinstance(event, sc2reader.events.game.GetControlGroupEvent):
                if isMacro(cgrps[team][control_group]):
                    if battle_detector.duringBattle(event.frame, battles):
                        battle_ct[team] += 1
                    else:
                        peace_ct[team] += 1

    #Calculating total peacetime and total battletime (in seconds) for the game
    for battle in battles:
        starttime = (battle[0]/frames)*seconds
        endtime = (battle[1]/frames)*seconds
        duration = endtime - starttime
        battle_time += duration
    peace_time = seconds - battle_time

    if battle_time == 0 or peace_time == 0:
        print("battle time or peace time is zero")
        #replay is not valid to calculate a ratio
        raise RuntimeError()

    #Calculating rates
    p1_peace_rate = peace_ct[1]/peace_time
    p2_peace_rate = peace_ct[2]/peace_time
    p1_battle_rate = battle_ct[1]/battle_time
    p2_battle_rate = battle_ct[2]/battle_time

    return p1_peace_rate, p1_battle_rate, p2_peace_rate, p2_battle_rate

def isMacro(cgrp):
    for unit in cgrp:
        if unit.is_army:
            return False
    return True

def control_group_stats(replay):
    r = replay

    p1_cps, p2_cps = commandsPerSecond(r.game_events, r.length.seconds)
    battles = battle_detector.buildBattleList(r)
    p1_peace_rate, p1_battle_rate, p2_peace_rate, p2_battle_rate = macroRatio(r.game_events, battles, r.frames, r.length.seconds)

    return p1_cps, p1_peace_rate, p1_battle_rate, p2_cps, p2_peace_rate, p2_battle_rate
