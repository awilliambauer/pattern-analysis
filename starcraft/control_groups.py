#A program to calculate statistics on control group usage for players
#in StarCraft 2

import sc2reader
from sc2reader.engine.plugins import SelectionTracker
from selection_plugin import ActiveSelection
import battle_detector
import json
import os

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
    p1_cgrps = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
    p2_cgrps = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
    p1_peace_ct, p2_peace_ct = 0, 0
    p1_battle_ct, p2_battle_ct = 0, 0
    peace_time, battle_time = 0, 0

    #building up peace and battle macro command counts for each player
    for event in game_events:
        if isinstance(event, sc2reader.events.game.ControlGroupEvent):
            #new control group
            if isinstance(event, sc2reader.events.game.SetControlGroupEvent):
                if event.player.pid == 1:
                    p1_cgrps[event.control_group] = event.player.selection[10]
                elif event.player.pid == 2:
                    p2_cgrps[event.control_group] = event.player.selection[10]
            #adding units to an existing control group
        elif isinstance(event, sc2reader.events.game.AddToControlGroupEvent):
                units = event.player.selection[10]
                if event.player.pid == 1:
                    for unit in units:
                        if not(unit in p1_cgrps[event.control_group]):
                            p1_cgrps[event.control_group].append(unit)
                elif event.player.pid == 2:
                    for unit in units:
                        if not(unit in p2_cgrps[event.control_group]):
                            p2_cgrps[event.control_group].append(unit)
            #Selecting a pre-set control group
        elif isinstance(event, sc2reader.events.game.GetControlGroupEvent):
                if event.player.pid == 1 and isMacro(p1_cgrps[event.control_group]):
                    if battle_detector.duringBattle(event.frame, battles):
                        p1_battle_ct += 1
                    else:
                        p1_peace_ct += 1
                elif event.player.pid == 1 and isMacro(p2_cgrps[event.control_group]):
                    if battle_detector.duringBattle(event.frame, battles):
                        p2_battle_ct += 1
                    else:
                        p2_peace_ct += 1

    #Calculating total peacetime and total battletime (in seconds) for the game
    for battle in battles:
        starttime = (battle[0]/frames)*seconds
        endtime = (battle[1]/frames)*seconds
        duration = endtime - starttime
        battle_time += duration
    peace_time = seconds - battle_time

    #Calculating rates
    p1_peace_rate = p1_peace_ct/peace_time
    p2_peace_rate = p2_peace_ct/peace_time
    p1_battle_rate = p1_battle_ct/battle_time
    p2_battle_rate = p2_battle_ct/battle_time

    #added small fudge factors to deal with either rate being 0
    p1_ratio = (p1_peace_rate + 0.001)/(p1_battle_rate + 0.001)
    p2_ratio = (p2_peace_rate + 0.001)/(p2_battle_rate + 0.001)

    return p1_ratio, p2_ratio


def isMacro(cgrp):
    for unit in cgrp:
        if unit.is_army:
            return False
    return True


def control_group_stats(replay):
    r = replay

    p1_cps, p2_cps = commandsPerSecond(r.game_events, r.length.seconds)
    battles = battle_detector.buildBattleList(r)
    p1_ratio, p2_ratio = macroRatio(r.game_events, battles, r.frames, r.length.seconds)

    return p1_cps, p1_ratio, p2_cps, p2_ratio
