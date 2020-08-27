# A script to track battles and instances of harassing in StarCraft 2
# Alison Cameron
# July 2020

import sc2reader
from collections import defaultdict
import os

def buildBattleList(replay):
    '''buildBattleList takes in a replay object previously loaded by sc2reader
    and returns a list of battles and instances of harassing, where each of
    these conflicts is a tuple containing the frame that the battle began,
    the frame that the battle ended, and the location of the battle.

    An encounter between teams/players is considered a battle if greater than 10%
    of either team's army value is destroyed.
    An encounter between teams/players is considered harassment if less than 10%
    of either team's army value is destroyed, but at least 4 units were destroyed -
    of which one must be a non-defensive building OR at least half of the units
    must be workers.'''
    # unable to compute battles for pre 2.0.7
    if replay.build < 25446:
        return 0

    # initializing the list of battles and harssing, where each instance is a tuple that contains
    # the frame that the engagement began and the frame that the engagement ended
    battles = []
    harassing = []

    MAX_DEATH_SPACING_FRAMES = 160.0 # max number of frames between deaths for
    # deaths to be considered part of the same engagement

    BATTLE = 0.1 # threshold of army that must be killed in order
    # for the engagement to be considered a battle

    owned_units = []
    killed_units = [] # used to determine death of units as an identifier for battles
    for obj in replay.objects.values():
        if obj.owner is not None:
            if (replay.build >= 25446 or obj.is_army) and obj.minerals is not None and obj.finished_at is not None:
                owned_units.append(obj)
                if obj.died_at is not None:
                    killed_units.append(obj)

    # sorted by frame each unit died at
    killed_units = sorted(killed_units, key=lambda obj: obj.died_at)

    engagements = []
    dead_units = []
    current_engagement = None

    # building the list of engagements
    for unit in killed_units:
        if(unit.killing_player is not None or replay.build<25446) and (unit.minerals + unit.vespene > 0):
            dead = unit
            dead_units.append(dead)
            # create a new engagement
            if current_engagement is None or (dead.died_at - current_engagement[2] > MAX_DEATH_SPACING_FRAMES):
                current_engagement = [[dead], dead.died_at, dead.died_at]
                engagements.append(current_engagement)
            # add information to current engagement
            else:
                current_engagement[0].append(dead)
                current_engagement[2] = dead.died_at

    # calculating the loss for each engagement and adding it to the list of
    # battles if greater than 10% of a team's army value is destroyed
    for engagement in engagements:
        killed = defaultdict(int)
        units_at_start = defaultdict(int)
        born_during_battle = defaultdict(int)
        killed_econ = defaultdict(int)
        # calculating loss for each team
        for dead in engagement[0]:
            deadvalue = dead.minerals + dead.vespene
            if dead.is_army:
                killed[dead.owner.team] += deadvalue
            elif replay.build >= 25446:
                killed_econ[dead.owner.team] += deadvalue

        # differentiating between units that were born before vs. during battle
        for unit in owned_units:
            # units born before battle
            if unit.finished_at < engagement[1]:
                units_at_start[unit.owner.team] += unit.minerals + unit.vespene
            # units born during battle
            elif unit.finished_at >= engagement[1] and unit.finished_at < engagement[2]:
                born_during_battle[unit.owner.team] += unit.minerals + unit.vespene

        # deciding whether an engagement meets the threshold to be a battle
        if engagement[2] > engagement[1]:
            defense_buildings = ["Bunker", "MissileTurret", "PlanetaryFortress",
                                "PhotonCannon", "ShieldBattery", "SpineCrawler",
                                "SporeCrawler"]
            for team in replay.teams:
                total_units = len(engagement[0])
                worker_count = 0
                non_defense = False
                for unit in engagement[0]:
                    if unit.is_worker:
                        worker_count += 1
                    elif not(unit.name in defense_buildings):
                        non_defense = True
                perc_worker = worker_count/total_units
                perc_died = float(killed[team] + killed_econ[team])/(units_at_start[team] + born_during_battle[team])
                if(units_at_start[team] > 0) and (perc_died >= BATTLE):
                    # greater than 10% of a team's army value was killed, add to battles
                    tuple = (engagement[1], engagement[2])
                    if tuple not in battles:
                        battles.append(tuple)
                elif(units_at_start[team] > 0) and (perc_died < BATTLE) and (total_units >= 4) and (perc_worker >= 0.5 or non_defense):
                    tuple = (engagement[1], engagement[2])
                    if tuple not in harassing:
                        harassing.append(tuple)

    battle_dict = initializeDictionary(battles)
    harassing_dict = initializeDictionary(harassing)
    t_events = replay.tracker_events
    # compiling a list of locations for all battles and harassing
    for event in t_events:
        if isinstance(event, sc2reader.events.tracker.UnitDiedEvent):
            frame = event.frame
            location = event.location
            for battle in battles:
                if frame >= battle[0] and frame <= battle[1]:
                    battle_dict[battle].append(location)
                    break
            for harass in harassing:
                if frame >= harass[0] and frame <= harass[1]:
                    harassing_dict[harass].append(location)
                    break

    # averaging the locations for each battle and harassing
    new_battles = averageLocations(battles, battle_dict)
    new_harassing = averageLocations(harassing, harassing_dict)

    return new_battles, new_harassing

def initializeDictionary(list):
    '''initializeDictionary returns a dictionary where the items in the list
    are keys and the values are empty lists. This is used to aid buildBattleList.'''
    dict = {}
    for item in list:
        dict[item] = []
    return dict

def averageLocations(list, dict):
    '''averageLocations is used to aid buildBattleList by averaging
    a list of locations for each battle in a dictionary. It returns
    a list of conflicts where each conflict is a tuple of the format
    (start frame, end frame, (location - x, location -y))'''
    new_list = []
    for item in list:
        locations = dict[item]
        x_vals = 0
        y_vals = 0
        for place in locations:
            x_vals += place[0]
            y_vals += place[1]
        x_avg = int(x_vals/len(locations))
        y_avg = int(y_vals/len(locations))
        new_tuple = (item[0], item[1], (x_avg, y_avg))
        new_list.append(new_tuple)
    return new_list

def duringBattle(frame, battles):
    '''duringBattle returns true if a frame takes place during a battle.
    The parameters are a frame of the game and a list of battles returned by
    buildBattleList.'''
    for battle in battles:
        if frame >= battle[0] and frame <= battle[1]:
            return True

def toTime(battles, frames, seconds):
    '''toTime converts the frames of each battle into standard time format.
    This is intended to aid in the verification of when battles occur when rewatching
    a processed replay. toTime takes in a list of battles returned by buildBattleList,
    the total frames in a replay, and the length of the game in seconds. toTime
    returns a list of strings of nicely formatted times of battles.'''
    timeList = []

    for i in range(len(battles)-1):
        startframe = battles[i][0]
        endframe = battles[i][1]
        starttime = (startframe/frames)*seconds
        endtime = (endframe/frames)*seconds
        startminStr = "{:2d}".format(int(starttime//60))
        startsecStr = "{:05.2f}".format(starttime%60)
        starttimeStr = startminStr + ":" + startsecStr
        endminStr = "{:2d}".format(int(endtime//60))
        endsecStr = "{:05.2f}".format(endtime%60)
        endtimeStr = endminStr + ":" + endsecStr
        battletime = "Battle #{} starts at {} and ends at {}".format(i, starttimeStr, endtimeStr)
        timeList.append(battletime)
    return timeList

def printTime(timeList):
    '''printTime neatly prints the list of strings returned by toTime'''
    for battletime in timeList:
        print(battletime)
