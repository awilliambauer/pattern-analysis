# Alison Cameron
# June 2020
# A program to detect scouting behavior of players in StarCraft 2

import sc2reader
import math
import statistics
import battle_detector

def buildEventLists(tracker_events, game_events):
    '''buildEventLists is used to build up a list of events related to
    scouting behavior. It takes in a replay's tracker events and game events.
    It returns one list of all relevant events.'''

    events = []
    ability_names = ["LandCommandCenter", "LandOrbitalCommand", "RightClick", "Attack"]
    base_names = ["Hatchery", "Lair", "Hive", "Nexus", "CommandCenter", "CommandCenterFlying",
                        "OrbitalCommand", "OrbitalCommandFlying","PlanetaryFortress"]
    team1 = 1
    team2 = 2
    start1 = False
    start2 = False
    team1_count = 0
    team2_count = 0
    for t_event in tracker_events:
        # checking for the creation of new bases
        if isinstance(t_event, sc2reader.events.tracker.UnitInitEvent) and (t_event.unit.name in base_names):
            events.append(t_event)
        # Adding information about units
        elif isinstance(t_event, sc2reader.events.tracker.UnitBornEvent) and (t_event.unit.is_army or t_event.unit.is_worker):
            events.append(t_event)
        # More information about unit positions
        elif isinstance(t_event, sc2reader.events.tracker.UnitPositionsEvent):
            events.append(t_event)
        # removing dead units
        elif isinstance(t_event, sc2reader.events.tracker.UnitDiedEvent) and (t_event.unit.is_army or t_event.unit.is_worker):
            events.append(t_event)

    for g_event in game_events:
        # filtering through camera events
        if isinstance(g_event, sc2reader.events.game.CameraEvent):
            events.append(g_event)
            if g_event.player:
                if g_event.player.pid == 1:
                    team1_count += 1
                elif g_event.player.pid == 2:
                    team2_count += 1
            else:
                raise RuntimeError()
        # account for moving terran bases and moving units
        elif isinstance(g_event, sc2reader.events.game.TargetUnitCommandEvent) and (g_event.ability_name in ability_names):
            events.append(g_event)
        elif isinstance(g_event, sc2reader.events.game.TargetPointCommandEvent) and (g_event.ability_name in ability_names):
            events.append(g_event)


    # if either team has 0 camera events, scouting behavior cannot be detected and
    # the replay is invalid
    if team1_count == 0 or team2_count == 0:
        raise RuntimeError()

    sorted_events = sorted(events, key=lambda e: e.frame)
    return sorted_events

def initializeScoutingDictionaries(frames):
    dicts = {1: {}, 2: {}}
    for i in range(1, 3):
        for j in range(1, frames+1):
            dicts[i][j] = ["", []]
    return dicts

def buildScoutingDictionaries(events, objects, frames):
    '''buildScoutingDictionaries returns dictionaries for each player where the
    keys are the frame and the value is the state of scouting. "No scouting"
    indicates the team/player is not looking at any bases, "Scouting themself"
    indicates the team/player is looking at their own base, and
    "Scouting opponent" indicates the team/player is looking at their opponent's
    base. buildScoutingDictionaries takes in a list of filtered events returned
    by buildEventLists.'''

    team1 = 1
    team2 = 2

    scouting_states = initializeScoutingDictionaries(frames)

    # Dictionaries for each team of the locations of bases where the keys are unit ids
    # and the values are locations (as tuples of (x, y) coordinates)
    bases = {1: {}, 2: {}}
    og_bases = {1: {}, 2: {}}
    # Add starting bases
    base_names = ["Hatchery", "Lair", "Hive", "Nexus", "CommandCenter", "CommandCenterFlying",
                        "OrbitalCommand", "OrbitalCommandFlying","PlanetaryFortress"]
    terran_bases = base_names[4:]
    for i in range(1, 3):
        start_base = [u for u in objects if u.name in base_names and u.owner != None and u.owner.pid == i and u.finished_at == 0][0]
        bases[i][start_base.id] = start_base.location
        og_bases[i][start_base.id] = start_base.location

    # Dictionaries for each team of the active units where the keys are the unit ids
    # and the values are locations (as tuples of (x, y) coordinates)
    active_units = {1: {}, 2: {}}

    # Used for updating the scouting dictionaries
    prev_states = {1: "Viewing themself", 2: "Viewing themself"}
    prev_frames = {1: 0, 2: 0}

    first_instance = {1: True, 2: True}

    # iterating through events in order
    for event in events:
        i = event.frame
        # accounting for new bases
        if isinstance(event, sc2reader.events.tracker.UnitInitEvent):
            cur_team = event.control_pid
            bases[cur_team][event.unit_id] = event.location
        # adding new units to the list of active units
        elif isinstance(event, sc2reader.events.tracker.UnitBornEvent):
            cur_team = event.control_pid
            active_units[cur_team][event.unit_id] = event.location

        # updating unit positions
        elif isinstance(event, sc2reader.events.tracker.UnitPositionsEvent):
            for unit in event.units.keys():
                cur_team = unit.owner.pid
                location = event.units[unit]
                active_units[cur_team][unit.id] = location

        # removing dead units
        elif isinstance(event, sc2reader.events.tracker.UnitDiedEvent):
            cur_team = event.unit.owner.pid
            if event.unit_id in active_units[cur_team]:
                active_units[cur_team].pop(event.unit_id)

        # accounting for Terran bases moving, updating unit positions, and the first instance of scouting
        elif isinstance(event, sc2reader.events.game.TargetUnitCommandEvent) or isinstance(event, sc2reader.events.game.TargetPointCommandEvent):
            cur_team = event.player.pid
            # moving Terran bases
            if "Point" in event.name and event.ability_name in ["LandCommandCenter", "LandOrbitalCommand"]:
                cur_selection = event.player.selection[10]
                for unit in cur_selection:
                    if unit.name in terran_bases:
                        bases[cur_team][unit.id] = event.location
                        break
            # updating unit positions and checking for the first instance of scouting
            elif event.ability_name in ["RightClick", "Attack"]:
                if "Unit" in event.name:
                    active_units[cur_team][event.target_unit_id] = event.location
                # checking for the first instance of scouting - units ordered to
                # the opponent's base
                if first_instance[cur_team]:
                    if cur_team == 1:
                        opp_team = 2
                    elif cur_team == 2:
                        opp_team = 1
                    target_location = event.location
                    if withinDistance(target_location, bases[opp_team], 75):
                        first_instance[cur_team] = False
                        scouting_states[cur_team][i][1].append("Sending units to the opponent's base")

        # checking camera events
        elif isinstance(event, sc2reader.events.game.CameraEvent):
            if event.player.is_observer or event.player.is_referee:
                continue

            cur_team = event.player.pid
            if cur_team == 1:
                opp_team = 2
            elif cur_team == 2:
                opp_team = 1
            camera_location = event.location
            # looking at their own base
            if withinDistance(camera_location, bases[cur_team], 25):
                scouting_states[cur_team] = updatePrevScoutStates(scouting_states[cur_team], i, prev_frames[cur_team], prev_states[cur_team])
                scouting_states[cur_team][i][0] = "Viewing themself"
                prev_frames[cur_team] = i
                prev_states[cur_team] = "Viewing themself"
            # looking at their opponent's main base
            elif withinDistance(camera_location, og_bases[opp_team], 25): #and withinDistance(camera_location, active_units[cur_team], 25):
                scouting_states[cur_team] = updatePrevScoutStates(scouting_states[cur_team], i, prev_frames[cur_team], prev_states[cur_team])
                location = (int(camera_location[0]), int(camera_location[1]))
                scouting_states[cur_team][i][0] = "Viewing opponent - main base"
                scouting_states[cur_team][i].append(location)
                prev_frames[cur_team] = i
                prev_states[cur_team] = "Viewing opponent - main base"
                first_instance[cur_team] = False
            # looking at their opponent's expansion bases
            elif withinDistance(camera_location, bases[opp_team], 25): #and withinDistance(camera_location, active_units[cur_team], 25):
                scouting_states[cur_team] = updatePrevScoutStates(scouting_states[cur_team], i, prev_frames[cur_team], prev_states[cur_team])
                location = (int(camera_location[0]), int(camera_location[1]))
                scouting_states[cur_team][i][0] = "Viewing opponent - expansions"
                scouting_states[cur_team][i].append(location)
                prev_frames[cur_team] = i
                prev_states[cur_team] = "Viewing opponent - expansions"
                first_instance[cur_team] = False
            # not looking at a base
            else:
                scouting_states[cur_team] = updatePrevScoutStates(scouting_states[cur_team], i, prev_frames[cur_team], prev_states[cur_team])
                scouting_states[cur_team][i][0] = "Viewing empty map space"
                prev_frames[cur_team] = i
                prev_states[cur_team] = "Viewing empty map space"

    return scouting_states[1], scouting_states[2]


def withinDistance(location, list, distance):
    '''withinDistance returns true if the input location is within a
    certain range of any location in the list. The location is a tuple
    and the list is a list of locations as tuples.'''
    loc_x, loc_y = location[0], location [1]
    keys = list.keys()
    for key in keys:
        loc = list[key]
        x, y = loc[0], loc[1]
        distance_apart = math.sqrt((loc_x - x)**2 + (loc_y - y)**2)
        if distance_apart <= distance:
            return True
    return False

def updatePrevScoutStates(scouting_dict, frame, prev_frame, prev_state):
    '''updatePrevScoutStates updates the input scouting dictionary from
    the prev_frame to frame with prev_state and returns the scouting
    dictionary back.'''
    if(prev_frame >= frame):
        return scouting_dict

    keys = scouting_dict.keys()
    i = prev_frame + 1
    while(i != frame):
        scouting_dict[i][0] = prev_state
        i += 1
    return scouting_dict

def checkFirstInstance(scouting_dict, scale):
    frame_jump45 = 45*int(scale)
    send_units = "Sending units to the opponent's base"
    keys = scouting_dict.keys()
    for key in keys:
        state = scouting_dict[key][1]
        # recorded instance of ordering units
        if send_units in state:
            # check if there is a regular scouting instance within the next 45 secs
            for i in range(key+1, frame_jump45+key+1):
                # make sure it doesn't throw a key error
                if i in keys:
                    if isScouting(scouting_dict[i]):
                        # if there is an instance of scouting, reset the original
                        scouting_dict[key][1].remove(send_units)
                        return scouting_dict
            # No instance that already corresponds to the unit ordering, consider it scouting
            # for 10 seconds
            frame_jump10 = 10*int(scale)
            for i in range(key, key+frame_jump10+1):
                if not(send_units in scouting_dict[i][1]):
                    scouting_dict[i][1].append(send_units)
            return scouting_dict
    # No ordering of units, return original dictionary
    return scouting_dict

def isScouting(frame_list):
    state = frame_list[0]
    events = frame_list[1]
    # viewing opponent but not harassing or engaged in battle
    if ("Viewing opponent" in state) and not("Harassing" in events) and not("Engaged in Battle" in events):
        return True
    # viewing anything, but sending units to the opponent's base for the first instance of scouting
    elif "Sending units to the opponent's base" in events:
        return True
    else:
        return False

def removeEmptyFrames(scouting_dict, frames):
    frame = frames
    initial_list = ["", []]
    state = scouting_dict[frame]
    while(state == initial_list):
        scouting_dict.pop(frame)
        frame -= 1
        state = scouting_dict[frame]
    return scouting_dict

def toTime(scouting_dict, frames, seconds):
    '''Creates and returns time-formatted dictionary of the time of game when
    a player's scouting state changes. Takes in a scouting dictionary, the total
    number of frames in the game, and the length of the game in seconds. Most
    useful for verification and testing.'''
    length = len(scouting_dict.keys())
    time_dict = {}


    state = scouting_dict[1]
    stateStr = state[0]
    if not(stateStr):
        stateStr = "No camera data"
    for event in state[1]:
        stateStr = stateStr + ", while" + event
    time = (1/frames)*(seconds)
    minStr = "{:2d}".format(int(time//60))
    secStr = "{:05.2f}".format(time%60)
    timeStr = minStr + ":" + secStr
    time_dict[timeStr] = stateStr

    frame = 2
    while(frame <= length):
        if scouting_dict[frame] != state:
            state = scouting_dict[frame]
            stateStr = state[0]
            if not(stateStr):
                stateStr = "No camera data"
            for event in state[1]:
                stateStr = stateStr + ", while " + event
            time = (frame/frames)*(seconds)
            minStr = "{:2d}".format(int(time//60))
            secStr = "{:05.2f}".format(time%60)
            timeStr = minStr + ":" + secStr
            time_dict[timeStr] = stateStr
        frame += 1
    return time_dict

def printTime(time_dict):
    '''Used to neatly print a time dictionary returned by toTime.'''
    keys = time_dict.keys()
    for key in keys:
        print(key, end = "")
        print(" -> ", end = "")
        print(time_dict[key])

def scouting_stats(scouting_dict):
    '''scouting_stats calculates the number of frames that a player
    spends scouting their opponent and the number of times they initiate scouting.
    It takes in a scouting dictionary returned by buildScoutingDictionaries
    or integrateBattles and returns the number of times and the number of frames.'''
    num_times = 0
    total_frames = 0
    scouting_frames = 0
    cur_scouting = False

    length = len(scouting_dict.keys())
    if isScouting(scouting_dict[1]):
        num_times += 1
        scouting_frames += 1
        cur_scouting = True
    total_frames += 1
    frame = 2
    while(frame < length):
        total_frames += 1
        if isScouting(scouting_dict[frame]):
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

    # calculating rates based on counts
    scouting_fraction = scouting_frames/total_frames
    scouting_rate = num_times/total_frames

    return num_times, scouting_frames

def integrateEngagements(scouting_dict, engagements, scale, replacement):
    '''integrateEngagements is used to cross-check a scouting dictionary with a
    list of engagements(battles or harassing). More specifically, it is used to
    avoid false positives of "scouting" an opponent during engagements.
    integrateEngagements takes in a scouting dictionary returned by
    buildScoutingDictionaries, a list of engagements returned by
    battle_detector.buildBattleList, and a string indicating what the false
    positives should be replaced by. It returns the updated scouting dictionary.'''
    frame_jump7 = 7*int(scale)
    keys = scouting_dict.keys()
    start_engagement = False
    start_frame = None
    end_engagement = False
    end_frame = None
    length = len(keys)
    frame = 1
    while frame < length:
        if battle_detector.duringBattle(frame, engagements):
            if not(replacement in scouting_dict[frame][1]):
                scouting_dict[frame][1].append(replacement)
            if not(isScouting(scouting_dict[frame-1])):
                start_engagement = True
                start_frame = frame
            if not(isScouting(scouting_dict[frame+1])):
                end_engagement = True
                end_frame = frame

        else:
            # reset scouting instances if they are within 7 seconds of a battle
            if start_engagement:
                for i in range(start_frame-frame_jump7, start_frame):
                    if i in keys and not(replacement in scouting_dict[i][1]):
                        scouting_dict[i][1].append(replacement)
            if end_engagement:
                for i in range(end_frame, end_frame+frame_jump7):
                    if i in keys and not(replacement in scouting_dict[i][1]):
                        scouting_dict[i][1].append(replacement)
            end_engagement = False
            start_engagement = False
        frame += 1
    return scouting_dict

def categorize_player(scouting_dict, frames):
    '''categorize_player is used to sort players based on their scouting
    behavior. The categories are numerical (1-4) and are returned by this
    function. The following is a summary of each category.

    1. No scouting - there are no scouting tags in a player's dictionary

    2. Only scouts in the beginning - the only existing scouting tags happen
    within the first 25% of the game

    3. Sporadic scouters - a player scouts past the first 25% of the game,
    but the time intervals between instances of scouting are inconsistent.
    Intervals are considered inconsistent if the standard deviation is
    greater than or equal to half of the mean of all intervals.

    4. Consistent scouters - a player scouts past the first 25% of the game,
    and the time intervals between instances of scouting are consistent.
    Intervals are considered consistent if the standard deviation is less
    than half of the mean.'''

    no_scouting = True
    beginning_scouting = True
    intervals = []

    keys = scouting_dict.keys()
    interval = 0
    after_first = False
    for key in keys:
        state = scouting_dict[key]
        if isScouting(state):
            after_first = True
            no_scouting = False
            if interval:
                intervals.append(interval)
            if key/frames > 0.25:
                beginning_scouting = False
            interval = 0

        else:
            if after_first:
                interval += 1
    # a non-scouter
    if no_scouting:
        category = 1
        return category

    # only scouts in the beginning
    if beginning_scouting:
        category = 2
        return category

    if len(intervals) >= 0 and len(intervals) <= 2:
        category = 3
        return category

    mean_interval = statistics.mean(intervals)
    stdev = statistics.pstdev(intervals)

    # Sporadic scouter
    if stdev/mean_interval >= 0.5:
        category = 3
    # Consistent scouter
    elif stdev/mean_interval < 0.5:
        category = 4

    return category

def avg_interval(scouting_dict, scale):
    '''avg_interval returns the average time interval (in seconds) between
        periods of scouting. It takes in a completes scouting dictionary
        returned by final_scouting_states(replay) as well as the scale,
        which is used to convert frames to seconds. The scale can be accessed
        by sc2reader.contants.GAME_SPEED_FACTOR[replay.expansion][replay.speed].'''
    new_scale = 16*scale
    intervals = []
    keys = scouting_dict.keys()
    after_first = False
    any_scouting = False
    start_interval = 0
    interval = 0

    for key in keys:
        state = scouting_dict[key]
        if isScouting(state):
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
        return start_interval/new_scale
    elif len(intervals) == 0 and not(any_scouting):
        # no scouting ocurred, return a flag to indicate this
        return -1
    elif len(intervals) > 0:
        # 2 or more instances of scouting ocurred, find the average interval between
        mean_interval = (statistics.mean(intervals))/new_scale
        return mean_interval

def scouting_timefrac_list(scouting_dict, frames):
    '''scouting_timefrac_list returns a list of instances where scouting
        occurs as fractions of the total gametime. It takes in a completed
        scouting dictionary returned by final_scouting_states(replay), as
        well as the total number of frames in the game (r.frames).'''
    time_fracs = []
    keys = scouting_dict.keys()
    cur_scouting = False
    for key in keys:
        state = scouting_dict[key]
        if isScouting(state):
            if not(cur_scouting):
                frac = key/frames
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
        if isScouting(state):
            if not(cur_scouting):
                time_frames.append(key)
            cur_scouting = True
        else:
            cur_scouting = False
    return time_frames

def hasInitialScouting(scouting_dict, frames, battles):
    keys = scouting_dict.keys()
    if len(battles) == 0:
        first_battle_start = frames
    else:
        first_battle_start = battles[0][0]
    for key in keys:
        state = scouting_dict[key]
        if isScouting(state) and key/frames <= 0.25 and key < first_battle_start:
            # True
            return 1
    # False
    return 0

def scoutsMainBase(scouting_dict):
    keys = scouting_dict.keys()
    mainbase_ct = 0
    scouting_ct = 0
    cur_scouting = False
    for key in keys:
        state = scouting_dict[key]
        if isScouting(state) and not(cur_scouting):
            scouting_ct += 1
            cur_scouting = True

            if "main base" in state[0]:
                mainbase_ct += 1

        else:
            cur_scouting = False

    if scouting_ct == 0:
        # No scouting, return a flag
        return -1

    if mainbase_ct/scouting_ct >= 0.5:
        # True
        return 1
    else:
        # False
        return 0

def scoutNewAreas(scouting_dict):
    locations = []
    keys = scouting_dict.keys()
    new_instance = True
    for key in keys:
        state = scouting_dict[key]
        if isScouting(state) and "Viewing opponent" in state[0] and len(state) == 3:
            new_instance = False
            locations.append(state[2])

    #1 or less instances of scouting, the player does not scout new areas consistently
    if len(locations) <= 1:
        return 0

    distances_apart = []
    for i in range(len(locations)):
        for j in range(i+1, len(locations)):
            first_loc = locations[i]
            second_loc = locations[j]
            first_x, first_y = first_loc[0], first_loc[1]
            second_x, second_y = second_loc[0], second_loc[1]
            x_diff, y_diff = abs(first_x-second_x), abs(first_y-second_y)
            distance_apart = math.sqrt(x_diff**2 + y_diff**2)
            distances_apart.append(distance_apart)

    avg_distance = statistics.mean(distances_apart)

    if avg_distance > 20:
        # True
        return 1
    else:
        # False
        return 0

def scoutBetweenBattles(scouting_dict, battles, frames):
    num_battles = len(battles)

    # not enough battles for a player to scout between them, return a flag
    if num_battles <= 1:
        return -1

    # creating a dictionary of frames of peacetime between battles
    peacetime_dict = {}
    peacetime_list = []
    for i in range(num_battles-1):
        battle = battles[i]
        next_battle = battles[i+1]
        peace_start = battle[1]+1
        peace_end = next_battle[0]-1
        peacetime_dict[(peace_start, peace_end)] = 0
        peacetime_list.append((peace_start, peace_end))

    # checking for scouting in between battles
    scouting_keys = scouting_dict.keys()
    peacetime_keys = peacetime_dict.keys()
    new_instance = True
    for s_key in scouting_keys:
        state = scouting_dict[s_key]
        # avoiding counting one instance of scouting more than once
        if new_instance and isScouting(state):
            new_instance = False
            # checking which period of peacetime the scouting occurs
            for p_key in peacetime_keys:
                if s_key >= p_key[0] and s_key <= p_key[1]:
                    # adding to the count of the correct peacetime
                    peacetime_dict[p_key] += 1
                    # breaking out of inner loop
                    break
        else:
            #If the scouting state changes or it is no longer during peacetime, then reset to a new instance
            if not(isScouting(state)) or not(battle_detector.duringBattle(s_key, peacetime_list)):
                new_instance = True

    # determine if there are enough scouting instances between battles
    nums_between = []
    for p_key in peacetime_keys:
        num = peacetime_dict[p_key]
        nums_between.append(num)

    avg_between = statistics.mean(nums_between)
    if avg_between >= 0.7:
        # at least one instance of scouting for 70% of peacetime periods
        return 1
    else:
        return 0

def final_scouting_states(replay):
    '''final_scouting_states is the backbone of scouting_detector.py. It does
        all of the error checking needed, as well as combines all functions to
        create completed scouting dictionaries - to then be used by other functions.
        It takes in a previously loaded replay object from sc2reader and returns
        completed scouting dictionaries for each player.'''
    r = replay

    if r.winner is None:
        print(r.filename, "has no winner information")
        raise RuntimeError()

    try:
        # some datafiles did not have a 'Controller' attribute
        if r.attributes[1]["Controller"] == "Computer" or r.attributes[2]["Controller"] == "Computer":
            print(r.filename, "is a player vs. AI game")
            raise RuntimeError()
    except:
        raise RuntimeError()

    if r.length.seconds < 300:
        print(r.filename, "is shorter than 5 minutes")
        raise RuntimeError()

    if len(r.players) != 2:
        print(r.filename, "is not a 1v1 game")
        raise RuntimeError()

    tracker_events = r.tracker_events
    game_events = r.game_events
    frames = r.frames
    scale = 16*sc2reader.constants.GAME_SPEED_FACTOR[r.expansion][r.speed]

    allEvents = buildEventLists(tracker_events, game_events)
    objects = r.objects.values()
    team1_scouting_states, team2_scouting_states = buildScoutingDictionaries(allEvents, objects, frames)

    battles, harassing = battle_detector.buildBattleList(r)
    team1_scouting_states = integrateEngagements(team1_scouting_states, battles, scale, "Engaged in Battle")
    team2_scouting_states = integrateEngagements(team2_scouting_states, battles, scale, "Engaged in Battle")
    team1_scouting_states = integrateEngagements(team1_scouting_states, harassing, scale, "Harassing")
    team2_scouting_states = integrateEngagements(team2_scouting_states, harassing, scale, "Harassing")

    team1_scouting_states = checkFirstInstance(team1_scouting_states, scale)
    team2_scouting_states = checkFirstInstance(team2_scouting_states, scale)

    team1_scouting_states = removeEmptyFrames(team1_scouting_states, frames)
    team2_scouting_states = removeEmptyFrames(team2_scouting_states, frames)

    # battleTimes = battle_detector.toTime(battles, frames, r.length.seconds)
    # battle_detector.printTime(battleTimes)

    return team1_scouting_states, team2_scouting_states

def scouting_freq_and_cat(replay):
    '''scouting_freq_and_cat takes in a previously loaded replay
    from sc2reader and returns the scouting frequency (instances per second)
    for each player, how their scouting behavior is categorized, as well as
    the winner of the game.'''
    r = replay

    try:
        frames = r.frames
        seconds = r.real_length.total_seconds()

        team1_scouting_states, team2_scouting_states = final_scouting_states(r)

        team1_num_times, team1_time = scouting_stats(team1_scouting_states)
        team2_num_times, team2_time = scouting_stats(team2_scouting_states)

        team1_freq = team1_num_times / seconds
        team2_freq = team2_num_times / seconds

        team1_cat = categorize_player(team1_scouting_states, frames)
        team2_cat = categorize_player(team2_scouting_states, frames)

        return team1_freq, team1_cat, team2_freq, team2_cat, r.winner.number

    except:
        print(replay.filename + "contains errors within scouting_detector")
        raise

def scouting_analysis(replay):
    r = replay
    try:
        team1_scouting_states, team2_scouting_states = final_scouting_states(r)
        battles, harassing = battle_detector.buildBattleList(r)

        frames = r.frames
        team1_initial = hasInitialScouting(team1_scouting_states, frames, battles)
        team2_initial = hasInitialScouting(team2_scouting_states, frames, battles)

        team1_cat = categorize_player(team1_scouting_states, frames)
        team2_cat = categorize_player(team2_scouting_states, frames)

        team1_base = scoutsMainBase(team1_scouting_states)
        team2_base = scoutsMainBase(team2_scouting_states)

        team1_newAreas = scoutNewAreas(team1_scouting_states)
        team2_newAreas = scoutNewAreas(team2_scouting_states)

        team1_betweenBattles = scoutBetweenBattles(team1_scouting_states, battles, frames)
        team2_betweenBattles = scoutBetweenBattles(team2_scouting_states, battles, frames)

        return {1: [team1_cat, team1_initial, team1_base, team1_newAreas, team1_betweenBattles],
                2: [team2_cat, team2_initial, team2_base, team2_newAreas, team2_betweenBattles]}
    except:
        print(replay.filename + "contains errors within scouting_detector")
        raise

def scouting_times(replay, which):
    '''scouting_times takes in a previously loaded replay from sc2reader as
        well as an integer (either 1 or 2) indicating which type of time list
        will be returned. 1 indicates a list of when scouting occurs as fractions
        of gametime, whereas 2 indicates a list of absolute frames.'''
    r = replay

    try:
        frames = r.frames
        team1_scouting_states, team2_scouting_states = final_scouting_states(r)

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
        print(replay.filename + "contains errors within scouting_detector")
        raise

def scouting_interval(replay):
    # scouting_interval takes in a previously loaded replay from sc2reader
        # and returns the average time (in seconds) between periods of scouting
        # for each player.'''
    r = replay
    try:
        factors = sc2reader.constants.GAME_SPEED_FACTOR
        scale = factors[r.expansion][r.speed]

        team1_scouting_states, team2_scouting_states = final_scouting_states(r)

        team1_avg_int = avg_interval(team1_scouting_states, scale)
        team2_avg_int = avg_interval(team2_scouting_states, scale)

        return team1_avg_int, team2_avg_int

    except:
        print(replay.filename + "contains errors within scouting_detector")
        raise

def print_verification(replay):
    # print_verification takes in a previously loaded replay from sc2reader
        # and prints out information useful for verification. More specifically,
        # it prints out the game time at which scouting states switch for each
        # team/player'''
    r = replay

    try:
        team1_scouting_states, team2_scouting_states = final_scouting_states(r)

        frames = r.frames
        seconds = r.length.seconds
        team1_time_dict = toTime(team1_scouting_states, frames, seconds)
        team2_time_dict = toTime(team2_scouting_states, frames, seconds)

        print("---Team 1---")
        printTime(team1_time_dict)
        print("\n\n---Team 2---")
        printTime(team2_time_dict)

    except:
        print(replay.filename + "contains errors within scouting_detector")
        raise
