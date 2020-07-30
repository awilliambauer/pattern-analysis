#Alison Cameron
#June 2020
#A program to detect scouting behavior of players in StarCraft 2

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
        #checking for the creation of new bases
        if isinstance(t_event, sc2reader.events.tracker.UnitInitEvent) and (t_event.unit.name in base_names):
            events.append(t_event)
        #Adding information about units
        elif isinstance(t_event, sc2reader.events.tracker.UnitBornEvent) and (t_event.unit.is_army or t_event.unit.is_worker):
            events.append(t_event)
        #More information about unit positions
        elif isinstance(t_event, sc2reader.events.tracker.UnitPositionsEvent):
            events.append(t_event)
        #removing dead units
        elif isinstance(t_event, sc2reader.events.tracker.UnitDiedEvent) and (t_event.unit.is_army or t_event.unit.is_worker):
            events.append(t_event)

    for g_event in game_events:
        #filtering through camera events
        if isinstance(g_event, sc2reader.events.game.CameraEvent):
            events.append(g_event)
            if g_event.player:
                if g_event.player.pid == 1:
                    team1_count += 1
                elif g_event.player.pid == 2:
                    team2_count += 1
            else:
                raise RuntimeError()
        #account for moving terran bases and moving units
        elif isinstance(g_event, sc2reader.events.game.TargetUnitCommandEvent) and (g_event.ability_name in ability_names):
            events.append(g_event)
        elif isinstance(g_event, sc2reader.events.game.UpdateTargetUnitCommandEvent) and (g_event.ability_name in ability_names):
            events.append(g_event)

    #if either team has 0 camera events, scouting behavior cannot be detected and
    #the replay is invalid
    if team1_count == 0 or team2_count == 0:
        raise RuntimeError()

    sorted_events = sorted(events, key=lambda e: e.frame)
    return sorted_events


def buildScoutingDictionaries(events, objects):
    '''buildScoutingDictionaries returns dictionaries for each player where the
    keys are the frame and the value is the state of scouting. "No scouting"
    indicates the team/player is not looking at any bases, "Scouting themself"
    indicates the team/player is looking at their own base, and
    "Scouting opponent" indicates the team/player is looking at their opponent's
    base. buildScoutingDictionaries takes in a list of filtered events returned
    by buildEventLists.'''

    team1 = 1
    team2 = 2

    scouting_states = {1: {}, 2: {}}

    # Dictionaries for each team of the locations of bases where the keys are unit ids
    # and the values are locations (as tuples of (x, y) coordinates)
    bases = {1: {}, 2: {}}
    #Add starting bases
    base_names = set(["Hatchery", "Lair", "Hive", "Nexus", "CommandCenter", "CommandCenterFlying",
                        "OrbitalCommand", "OrbitalCommandFlying","PlanetaryFortress"])
    for i in range(1, 3):
        start_base = [u for u in objects if u.name in base_names and u.owner != None and u.owner.pid == i and u.finished_at == 0][0]
        bases[i][start_base.id] = start_base.location

    # Dictionaries for each team of the active units where the keys are the unit ids
    # and the values are locations (as tuples of (x, y) coordinates)
    active_units = {1: {}, 2: {}}

    #Used for updating the scouting dictionaries
    prev_states = {1: "Viewing themself", 2: "Viewing themself"}
    prev_frames = {1: 0, 2: 0}

    #iterating through events in order
    for event in events:
        i = event.frame
        #accounting for new bases
        if isinstance(event, sc2reader.events.tracker.UnitInitEvent):
            cur_team = event.control_pid
            bases[cur_team][event.unit_id] = event.location

        #accounting for Terran bases moving and updating unit positions
        elif isinstance(event, sc2reader.events.game.TargetUnitCommandEvent):
            cur_team = event.player.pid
            #moving Terran bases
            if event.ability_name in ["LandCommandCenter", "LandOrbitalCommand"]:
                bases[cur_team][event.target_unit_id] = event.location
            #updating unit positions
            elif event.ability_name in ["RightClick", "Attack"]:
                active_units[cur_team][event.target_unit_id] = event.location

        #accounting for Terran bases moving and updating unit positions
        elif isinstance(event, sc2reader.events.game.UpdateTargetUnitCommandEvent):
            cur_team = event.player.pid
            #moving Terran bases
            if event.ability_name in ["LandCommandCenter", "LandOrbitalCommand"]:
                bases[cur_team][event.target_unit_id] = event.location
            #updating unit positions
            elif event.ability_name in ["RightClick", "Attack"]:
                active_units[cur_team][event.target_unit_id] = event.location

        #adding new units to the list of active units
        elif isinstance(event, sc2reader.events.tracker.UnitBornEvent):
            cur_team = event.control_pid
            active_units[cur_team][event.unit_id] = event.location

        #updating unit positions
        elif isinstance(event, sc2reader.events.tracker.UnitPositionsEvent):
            for unit in event.units.keys():
                cur_team = unit.owner.pid
                location = event.units[unit]
                active_units[cur_team][unit.id] = location

        #removing dead units
        elif isinstance(event, sc2reader.events.tracker.UnitDiedEvent):
            cur_team = event.unit.owner.pid
            if event.unit_id in active_units[cur_team]:
                active_units[cur_team].pop(event.unit_id)

        #checking camera events
        elif isinstance(event, sc2reader.events.game.CameraEvent):
            if event.player.is_observer or event.player.is_referee:
                continue

            cur_team = event.player.pid
            if cur_team == 1:
                opp_team = 2
            elif cur_team == 2:
                opp_team = 1
            camera_location = event.location
            #looking at their own base
            if withinDistance(camera_location, bases[cur_team]):
                scouting_states[cur_team] = updatePrevScoutStates(scouting_states[cur_team], i, prev_frames[cur_team], prev_states[cur_team])
                scouting_states[cur_team][i] = "Viewing themself"
                prev_frames[cur_team] = i
                prev_states[cur_team] = "Viewing themself"
            #looking at their opponent's base and has a unit with them
            elif withinDistance(camera_location, bases[opp_team]) and withinDistance(camera_location, active_units[cur_team]):
                scouting_states[cur_team] = updatePrevScoutStates(scouting_states[cur_team], i, prev_frames[cur_team], prev_states[cur_team])
                scouting_states[cur_team][i] = "Scouting opponent"
                prev_frames[cur_team] = i
                prev_states[cur_team] = "Scouting opponent"
            #not looking at a base
            else:
                scouting_states[cur_team] = updatePrevScoutStates(scouting_states[cur_team], i, prev_frames[cur_team], prev_states[cur_team])
                scouting_states[cur_team][i] = "No scouting"
                prev_frames[cur_team] = i
                prev_states[cur_team] = "No scouting"

    return scouting_states[1], scouting_states[2]


def withinDistance(location, list):
    '''withinDistance returns true if the input location is within a
    certain range of any location in the list. The location is a tuple
    and the list is a list of locations as tuples.'''
    loc_x, loc_y = location[0], location [1]
    keys = list.keys()
    for key in keys:
        loc = list[key]
        x, y = loc[0], loc[1]
        distance_apart = math.sqrt((loc_x - x)**2 + (loc_y - y)**2)
        if distance_apart <= 25:
            return True
    return False

def updatePrevScoutStates(scouting_dict, frame, prev_frame, prev_state):
    '''updatePrevScoutStates updates the input scouting dictionary from
    the prev_frame to frame with prev_state and returns the scouting
    dictionary back.'''
    if(prev_frame >= frame):
        return scouting_dict

    i = prev_frame + 1
    while(i != frame):
        scouting_dict[i] = prev_state
        i += 1
    return scouting_dict

def toTime(scouting_dict, frames, seconds):
    '''Creates and returns time-formatted dictionary of the time of game when
    a player's scouting state changes. Takes in a scouting dictionary, the total
    number of frames in the game, and the length of the game in seconds. Most
    useful for verification and testing.'''
    length = len(scouting_dict.keys())
    time_dict = {}

    state = scouting_dict[1]
    time = (1/frames)*(seconds)
    minStr = "{:2d}".format(int(time//60))
    secStr = "{:05.2f}".format(time%60)
    timeStr = minStr + ":" + secStr
    time_dict[timeStr] = state

    frame = 2
    while(frame <= length):
        if scouting_dict[frame] != state:
            state = scouting_dict[frame]
            time = (frame/frames)*(seconds)
            minStr = "{:2d}".format(int(time//60))
            secStr = "{:05.2f}".format(time%60)
            timeStr = minStr + ":" + secStr
            time_dict[timeStr] = state
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
    if scouting_dict[1] == "Scouting opponent":
        num_times += 1
        scouting_frames += 1
        cur_scouting = True
    total_frames += 1
    frame = 2
    while(frame < length):
        total_frames += 1
        if scouting_dict[frame] == "Scouting opponent":
            #if the player is in a streak of scouting
            if cur_scouting == True:
                scouting_frames += 1
            #if the player just switched states from something else
            #to scouting their opponent
            else:
                num_times += 1
                scouting_frames += 1
                cur_scouting = True
        else:
            cur_scouting = False
        frame += 1

    #calculating rates based on counts
    scouting_fraction = scouting_frames/total_frames
    scouting_rate = num_times/total_frames

    return num_times, scouting_frames

def integrateBattles(scouting_dict, battles):
    '''integrateBattles is used to cross-check a scouting dictionary with a
    list of battles. More specifically, it is used to avoid false-positives
    of "scouting" an opponent during battle. integrateBattles takes in a
    scouting dictionary returned by buildScoutingDictionaries and a list
    of battles returned by battle_detector.buildBattleList. It returns the
    updated scouting dictionary.'''
    length = len(scouting_dict.keys())
    frame = 1
    while frame < length:
        if scouting_dict[frame] == "Scouting opponent" and battle_detector.duringBattle(frame, battles):
            scouting_dict[frame] = "No scouting"
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
        if state == "Scouting opponent":
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
    #a non-scouter
    if no_scouting:
        category = 1
        return category

    #only scouts in the beginning
    if beginning_scouting:
        category = 2
        return category

    if len(intervals) == 0:
        category = 3
        return category

    mean_interval = statistics.mean(intervals)
    stdev = statistics.pstdev(intervals)

    #Sporadic scouter
    if stdev/mean_interval >= 0.5:
        category = 3
    #Consistent scouter
    elif stdev/mean_interval < 0.5:
        category = 4

    return category

def scouting_timefrac_list(scouting_dict, frames):
    time_fracs = []
    keys = scouting_dict.keys()
    cur_scouting = False
    for key in keys:
        state = scouting_dict[key]
        if state == "Scouting opponent":
            if not(cur_scouting):
                frac = key/frames
                time_fracs.append(frac)
            cur_scouting = True
        else:
            cur_scouting = False

    return time_fracs

def scouting_freq_and_cat(replay):
    '''scouting_freq_and_cat is the main function of this script. detect_scouting does
    error checking on replays and raises errors for replays with incomplete information,
    as well as combines all other functions. It takes in a previously loaded replay
    from sc2reader and returns the scouting frequency for each player, their
    category of scouting, as well as the winner of the game.'''
    r = replay

    # # Only applied to missing ability info, which doesn't matter for scouting detection
    # if hasattr(r, "marked_error") and r.marked_error:
    #     print("skipping", r.filename, "as it contains errors")
    #     print(r.filename, "has build", r.build, "but best available datapack is", r.datapack.id)
    #     raise RuntimeError()

    if r.winner is None:
        print(r.filename, "has no winner information")
        raise RuntimeError()

    try:
        #some datafiles did not have a 'Controller' attribute
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
    seconds = r.length.seconds

    try:
        allEvents = buildEventLists(tracker_events, game_events)
        objects = r.objects.values()
        team1_scouting_states, team2_scouting_states = buildScoutingDictionaries(allEvents, objects)

        battles = battle_detector.buildBattleList(r)
        team1_scouting_states = integrateBattles(team1_scouting_states, battles)
        team2_scouting_states = integrateBattles(team2_scouting_states, battles)

        team1_num_times, team1_time = scouting_stats(team1_scouting_states)
        team2_num_times, team2_time = scouting_stats(team2_scouting_states)

        team1_freq = team1_num_times / r.real_length.total_seconds()
        team2_freq = team2_num_times / r.real_length.total_seconds()

        team1_cat = categorize_player(team1_scouting_states, frames)
        team2_cat = categorize_player(team2_scouting_states, frames)

        # team1_time_dict = toTime(team1_scouting_states, frames, seconds)
        # team2_time_dict = toTime(team2_scouting_states, frames, seconds)

        # timelist = battle_detector.toTime(battles, frames, seconds)
        # print("---Battles---")
        # battle_detector.printTime(timelist)
        # print("---Team 1---")
        # printTime(team1_time_dict)
        # print("\n\n---Team 2---")
        # printTime(team2_time_dict)

        return team1_freq, team1_cat, team2_freq, team2_cat, r.winner.number

    except:
        print(replay.filename + "contains errors within scouting_detector")
        raise


def scouting_times(replay):
    r = replay

    tracker_events = r.tracker_events
    game_events = r.game_events
    frames = r.frames

    allEvents = buildEventLists(tracker_events, game_events)
    objects = r.objects.values()
    team1_scouting_states, team2_scouting_states = buildScoutingDictionaries(allEvents, objects)

    battles = battle_detector.buildBattleList(r)
    team1_scouting_states = integrateBattles(team1_scouting_states, battles)
    team2_scouting_states = integrateBattles(team2_scouting_states, battles)

    team1_time_list = scouting_timefrac_list(team1_scouting_states, frames)
    team2_time_list = scouting_timefrac_list(team2_scouting_states, frames)

    return team1_time_list, team2_time_list
