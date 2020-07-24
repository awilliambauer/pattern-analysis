#Alison Cameron
#June 2020
#A program to detect scouting behavior of players in StarCraft 2

import sc2reader
import math
import battle_detector

def buildEventLists(tracker_events, game_events):
    '''buildEventLists is used to build up a list of events related to
    scouting behavior. It takes in a replay's tracker events and game events.
    It returns one list of all relevant events.'''

    unit_init_events = []
    camera_events = []
    team1 = 1
    team2 = 2
    start1 = False
    start2 = False
    team1_count = 0
    team2_count = 0
    for t_event in tracker_events:
        #checking for starting bases
        if isinstance(t_event, sc2reader.events.tracker.UnitBornEvent):
            if (start1 == False) and (t_event.control_pid == team1):
                unit_init_events.append(t_event)
                start1 = True
            elif (start2 == False) and (t_event.control_pid == team2):
                unit_init_events.append(t_event)
                start2 = True
        #checking for the creation of new bases
        elif isinstance(t_event, sc2reader.events.tracker.UnitInitEvent) and (t_event.unit.name == "Hatchery" or t_event.unit.name == "CommandCenter" or t_event.unit.name == "Nexus"):
            unit_init_events.append(t_event)

    for g_event in game_events:
        #filtering through camera events
        if isinstance(g_event, sc2reader.events.game.CameraEvent):
            camera_events.append(g_event)
            if g_event.player:
                if g_event.player.pid == 1:
                    team1_count += 1
                elif g_event.player.pid == 2:
                    team2_count += 1
            else:
                raise RuntimeError()
        #account for moving terran bases
        elif isinstance(g_event, sc2reader.events.game.TargetUnitCommandEvent) and (g_event.ability_name == "LandCommandCenter" or g_event.ability_name == "LandOrbitalCommand"):
            unit_init_events.append(g_event)

    #if either team has 0 camera events, scouting behavior cannot be detected and
    #the replay is invalid
    if team1_count == 0 or team2_count == 0:
        raise RuntimeError()

    return unit_init_events + camera_events


def buildScoutingDictionaries(events):
    '''buildScoutingDictionaries returns dictionaries for each player where the
    keys are the frame and the value is the state of scouting. "No scouting"
    indicates the team/player is not looking at any bases, "Scouting themself"
    indicates the team/player is looking at their own base, and
    "Scouting opponent" indicates the team/player is looking at their opponent's
    base. buildScoutingDictionaries takes in a list of filtered events returned
    by buildEventLists.'''

    team1 = 1
    team2 = 2

    team1_scouting_states = {}
    team2_scouting_states = {}

    # Dictionaries of the locations of bases where the keys are unit ids
    # and the values are locations (as tuples of (x, y) coordinates)
    team1_bases = {}
    team2_bases = {}

    #Used for updating the scouting dictionaries
    prev_state1 = "Viewing themself"
    prev_frame1 = 0
    prev_state2 = "Viewing themself"
    prev_frame2 = 0

    #iterating through events in order
    for event in sorted(events, key=lambda e: e.frame):
        i = event.frame
        #accounting for new bases
        if isinstance(event, sc2reader.events.tracker.TrackerEvent):
            if (event.control_pid == team1) and not(event.unit_id in team1_bases):
                team1_bases[event.unit_id] = event.location
            elif(event.control_pid == team2) and not(event.unit_id in team2_bases):
                team2_bases[event.unit_id] = event.location
        #accounting for Terran bases moving
        elif isinstance(event, sc2reader.events.game.TargetUnitCommandEvent):
            if(event.player.pid == team1):
                team1_bases[event.target_unit_id] = event.location
            elif(event.player.pid == team2):
                team2_bases[event.target_unit_id] = event.location
        #checking camera events
        else:
            player = event.player.pid
            camera_location = event.location
            if player == team1:
                #team1 is looking at their own base
                if withinDistance(camera_location, team1_bases):
                    team1_scouting_states = updatePrevScoutStates(team1_scouting_states, i, prev_frame1, prev_state1)
                    team1_scouting_states[i] = "Viewing themself"
                    prev_frame1 = i
                    prev_state1 = "Viewing themself"
                #team1 is looking at their opponent's base
                elif withinDistance(camera_location, team2_bases):
                    team1_scouting_states = updatePrevScoutStates(team1_scouting_states, i, prev_frame1, prev_state1)
                    team1_scouting_states[i] = "Scouting opponent"
                    prev_frame1 = i
                    prev_state1 = "Scouting opponent"
                #team1 is not looking at a base
                else:
                    team1_scouting_states = updatePrevScoutStates(team1_scouting_states, i, prev_frame1, prev_state1)
                    team1_scouting_states[i] = "No scouting"
                    prev_frame1 = i
                    prev_state1 = "No scouting"

            elif player == team2:
                #team2 is looking at their own base
                if withinDistance(camera_location, team2_bases):
                    team2_scouting_states = updatePrevScoutStates(team2_scouting_states, i, prev_frame2, prev_state2)
                    team2_scouting_states[i] = "Viewing themself"
                    prev_frame2 = i
                    prev_state2 = "Viewing themself"
                #team2 is looking at their opponent's base
                elif withinDistance(camera_location, team1_bases):
                    team2_scouting_states = updatePrevScoutStates(team2_scouting_states, i, prev_frame2, prev_state2)
                    team2_scouting_states[i] = "Scouting opponent"
                    prev_frame2 = i
                    prev_state2 = "Scouting opponent"
                #team2 is not looking at a base
                else:
                    team2_scouting_states = updatePrevScoutStates(team2_scouting_states, i, prev_frame2, prev_state2)
                    team2_scouting_states[i] = "No scouting"
                    prev_frame2 = i
                    prev_state2 = "No scouting"
    return team1_scouting_states, team2_scouting_states


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

def detect_scouting(replay):
    '''detect_scouting is the main function of this script. detect_scouting does
    error checking on replays and raises errors for replays with incomplete information,
    as well as combines all other functions. It takes in a previously loaded replay
    from sc2reader and returns the scouting frequency for each player, as well as
    the winner of the game.'''
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
        team1_scouting_states, team2_scouting_states = buildScoutingDictionaries(allEvents)

        battles = battle_detector.buildBattleList(r)
        team1_scouting_states = integrateBattles(team1_scouting_states, battles)
        team2_scouting_states = integrateBattles(team2_scouting_states, battles)

        team1_num_times, team1_time = scouting_stats(team1_scouting_states)
        team2_num_times, team2_time = scouting_stats(team2_scouting_states)

        return team1_num_times / r.real_length.total_seconds(), team2_num_times / r.real_length.total_seconds(), r.winner.number

    except:
        print(replay.filename + "contains errors within scouting_detector")
        raise
