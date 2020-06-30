#Alison Cameron
#June 2020
#A program to detect scouting behavior of players in StarCraft 2

import sc2reader
import math

def buildEventDictionaries(tracker_events, game_events):
    '''Builds a list of all relevant events for scouting detection'''

    unit_init_events = []
    camera_events = []
    team1 = 1
    team2 = 2
    start1 = False
    start2 = False
    for t_event in tracker_events:
        #checking for starting bases
        if isinstance(t_event, sc2reader.events.tracker.UnitBornEvent):
            if (start1 == False) and (t_event.control_pid == team1):
                unit_init_events.append(t_event)
                start1 = True
            elif (start2 == False) and (t_event.control_pid == team2):
                unit_init_events.append(t_event)
                start2 = True
        elif isinstance(t_event, sc2reader.events.tracker.UnitInitEvent) and (t_event.unit.name == "Hatchery" or t_event.unit.name == "CommandCenter" or t_event.unit.name == "Nexus"):
            unit_init_events.append(t_event)

    for g_event in game_events:
        if isinstance(g_event, sc2reader.events.game.CameraEvent):
            camera_events.append(g_event)
        #account for moving terran bases
        elif isinstance(g_event, sc2reader.events.game.TargetUnitCommandEvent) and (g_event.ability_name == "LandCommandCenter" or g_event.ability_name == "LandOrbitalCommand"):
            unit_init_events.append(g_event)

    return unit_init_events + camera_events


def buildScoutingDictionaries(events):
    '''Builds dictionaries where the keys are the frame and the value is the state of
       scouting. "No scouting" indicates the team/player is not looking at any bases,
       "Scouting themself" indicates the team/player is looking at their own base, and
       "Scouting opponent" indicates the team/player is looking at their opponent's base'''

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
                    updatePrevScoutStates(team1_scouting_states, i, prev_frame1, prev_state1)
                    team1_scouting_states[i] = "Viewing themself"
                    prev_frame1 = i
                    prev_state1 = "Viewing themself"
                #team1 is looking at their opponent's base
                elif withinDistance(camera_location, team2_bases):
                    updatePrevScoutStates(team1_scouting_states, i, prev_frame1, prev_state1)
                    team1_scouting_states[i] = "Scouting opponent"
                    prev_frame1 = i
                    prev_state1 = "Scouting opponent"
                #team1 is not looking at a base
                else:
                    updatePrevScoutStates(team1_scouting_states, i, prev_frame1, prev_state1)
                    team1_scouting_states[i] = "No scouting"
                    prev_frame1 = i
                    prev_state1 = "No scouting"

            elif player == team2:
                #team2 is looking at their own base
                if withinDistance(camera_location, team2_bases):
                    updatePrevScoutStates(team2_scouting_states, i, prev_frame2, prev_state2)
                    team2_scouting_states[i] = "Viewing themself"
                    prev_frame2 = i
                    prev_state2 = "Viewing themself"
                #team2 is looking at their opponent's base
                elif withinDistance(camera_location, team1_bases):
                    updatePrevScoutStates(team2_scouting_states, i, prev_frame2, prev_state2)
                    team2_scouting_states[i] = "Scouting opponent"
                    prev_frame2 = i
                    prev_state2 = "Scouting opponent"
                #team2 is not looking at a base
                else:
                    updatePrevScoutStates(team2_scouting_states, i, prev_frame2, prev_state2)
                    team2_scouting_states[i] = "No scouting"
                    prev_frame2 = i
                    prev_state2 = "No scouting"
    return team1_scouting_states, team2_scouting_states


def withinDistance(location, list):
    '''Returns true if input location is within a distance of any locations in
       base dictionary'''
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
    '''Updates all frames after prev_frame and before current frame to the prev_state'''
    if(prev_frame >= frame):
        return;

    i = prev_frame + 1
    while(i != frame):
        scouting_dict[i] = prev_state
        i += 1

def toTime(scouting_dict, frames, seconds):
    '''Creates a formatted dictionary of the time of game when a player's
        scouting state changes. Most useful for verification and testing.'''
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
    '''Used to neatly print a time dictionary in an easy to read way.'''
    keys = time_dict.keys()
    for key in keys:
        print(key, end = "")
        print(" -> ", end = "")
        print(time_dict[key])

def detect_scouting(filename):
    r = sc2reader.load_replay(filename)
    tracker_events = r.tracker_events
    game_events = r.game_events
    frames = r.frames
    seconds = r.length.seconds

    allEvents = buildEventDictionaries(tracker_events, game_events)
    team1_scouting_states, team2_scouting_states = buildScoutingDictionaries(allEvents)

    team1_times = toTime(team1_scouting_states, frames, seconds)
    team2_times = toTime(team2_scouting_states, frames, seconds)

    print("\n--------Team 1---------")
    #print(team1_scouting_states)
    printTime(team1_times)

    print("\n\n--------Team 2---------")
    #print(team2_scouting_states)
    printTime(team2_times)

detect_scouting("replays/ggtracker_341316.SC2Replay")
