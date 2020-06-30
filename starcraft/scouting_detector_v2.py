#Alison Cameron
#May 2020
#A program to detect scouting behavior of players in StarCraft 2

import sc2reader
import math

team1 = 1
team2 = 2

# Dictionaries where the keys are the frame and the value is the state of
# scouting. "No scouting" indicates the team/player is not looking at any bases,
# "Scouting themself" indicates the team/player is looking at their own base, and
# "Scouting opponent" indicates the team/player is looking at their opponent's base
team1_scouting_states = {}
team2_scouting_states = {}

#Dictionaries of the times at which each team switches their scouting state.
#These are intended to assist in the verification process
team1_times = {}
team2_times = {}

# Dictionaries of the locations of bases where the keys are unit ids
# and the values are locations (as tuples of (x, y) coordinates)
team1_bases = {}
team2_bases = {}

#r = sc2reader.load_replay("myReplays/ggtracker_335956.SC2Replay", load_map = True)
#r = sc2reader.load_replay("myReplays/ggtracker_336367.SC2Replay", load_map = True)
#r = sc2reader.load_replay("myReplays/ggtracker_341316.SC2Replay", load_map = True)
r = sc2reader.load_replay("replays/Deathaura.SC2Replay", load_map = True)
#r = sc2reader.load_replay("myReplays/Submarine.SC2Replay", load_map = True)
#r = sc2reader.load_replay("myReplays/terranMove.SC2Replay", load_map = True)
tracker_events = r.tracker_events
game_events = r.game_events
frames = r.frames

#lists of unit init events and camera events
unit_init_events = []
camera_events = []

def buildEventDictionaries():
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
        #accounting for Terrans moving their bases
        elif isinstance(t_event, sc2reader.events.tracker.UnitTypeChangeEvent) and t_event.unit.name == "CommandCenter":
            unit_init_events.append(t_event)

    for g_event in game_events:
        if isinstance(g_event, sc2reader.events.game.CameraEvent):
            camera_events.append(g_event)


def buildScoutingDictionaries_v2():
    prev_state1 = "Viewing themself"
    prev_frame1 = 0
    prev_state2 = "Viewing themself"
    prev_frame2 = 0

    for event in sorted(unit_init_events + camera_events, key=lambda e: e.frame):
        i = event.frame
        #accounting for Terran bases moving
        if isinstance(event, sc2reader.events.tracker.UnitTypeChangeEvent):
            #UnitTypeChangeEvents do not have locations, so I need to figure out
            #some other way of finding the new location
            if(event.unit.owner == r.players[0]):
                team1_bases[event.unit_id] = event.location
            elif(event.unit.owner == r.players[1]):
                team2_bases[event.unit_id] = event.location
        #accounting for new bases
        elif isinstance(event, sc2reader.events.tracker.TrackerEvent):
            if (event.control_pid == team1) and not(event.unit_id in team1_bases):
                team1_bases[event.unit_id] = event.location
            elif(event.control_pid == team2) and not(event.unit_id in team2_bases):
                team2_bases[event.unit_id] = event.location
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

def withinDistance(location, list):
    #returns true if location is within a distance of any locations in base dictionary
    loc_x, loc_y = location[0], location [1]
    keys = list.keys()
    for key in keys:
        loc = list[key]
        x, y = loc[0], loc[1]
        distance_apart = math.sqrt((loc_x - x)**2 + (loc_y - y)**2)
        if distance_apart <= 20:
            return True
    return False

def updatePrevScoutStates(scouting_dict, frame, prev_frame, prev_state):
    #updates all frames after prev_frame and before current frame to the prev_state
    if(prev_frame >= frame):
        return;

    i = prev_frame + 1
    while(i != frame):
        scouting_dict[i] = prev_state
        i += 1

def toTime(time_dict, scouting_dict):
    length = len(scouting_dict.keys())

    state = scouting_dict[1]
    time = (1/frames)*(r.length.seconds)
    minStr = "{:2d}".format(int(time//60))
    secStr = "{:05.2f}".format(time%60)
    timeStr = minStr + ":" + secStr
    time_dict[timeStr] = state

    frame = 2
    while(frame <= length):
        if scouting_dict[frame] != state:
            state = scouting_dict[frame]
            time = (frame/frames)*(r.length.seconds)
            minStr = "{:2d}".format(int(time//60))
            secStr = "{:05.2f}".format(time%60)
            timeStr = minStr + ":" + secStr
            time_dict[timeStr] = state
        frame += 1

def printTime(time_dict):
    keys = time_dict.keys()
    for key in keys:
        print(key, end = "")
        print(" -> ", end = "")
        print(time_dict[key])

def main():
    buildEventDictionaries()
    buildScoutingDictionaries_v2()


    print("\n--------Team 1---------")
    #print(team1_scouting_states)
    toTime(team1_times, team1_scouting_states)
    printTime(team1_times)
    print("\n\n--------Team 2---------")
    #print(team2_scouting_states)
    toTime(team2_times, team2_scouting_states)
    printTime(team2_times)


main()


# Not worrying about this right now, but we need to add a way to delete bases
# once they are destroyed.

# Lastly, don't you need a unit with you while your camera is moving to
# properly scout? Also, how will we tell the difference between scouting and
# a battle? Will we set a threshold of the amount of players in motion?
