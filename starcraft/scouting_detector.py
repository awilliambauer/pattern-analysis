# Alison Cameron
# June 2020
# A script to detect scouting behavior and various metrics of players in StarCraft 2
from sc2.position import Point2
import sc2reader
import math
import statistics
from math import dist
import traceback
import battle_detector
from unit_prediction import get_position_estimate_along_path, get_movement_speed, is_flying_unit

# MAGIC CONSTANTS
# the distance a unit or camera view needs to be from any base of an opponent before it's scouting
SCOUTING_CAMERA_DISTANCE_FROM_BASE = 25
SCOUTING_UNIT_DISTANCE_FROM_BASE = 25
# the maximum time after a unit arrives at an opponents base before the player views that base during which it can be
# considered scouting
SCOUTING_MAX_TIME_AFTER_UNIT_ARRIVES = 30 * 22.4

unit_vision_ranges = {
    "LurkerBurrowed": 10,
    "OverlordTransport": 11,
    "MotershipCore": 14,
    "LurkerEgg": 9,
    "Mothership": 14,
    "WidowMineBurrowed": 7,
    "SiegeTankSieged": 11,
    "Carrier": 12,
    "Tempest": 12,
    "Battlecruiser": 12,
    "SensorTower": 12,
    "BroodLord": 12,
    "Hive": 12,
    "Nexus": 11,
    "Observer": 11,
    "PhotonCannon": 11,
    "Cyclone": 11,
    "Ghost": 11,
    "Medivac": 11,
    "CommandCenter": 11,
    "MissileTurret": 11,
    "OrbitalCommand": 11,
    "PlanetaryFortress": 11,
    "Raven": 11,
    "SiegeTank": 11,
    "Thor": 11,
    "Lair": 11,
    "Mutalisk": 11,
    "Overlord": 11,
    "Overseer": 11,
    "SpineCrawler": 11,
    "SporeCrawler": 11,
    "Viper": 11,
    "Colossus": 10,
    "HighTemplar": 10,
    "Oracle": 10,
    "Phoenix": 10,
    "Sentry": 10,
    "Stalker": 10,
    "VoidRay": 10,
    "WarpPrism": 10,
    "Banshee": 10,
    "Bunker": 10,
    "Hellbat": 10,
    "Hellion": 10,
    "Liberator": 10,
    "LiberatorAG": 10,
    "Marauder": 10,
    "Viking": 10,
    "VikingAssault":10,
    "Corruptor": 10,
    "Hatchery": 10,
    "Infestor": 10,
    "Lurker": 10,
    "NydusWorm": 10,
    "SwarmHost": 10,
    "Adept": 9,
    "Archon": 9,
    "Disruptor": 9,
    "Immortal": 9,
    "MothershipCore": 9,
    "Zealot": 9,
    "Marine": 9,
    "Reaper": 9,
    "Hydralisk": 9,
    "InfestedTerran": 9,
    "Queen": 9,
    "Ravager": 9,
    "Roach": 9,
    "Ultralisk": 9,
    "DarkTemplar": 8,
    "Probe": 8,
    "MULE": 8,
    "SCV": 8,
    "Baneling": 8,
    "Changeling": 8,
    "Drone": 8,
    "Zergling": 8,
    "Interceptor": 7,
    "Auto-Turret": 7,
    "PointDefenseDrone": 7,
    "WidowMine": 7,
    "Broodling": 7,
    "Locust": 6,
    "Cocoon": 5,
    "Larva": 5,
}


def get_unit_vision_range(unit_name):
    if "Changeling" in unit_name:
        return get_unit_vision_range("Changeling")
    if unit_name not in unit_vision_ranges:
        with open("missing_unit_vision.txt","a") as f:
            f.write(unit_name + "\n")
        return 9 # todo make this unnecessary
    return unit_vision_ranges[unit_name]


class UpdatePathingUnitPositionsEvent:  # see build_scouting_dictionaries for purpose
    def __init__(self, frame):
        self.frame = frame


def build_event_lists(tracker_events, game_events):
    '''build_event_lists is used to build up a list of events related to
    scouting behavior. It takes in a replay's tracker events and game events.
    It returns one list of all relevant events.'''

    events = []
    ability_names = ["RightClick", "Attack"]
    base_names = ["Hatchery", "Lair", "Hive", "Nexus", "CommandCenter", "CommandCenterFlying",
                  "OrbitalCommand", "OrbitalCommandFlying", "PlanetaryFortress"]
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
        elif isinstance(t_event, sc2reader.events.tracker.UnitBornEvent) and (
                t_event.unit.is_army or t_event.unit.is_worker):
            events.append(t_event)
        # More information about unit positions
        elif isinstance(t_event, sc2reader.events.tracker.UnitPositionsEvent):
            events.append(t_event)
        # removing dead units
        elif isinstance(t_event, sc2reader.events.tracker.UnitDiedEvent) and (
                t_event.unit.is_army or t_event.unit.is_worker):
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
        # account for moving units
        elif isinstance(g_event, sc2reader.events.game.TargetUnitCommandEvent) and (
                g_event.ability_name in ability_names):
            events.append(g_event)
        elif isinstance(g_event, sc2reader.events.game.TargetPointCommandEvent) and (
                g_event.ability_name in ability_names):
            events.append(g_event)

    # if either team has 0 camera events, scouting behavior cannot be detected and
    # the replay is invalid
    if (team1_count == 0 or team2_count == 0):
        print("replay is invalid because there are no camera events")
        raise RuntimeError()

    sorted_events = sorted(events, key=lambda e: e.frame)
    return sorted_events


class UnitData:

    def __init__(self, unit, pos):
        self.unit = unit
        self.id = unit.id
        self.pos = (pos[0], pos[1]) if pos is not None else None  # making sure that we aren't using z coord
        self.path = None
        self.path_start_frame = None
        self.unit_speed = get_movement_speed(unit.name)
        self.flying = is_flying_unit(unit.name)

    def get_position_estimate(self, frame):
        if self.path is None:
            return self.pos
        if len(self.path) == 1:
            return self.path[0]
        if is_flying_unit(self.unit.name):
            distance_moved = (frame - self.path_start_frame) * self.unit_speed
            if distance_moved > (self.path[-1] - self.path[0]).length:
                return self.path[-1]
            direction = (self.path[-1] - self.path[0]).normalized
            return self.path[0] + direction * distance_moved
        return get_position_estimate_along_path(self.path, self.path_start_frame, frame, self.unit_speed)


def update_estimated_unit_positions(units, team, current_frame):
    for unit in units[team].values():
        unit.pos = unit.get_position_estimate(current_frame)


def initialize_scouting_dictionaries(frames):
    """initializeScoutingDictionaries takes in the total frames in a game and
    returns a dictionary initialized in the proper format to be used by
    buildScoutingDictionaries"""
    dicts = {1: {}, 2: {}}
    for i in range(1, 3):
        for j in range(1, frames + 1):
            dicts[i][j] = ["", []]
    return dicts


def build_scouting_dictionaries(replay, events, objects, frames, current_map_path_data):
    '''buildScoutingDictionaries returns dictionaries for each player where the
    keys are the frame and the keys are a list of tags indicating what the player
    is viewing. Tags such as battles and harassing are added later by using
    integrateEngagements. This function takes in a previously loaded replay object,
    a list of events returned by buildEventLists, a list of objects obtained by
    replay.objects.values(), as well as the total frames in the game.

    currentMapPathData is the loaded statically calculated paths for the map of this
    replay. It is used to estimate unit positions.
    '''

    # UpdatePathingUnitPositionsEvent gets injected into the events list once every 23 frames to ensure that
    # there are regular checks for game events that we care about that happen based on unit positions.
    # without these events, the time at which we would check for these game events would depend on when there
    # are other events. Thus, if there were no events for a long period of time, we would have no idea whether
    # or not scouting occurred then
    injected_events = events.copy()
    for frame in range(0, replay.frames, 23):  # step approximately every second
        injected_events.append(UpdatePathingUnitPositionsEvent(frame))
    injected_events = sorted(injected_events, key=lambda e: e.frame)

    scouting_states = initialize_scouting_dictionaries(frames)

    # Dictionaries for each team of the locations of bases where the keys are unit ids
    # and the values are locations (as tuples of (x, y) coordinates)
    og_bases = {1: {}, 2: {}}
    # Add starting bases
    base_names = ["Hatchery", "Lair", "Hive", "Nexus", "CommandCenter", "CommandCenterFlying",
                  "OrbitalCommand", "OrbitalCommandFlying", "PlanetaryFortress"]

    for cur_frame in range(1, 3):
        start_base = \
            [u for u in objects if
             u.name in base_names and u.owner is not None and u.owner.pid == cur_frame and u.finished_at == 0][0]
        og_bases[cur_frame][start_base.id] = start_base.location

    units = {1: {}, 2: {}}  # todo make sure we aren't caring about the fog of war unit that defaults to id 0
    # indexed by team, then by base, returns tuple of unit and frame arrived at base
    possible_scouting_units = {1: {}, 2: {}}

    def set_unit_pos(team, unit, pos):
        for other_unit_id, other_unit_data in units[team].items():
            if other_unit_id == unit.id:
                other_unit_data.pos = pos
                return
        units[team][unit.id] = UnitData(unit, pos)

    # Used for updating the scouting dictionaries
    prev_states = {1: "Viewing themself", 2: "Viewing themself"}
    prev_frames = {1: 0, 2: 0}

    first_instance = {1: True, 2: True}

    # iterating through events in order
    for event in injected_events:
        cur_frame = event.frame
        # adding new units to the list of active units
        if isinstance(event, sc2reader.events.tracker.UnitBornEvent):
            if not event.unit.is_building:
                set_unit_pos(event.control_pid, event.unit, event.location)

        # updating unit positions
        elif isinstance(event, sc2reader.events.tracker.UnitPositionsEvent):
            for unit in event.units.keys():
                if not unit.is_building:
                    set_unit_pos(unit.owner.pid, unit, event.units[unit])

        # removing dead units
        elif isinstance(event, sc2reader.events.tracker.UnitDiedEvent):
            for unit_id in units[event.unit.owner.pid]:
                if unit_id == event.unit_id:
                    units[event.unit.owner.pid].pop(unit_id)
                    break

        # updating unit positions, and the first instance of scouting
        elif isinstance(event, sc2reader.events.game.TargetUnitCommandEvent) or isinstance(event,
                                                                                           sc2reader.events.game.TargetPointCommandEvent):
            cur_team = event.player.pid
            # updating unit positions and checking for the first instance of scouting
            if event.ability_name in ["RightClick", "Attack"]:

                # update the location of the unit we are targeting
                if "Unit" in event.name:
                    if event.target_unit is None:
                        # print("no target unit exists for targetunitcommandevent")
                        pass
                    elif not event.target_unit.is_building:
                        set_unit_pos(cur_team, event.target_unit, (event.location[0], event.location[1]))

                    # checking for the first instance of scouting - units ordered to
                # the opponent's base

                target_location = event.location
                # print("sending", event.active_selection, "to", target_location, "at sec", cur_frame / 22.4)
                # the current player has just ordered their selected non-building units to move to the location
                for selected_unit in event.active_selection:
                    if selected_unit.is_building:
                        continue
                    if selected_unit.id not in units[cur_team]:
                        # print("missing previous information about", selected_unit.name)
                        # if we have no previous information about the position of the unit, ignore it
                        set_unit_pos(cur_team, selected_unit, None)  # add it to our list with no pos info
                        continue
                    unit_data = units[cur_team][selected_unit.id]
                    if unit_data.pos is None:
                        # print("unit data exists but has no position for", selected_unit.name)
                        continue
                    unit_data.path = current_map_path_data.get_path(unit_data.pos, target_location)
                    unit_data.path_start_frame = event.frame
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
            if within_distance(camera_location, event.player.bases[cur_frame],
                               SCOUTING_CAMERA_DISTANCE_FROM_BASE):  # TODO inaccuracy? this checks with
                # if the camera x,y is within 25 units of the base x,y, but might want to take into account
                # width and height of base/camera
                scouting_states[cur_team] = updatePrevScoutStates(scouting_states[cur_team], cur_frame,
                                                                  prev_frames[cur_team],
                                                                  prev_states[cur_team])
                scouting_states[cur_team][cur_frame][0] = "Viewing themself"
                prev_frames[cur_team] = cur_frame
                prev_states[cur_team] = "Viewing themself"
            # looking at their opponent's original base
            else:
                viewing_opp_base = False
                for base, base_location in replay.player[opp_team].bases[cur_frame].items():
                    # print("the camera is at", camera_location, "the base is at", base_location)
                    # print("last time units arrived at this base: ",
                    #       last_time_unit_at_base[opp_team][base] if base in last_time_unit_at_base[
                    #           opp_team] else -1e10)
                    # print("cur frame:", cur_frame)
                    # print("units within scouting threshold:", (
                    #         cur_frame - (last_time_unit_at_base[opp_team][base] if base in last_time_unit_at_base[
                    #     opp_team] else -1e10)) < SCOUTING_MAX_TIME_AFTER_UNIT_ARRIVES)
                    # for each base existing at this frame
                    if dist(camera_location, base_location) < SCOUTING_CAMERA_DISTANCE_FROM_BASE and \
                            (cur_frame - (possible_scouting_units[opp_team][base][1] if base in possible_scouting_units[
                                opp_team] else -1e10)) < SCOUTING_MAX_TIME_AFTER_UNIT_ARRIVES:
                        # if the camera is within a certain distance and a unit has been there in the last X seconds
                        scouting_states[cur_team] = updatePrevScoutStates(scouting_states[cur_team], cur_frame,
                                                                          prev_frames[cur_team],
                                                                          prev_states[cur_team])
                        base_location = (int(camera_location[0]), int(camera_location[1]))
                        # print("scouting detected by player", cur_team, "with units",
                        #       possible_scouting_units[opp_team][base][0], "at second", cur_frame / 22.4)
                        scouting_states[cur_team][cur_frame][0] = "Scouting opponent - main base" if base in og_bases[
                            cur_team] else "Scouting opponent - expansions"
                        scouting_states[cur_team][cur_frame].append(base_location)
                        prev_frames[cur_team] = cur_frame
                        prev_states[cur_team] = "Scouting opponent - main base" if base in og_bases[
                            cur_team] else "Scouting opponent - expansions"
                        # first_instance[cur_team] = False
                        viewing_opp_base = True
                        break
                    # not looking at a base
                if not viewing_opp_base:
                    scouting_states[cur_team] = updatePrevScoutStates(scouting_states[cur_team], cur_frame,
                                                                      prev_frames[cur_team],
                                                                      prev_states[cur_team])
                    scouting_states[cur_team][cur_frame][0] = "Viewing empty map space"
                    prev_frames[cur_team] = cur_frame
                    prev_states[cur_team] = "Viewing empty map space"
        elif isinstance(event, UpdatePathingUnitPositionsEvent):
            for team in [1, 2]:
                update_estimated_unit_positions(units, team, cur_frame)
                units_scouting_base = {}
                for unit_id, unit_data in units[team].items():
                    if unit_data.pos is None:
                        continue
                    if first_instance[team]:
                        if team == 1:
                            opp_team = 2
                        elif team == 2:
                            opp_team = 1  # TODO potential bug here? players might be different ids?
                        for base, base_location in replay.player[opp_team].bases[cur_frame].items():
                            if dist(base_location, unit_data.pos) < get_unit_vision_range(unit_data.unit.name) * 2:
                                if base not in units_scouting_base:
                                    units_scouting_base[base] = []
                                units_scouting_base[base].append(unit_data.unit)
                                # possible_scouting_units[opp_team][base] = cur_frame
                                # print("second", cur_frame / 22.4, unit_data.unit.name,
                                #       "views base")
                                # first_instance[team] = False
                                scouting_states[team][cur_frame][1].append("Units at opponent base")
                for base, units_scouting in units_scouting_base.items():
                    possible_scouting_units[opp_team][base] = (units_scouting, cur_frame)
    return scouting_states[1], scouting_states[2]


def within_distance(location, list, distance):
    '''withinDistance returns true if the input location is within a
    certain range of any location in the list. The location is a tuple
    and the list is a dictionary of unit ids mapped to locations as tuples.
    The distance is dependent on the user but a reasonable input is within
    0-200 (the general size of a sc2 map)'''
    loc_x, loc_y = location[0], location[1]
    keys = list.keys()
    for key in keys:
        loc = list[key]
        x, y = loc[0], loc[1]
        distance_apart = math.sqrt((loc_x - x) ** 2 + (loc_y - y) ** 2)
        if distance_apart <= distance:
            return True
    return False


def updatePrevScoutStates(scouting_dict, frame, prev_frame, prev_state):
    '''updatePrevScoutStates takes in a scouting dictonary, and returns the
    same dictionary, but with the tag 'prev_state' added to all frames in
    between prev_frame and frame'''
    if (prev_frame >= frame):
        return scouting_dict

    keys = scouting_dict.keys()
    i = prev_frame + 1
    while (i != frame):
        scouting_dict[i][0] = prev_state
        i += 1
    return scouting_dict


def checkFirstInstance(scouting_dict, scale):
    '''checkFirstInstance is intended to avoid false positive scouting tags
    in the early game.
    Because it is possible for a player to 'scout' their opponent without
    actually looking at their base (a player can order units around the map
    and only view the mini-map), we added a sending units tag as a possible
    first instance of scouting. checkFirstInstance finds these tags and verifies
    that there does not already exist an instance of scouting within the next 45
    seconds, and if an instance does exist, checkFirstInstance eradicates this
    false positive scouting tag.
    checkFirstInstance takes in a scouting dictionary, as well as the scale -
    which indicates game speed in frames per second. It returns the updated and
    verified scouting dictionary'''
    frame_jump45 = 45 * int(scale)
    send_units = "Sending units to the opponent's base"
    keys = scouting_dict.keys()
    for key in keys:
        state = scouting_dict[key][1]
        # recorded instance of ordering units
        if send_units in state:
            # check if there is a regular scouting instance within the next 45 secs
            for i in range(key + 1, frame_jump45 + key + 1):
                # make sure it doesn't throw a key error
                if i in keys:
                    if is_scouting(scouting_dict[i]):
                        # if there is an instance of scouting, reset the original
                        scouting_dict[key][1].remove(send_units)
                        return scouting_dict
            # No instance that already corresponds to the unit ordering, consider it scouting
            # for 10 seconds
            frame_jump10 = 10 * int(scale)
            for i in range(key, key + frame_jump10 + 1):
                if not (send_units in scouting_dict[i][1]):
                    scouting_dict[i][1].append(send_units)
            return scouting_dict
    # No ordering of units, return original dictionary
    return scouting_dict


def is_scouting(frame_list):
    '''isScouting returns True if the combination of tags for a frame indicates
    that the player is scouting, and False if otherwise. isScouting takes in
    a list of tags that can be obtained by accessing scouting_dictionary[frame]'''
    state = frame_list[0]
    events = frame_list[1]
    # scouting opponent but not harassing or engaged in battle
    return ("Scouting opponent" in state) and not ("Harassing" in events) and not ("Engaged in Battle" in events)


def removeEmptyFrames(scouting_dict, frames):
    '''removeEmptyFrames deletes frames from a scouting dictionary
    that contain no information. Often times the total frames of a game
    are slightly longer than the frames for which there is camera and game
    information. This function takes in a scouting dictionary and the
    total frames in a game, and returns the updated scouting dictionary.'''
    frame = frames
    initial_list = ["", []]
    state = scouting_dict[frame]
    while (state == initial_list):
        scouting_dict.pop(frame)
        frame -= 1
        state = scouting_dict[frame]
    return scouting_dict


def to_time(scouting_dict, frames, seconds):
    '''Creates and returns time-formatted dictionary of the time of game when
    a player's scouting state changes. Takes in a scouting dictionary, the total
    number of frames in the game, and the length of the game in seconds. Most
    useful for verification and testing.'''
    length = len(scouting_dict.keys())
    time_dict = {}

    state = scouting_dict[1]
    stateStr = state[0]
    if not (stateStr):
        stateStr = "No camera data"
    for event in state[1]:
        stateStr = stateStr + ", while" + event
    time = (1 / frames) * (seconds)
    minStr = "{:2d}".format(int(time // 60))
    secStr = "{:05.2f}".format(time % 60)
    timeStr = minStr + ":" + secStr
    time_dict[timeStr] = stateStr

    frame = 2
    while (frame <= length):
        if scouting_dict[frame] != state:
            state = scouting_dict[frame]
            stateStr = state[0]
            if not (stateStr):
                stateStr = "No camera data"
            for event in state[1]:
                stateStr = stateStr + ", while " + event
            time = (frame / frames) * (seconds)
            minStr = "{:2d}".format(int(time // 60))
            secStr = "{:05.2f}".format(time % 60)
            timeStr = minStr + ":" + secStr
            time_dict[timeStr] = stateStr
        frame += 1
    return time_dict


def print_time(time_dict):
    '''Used to neatly print a time dictionary returned by toTime.'''
    keys = time_dict.keys()
    for key in keys:
        print(key, end="")
        print(" -> ", end="")
        print(time_dict[key])


def integrate_engagements(scouting_dict, engagements, scale, addition):
    '''integrateEngagements is used to cross-check a scouting dictionary with a
    list of engagements(battles or harassing). More specifically, it is used to
    avoid false positives of "scouting" an opponent during engagements.
    integrateEngagements takes in a scouting dictionary returned by
    buildScoutingDictionaries, a list of engagements returned by
    battle_detector.buildBattleList, the game speed in frames per second, and a
    string indicating what tag should be added to the list of tags for each
    frame during the period of engagements. It also adds this tag to 20 seconds
    before each engagement, as well as 20 after.

    For example, integrateEngagements(team1_scouting, battles, 22, "In Battle")
    will add the tag "In Battle" to 20 seconds before each battle in the list
    of battles, as well as 20 seconds after. This will prevent false positives
    for scouting, which are detected by isScouting.

    It returns the updated scouting dictionary.'''

    BUFFER = int(20 * scale)

    # TODO make list of tags a set so we can get away with not caring about duplicates
    for frame in scouting_dict:
        if battle_detector.duringBattle(frame, engagements):
            scouting_dict[frame][1].append(addition)
            # extend engagement tag BUFFER frames before it starts
            # TODO differentiate between these "fake" pre-engagement tags and the actual period of engagement
            if not battle_detector.duringBattle(frame - 1, engagements):
                for f in range(frame - BUFFER, frame):
                    if f in scouting_dict:
                        scouting_dict[f][1].append(addition)
            # extend engagement tag BUFFER frames after it ends
            if not battle_detector.duringBattle(frame + 1, engagements):
                for f in range(frame + 1, frame + BUFFER + 1):
                    if f in scouting_dict:
                        scouting_dict[f][1].append(addition)

    return scouting_dict


def final_scouting_states(replay, current_map_path_data):
    '''final_scouting_states is the backbone of scouting_detector.py. It does
        all of the error checking needed, as well as combines all functions to
        create completed scouting dictionaries - to then be used by other functions.
        It takes in a previously loaded replay object from sc2reader and returns
        completed scouting dictionaries for each player. This function is also
        critical in understanding the order in which scouting dictionaries
        must be built by using various functions in this file.'''
    r = replay

    tracker_events = r.tracker_events
    game_events = r.game_events
    frames = r.frames
    # scale = 16*sc2reader.constants.GAME_SPEED_FACTOR[r.expansion][r.speed]
    # it appears the scale is always 22.4 in our dataset, despite documentation to the contrary
    scale = 22.4

    allEvents = build_event_lists(tracker_events, game_events)
    objects = r.objects.values()
    team1_scouting_states, team2_scouting_states = build_scouting_dictionaries(r, allEvents, objects, frames,
                                                                               current_map_path_data)

    battles, harassing = battle_detector.buildBattleList(r)
    team1_scouting_states = integrate_engagements(team1_scouting_states, battles, scale, "Engaged in Battle")
    team2_scouting_states = integrate_engagements(team2_scouting_states, battles, scale, "Engaged in Battle")
    team1_scouting_states = integrate_engagements(team1_scouting_states, harassing, scale, "Harassing")
    team2_scouting_states = integrate_engagements(team2_scouting_states, harassing, scale, "Harassing")

    team1_scouting_states = checkFirstInstance(team1_scouting_states, scale)
    team2_scouting_states = checkFirstInstance(team2_scouting_states, scale)

    team1_scouting_states = removeEmptyFrames(team1_scouting_states, frames)
    team2_scouting_states = removeEmptyFrames(team2_scouting_states, frames)

    return team1_scouting_states, team2_scouting_states
