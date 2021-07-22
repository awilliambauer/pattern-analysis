from typing import List

from starcraft.sc2reader.events.game import GameEvent
from unit_prediction import get_position_estimate_along_path
from unit_info import get_unit_vision_radius, get_unit_movement_speed, is_flying_unit
import sc2reader


class _UnitState:
    def __init__(self, unit_data, pos=None):
        self.unit_data = unit_data
        self.id = unit_data.id
        self.pos = pos
        self.path_queue = None
        self.path_start_frame = None
        self.movement_speed = get_unit_movement_speed(unit_data.name)
        self.vision_radius = get_unit_vision_radius(unit_data.name)
        self.flying = is_flying_unit(unit_data.name)

    def finish_path(self, current_frame):
        self.path_queue.pop(0)
        self.path_start_frame = current_frame

    def get_position_estimate(self, current_frame):
        if len(self.path_queue) == 0:
            return self.pos
        path = self.path_queue[0]
        if is_flying_unit(self.unit_data.name):
            distance_moved = (current_frame - self.path_start_frame) * self.movement_speed
            difference = path[-1] - path[0]
            if not difference.length or distance_moved > difference.length:
                self.finish_path(current_frame)
                return path[-1]
            direction = difference.normalized
            return path[0] + direction * distance_moved
        position_on_path = get_position_estimate_along_path(path, self.path_start_frame, current_frame,
                                                            self.movement_speed)
        if position_on_path == path[-1]:
            self.finish_path(current_frame)
        return position_on_path


class _PlayerState:
    def __init__(self, id):
        self.id = id
        self.camera_pos = None
        self.upgrades = []
        self.scouting_units = []


class _GameState:
    def __init__(self):
        self.current_frame = 0
        self._unit_states = {}
        self.player_states = (_PlayerState(1), _PlayerState(2))

    def get_unit_pos(self, unit_id):
        return self._unit_states[unit_id].pos

    def set_unit_pos(self, unit_id, pos):
        self._unit_states[unit_id].pos = pos

    def get_camera_pos(self, player_id):
        return self.player_states[player_id].camera_pos

    def set_camera_pos(self, player_id, pos):
        self.player_states[player_id].camera_pos = pos

    def update_unit_positions(self, current_frame):
        for unit_state in self._unit_states:
            unit_state.pos = unit_state.get_position_estimate(current_frame)


class ScoutingInstance:
    def __init__(self, player, start_time, end_time, location, units_used):
        self.player = player
        self.start_time = start_time
        self.end_time = end_time
        self.location = location
        self.units_used = units_used


class GameTickEvent:
    def __init__(self, frame):
        self.frame = frame


def _get_events(replay) -> List[GameEvent]:
    tick_events = [GameTickEvent(frame) for frame in range(0, replay.frames, 23)]  # approx every second
    sorted_events = sorted(replay.game_events + replay.tracker_events + tick_events, key=lambda e: e.frame)
    return sorted_events


def get_scouting_instances(replay) -> List[ScoutingInstance]:
    events = _get_events(replay)
    event_handlers = _init_event_handlers()
    game_state = _GameState()
    scouting_instances = []
    for event in events:
        handlers = [handler for predicate, handler in event_handlers.items() if predicate(event)]
        for handler in handlers:
            handler(event, game_state)
    return scouting_instances



def _init_event_handlers():
    def event(event_type):
        return lambda e: isinstance(e, event_type)

    return {
        event(sc2reader.events.tracker.UnitBornEvent): handle_unit_born_event,
        event(sc2reader.events.tracker.UnitPositionsEvent): handle_unit_positions_event,
        event(sc2reader.events.tracker.UnitDiedEvent): handle_unit_died_event,
        lambda e: ((
                           isinstance(e, sc2reader.events.game.TargetUnitCommandEvent) or
                           isinstance(e, sc2reader.events.game.TargetPointCommandEvent)) and
                   e.ability_name in ["RightClick", "Attack"]): handle_move_command,
        lambda e: ((
                           isinstance(e, sc2reader.events.game.TargetUnitCommandEvent) or
                           isinstance(e, sc2reader.events.game.TargetPointCommandEvent)) and
                   e.ability_name in ["ScannerSweep"]): handle_scanner_sweep,
        event(sc2reader.events.game.CameraEvent): handle_camera_event,
        event(GameTickEvent): handle_game_tick_event
    }


def handle_unit_died_event(event, game_state):


# for unit_id in units[event.unit.owner.pid]:
#     if unit_id == event.unit_id:
#         units[event.unit.owner.pid].pop(unit_id)
#         break

def handle_unit_born_event(event, game_state):
    game_state.set_unit_pos(event.control_pid, event.unit, event.location)


def handle_unit_positions_event(event, game_state):
    for unit in event.units.keys():
        game_state.set_unit_pos(unit.owner.pid, unit, event.units[unit])


def handle_scanner_sweep(event, game_state):


def handle_move_command(event, game_state):
    cur_team = event.player.pid

    # updating unit positions and checking for the first instance of scouting
    if event.ability_name in ["RightClick", "Attack"]:
        target_location = event.location[:2]

        # update the location of the unit we are targeting
        if "Unit" in event.name:
            if event.target_unit is None:
                # print("no target unit exists for targetunitcommandevent")
                pass
            elif not event.target_unit.is_building:
                game_state.set_unit_pos(event.target_unit.id, target_location)

            # checking for the first instance of scouting - units ordered to
        # the opponent's base
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
            if event.flag["queued"]:
                if len(unit_data.paths) == 0:
                    start_pos = unit_data.pos
                else:
                    start_pos = unit_data.paths[-1][-1]  # the goal of the last path
            else:
                start_pos = unit_data.pos
            if not is_flying_unit(selected_unit.name):
                path = current_map_path_data.get_path(start_pos, target_location)
            else:
                path = [Point2(start_pos), Point2(target_location)]
            if path is not None:
                if event.flag["queued"]:
                    unit_data.paths.append(path)
                else:
                    unit_data.paths = [path]
            unit_data.path_start_frame = event.frame


def handle_camera_event(event, game_state):
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
                scouting_occurences[cur_team].append(
                    {cur_frame / 22.4: possible_scouting_units[opp_team][base][0]})
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


def handle_game_tick_event(event, game_state):
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
                    if dist(base_location, unit_data.pos) < get_unit_vision_radius(unit_data.unit.name) * 2:
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
