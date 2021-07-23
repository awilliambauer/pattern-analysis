from typing import List, Tuple
from sc2.position import Point2
from sc2reader.events.game import GameEvent
from unit_prediction import get_position_estimate_along_path, window
from unit_info import get_unit_vision_radius, get_unit_movement_speed, is_flying_unit, can_produce
import sc2reader
from math import dist
from enum import Enum
from collections import defaultdict, namedtuple

# MAGIC CONSTANTS
# the distance a unit or camera view needs to be from any base of an opponent before it's scouting
SCOUTING_CAMERA_DISTANCE_FROM_BASE = 25
SCOUTING_UNIT_DISTANCE_FROM_BASE = 25
# the maximum time after a unit arrives at an opponents base before the player views that base during which it can be
# considered scouting
SCOUTING_MAX_TIME_AFTER_UNIT_ARRIVES = 30 * 22.4
# the maximum time between instances of scouting after which they are considered separate
SCOUTING_MAX_TIME_TO_JOIN = 10 * 22.4


class _UnitState:
    def __init__(self, unit_data, pos=None):
        self.unit_data = unit_data
        self.owner = unit_data.owner
        self.id = unit_data.id
        self.pos = pos
        self.path_queue = []
        self.path_start_frame = None
        self.movement_speed = get_unit_movement_speed(unit_data.name)
        self.vision_radius = get_unit_vision_radius(unit_data.name)
        self.flying = is_flying_unit(unit_data.name)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, _UnitState):
            return False
        return o.id == self.id

    def __str__(self):
        return str(self.unit_data)

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


PotentialScoutingGroup = namedtuple("PotentialScoutingGroup",
                                    ["frame", "units_scouting", "units_being_scouted", "base_cluster"])


class _PlayerState:
    def __init__(self, player):
        self.id = player.pid
        self.race = player.play_race
        self.camera_pos = None
        self.upgrades = []
        self.bases = player.bases
        self.base_cluster = player.base_cluster
        self.potential_scouting_groups = []  # a list of PotentialScoutingGroups
        self.buildings_unseen = []
        self.active_scanner_sweeps = []
        self.rallies = {}  # building_id: rally_point_queue


class _GameState:
    def __init__(self, replay, map_path_data):
        self.map_path_data = map_path_data
        self.current_frame = 0
        self._unit_states = {}  # unit_id: _UnitState
        self.player_states = {1: _PlayerState(replay.players[0]), 2: _PlayerState(replay.players[1])}
        self.objects = replay.objects

    def unit_pos_exists(self, unit_id):
        return unit_id in self._unit_states

    def get_unit_state(self, unit_id):
        return self._unit_states[unit_id]

    def get_unit_pos(self, unit_id):
        return self.get_unit_state(unit_id).pos

    def set_unit_pos(self, unit, pos):
        if unit.id not in self._unit_states:
            self._unit_states[unit.id] = _UnitState(unit, pos)
        else:
            self._unit_states[unit.id].pos = pos

    def delete_unit(self, unit_id):
        if unit_id in self._unit_states:
            self._unit_states.pop(unit_id)

    def get_camera_pos(self, player_id):
        return self.player_states[player_id].camera_pos

    def set_camera_pos(self, player_id, pos):
        self.player_states[player_id].camera_pos = pos

    def update_unit_positions(self, current_frame):
        for unit_state in self._unit_states.values():
            unit_state.pos = unit_state.get_position_estimate(current_frame)


class ScoutingInstance:
    def __init__(self, player, start_time, end_time, location, units_used, units_scouted, scouting_type):
        self.player = player
        self.start_time = start_time
        self.end_time = end_time
        self.location = location
        self.units_used = units_used
        self.units_scouted = units_scouted
        self.scouting_type = scouting_type

    def __str__(self):
        return str(self.player) + ": " + str(self.start_time) + "-" + str(self.end_time) + \
               ", loc: " + str(self.location) + ", with: " + str(self.units_used) + ", on: " + str(self.units_scouted)


class GameTickEvent:
    def __init__(self, frame):
        self.frame = frame


def _get_events(replay) -> List[GameEvent]:
    tick_events = [GameTickEvent(frame) for frame in range(0, replay.frames, 23)]  # approx every second
    sorted_events = sorted(replay.game_events + replay.tracker_events + tick_events, key=lambda e: e.frame)
    return sorted_events


def get_scouting_instances(replay, map_path_data) -> Tuple[List[ScoutingInstance], List[ScoutingInstance]]:
    events = _get_events(replay)
    event_handlers = _init_event_handlers()
    game_state = _GameState(replay, map_path_data)
    unjoined_scouting_instances = []
    for event in events:
        handlers = [handler for predicate, handler in event_handlers.items() if predicate(event)]
        for handler in handlers:
            new_scouting_instances = handler(event, game_state)  # functions default to returning none
            if new_scouting_instances is not None:
                for scouting_instance in new_scouting_instances:
                    unjoined_scouting_instances.append(scouting_instance)

    scouting_instances_per_player = {1: [], 2: []}
    for scouting_instance in unjoined_scouting_instances:
        scouting_instances_per_player[scouting_instance.player].append(scouting_instance)
    for pid in [1, 2]:
        scouting_instances = scouting_instances_per_player[pid]
        idx = 0
        while idx < len(scouting_instances) - 1:
            first = scouting_instances[idx]
            second = scouting_instances[idx + 1]
            if second.start_time - first.end_time > SCOUTING_MAX_TIME_TO_JOIN:
                idx += 1
                continue
            # otherwise, join them
            scouting_instances.pop(idx + 1)
            scouting_instances.pop(idx)
            combined_units_used = first.units_used
            for unit in second.units_used:
                if unit not in combined_units_used:
                    combined_units_used.append(unit)
            combined_units_scouted = first.units_scouted
            for unit in second.units_scouted:
                if unit not in combined_units_scouted:
                    combined_units_scouted.append(unit)
            merged = ScoutingInstance(first.player, first.start_time, second.end_time,
                                      (first.location + second.location) / 2, combined_units_used,
                                      combined_units_scouted, first.scouting_type)
            scouting_instances.insert(idx, merged)

    return scouting_instances_per_player[1], scouting_instances_per_player[2]


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
    game_state.delete_unit(event.unit_id)


def handle_unit_born_event(event, game_state):
    game_state.set_unit_pos(event.unit, event.location)
    # want to find the nearest production facility and see if it has a rally
    if event.control_pid == 0:
        # not controlled by a player
        return
    buildings = game_state.player_states[event.control_pid].bases[event.frame].items()
    possible_production_buildings = list(
        filter(lambda id_and_loc: can_produce(game_state.objects[id_and_loc[0]].name, event.unit.name), buildings))
    if len(possible_production_buildings) == 0:
        return
    closest_compatible_building = min(
        possible_production_buildings,
        key=lambda id_and_loc: dist(id_and_loc[1][:2], event.location[:2]))


def handle_unit_positions_event(event, game_state):
    for unit in event.units.keys():
        game_state.set_unit_pos(unit, event.units[unit])


def handle_scanner_sweep(event, game_state):
    if event.player.pid == 1:
        opponent_id = 2
    else:
        opponent_id = 1
    buildings_being_scouted = []
    base_clusters_labeled = {}
    base_clusters = defaultdict(lambda: 0)
    for building_id, building_location in game_state.player_states[opponent_id].bases[event.frame].items():
        if dist(building_location, event.location[:2]) < 13:
            buildings_being_scouted.append(game_state.objects[building_id])
            base_cluster = game_state.player_states[opponent_id].base_cluster[event.frame][building_id]
            base_clusters_labeled[base_cluster.label] = base_cluster
            base_clusters[base_cluster.label] += 1
    if len(buildings_being_scouted) == 0:
        return
    most_common_base_cluster = max(base_clusters.items(), key=lambda cluster_count: cluster_count[1])[0]
    game_state.player_states[event.player.pid].potential_scouting_groups.append(
        PotentialScoutingGroup(event.frame, ["ScannerSweep"], buildings_being_scouted,
                               base_clusters_labeled[most_common_base_cluster]))


def handle_move_command(event, game_state):
    target_location = event.location[:2]
    # if it's a target unit command, we can update the position of the unit it's targeting
    if isinstance(event, sc2reader.events.game.TargetUnitCommandEvent):
        if event.target_unit is None:
            # print("no target unit exists for targetunitcommandevent")
            pass
        else:
            game_state.set_unit_pos(event.target_unit, target_location)
    # print("sending", event.active_selection, "to", target_location, "at sec", cur_frame / 22.4)
    for selected_unit in event.active_selection:
        if selected_unit.is_building:
            rallies = game_state.player_states[selected_unit.owner.pid].rallies
            if event.flag["queued"]:
                if selected_unit not in rallies:
                    rallies[selected_unit] = []
                rallies[selected_unit].append(event.location[:2])
            else:
                rallies[selected_unit] = [event.location[:2]]
            continue
        if not game_state.unit_pos_exists(selected_unit.id):
            # print("missing previous information about", selected_unit.name)
            game_state.set_unit_pos(selected_unit, None)  # add it to our list with no pos info
            # if we have no previous information about the position of the unit, ignore it
            continue
        unit_state = game_state.get_unit_state(selected_unit.id)
        if unit_state.pos is None:
            # print("unit data exists but has no position for", selected_unit.name)
            continue
        if event.flag["queued"]:
            if len(unit_state.path_queue) == 0:
                start_pos = unit_state.pos
            else:
                start_pos = unit_state.path_queue[-1][-1]  # the goal of the last path
        else:
            start_pos = unit_state.pos
        if not is_flying_unit(selected_unit.name):
            path = game_state.map_path_data.get_path(start_pos, target_location)
        else:
            path = [Point2(start_pos), Point2(target_location)]
        if path is not None:
            if event.flag["queued"]:
                unit_state.path_queue.append(path)
            else:
                unit_state.path_queue = [path]
        unit_state.path_start_frame = event.frame


def handle_camera_event(event, game_state):
    if event.player.is_observer or event.player.is_referee:
        return
    player_id = event.player.pid
    if player_id == 1:
        opponent_id = 2
    elif player_id == 2:
        opponent_id = 1
    camera_location = event.location

    buildings_in_range_of_camera = list(map(lambda building_and_loc: building_and_loc[0],
                                            filter(lambda building_and_loc: dist(camera_location,
                                                                                 building_and_loc[
                                                                                     1]) < SCOUTING_CAMERA_DISTANCE_FROM_BASE,
                                                   game_state.player_states[opponent_id].bases[event.frame].items())))

    actual_scouting_groups = []

    for potential_scouting_group in game_state.player_states[player_id].potential_scouting_groups:
        time_since_arrived = event.frame - potential_scouting_group.frame
        if time_since_arrived > SCOUTING_MAX_TIME_AFTER_UNIT_ARRIVES:
            # this event was too long ago, remove it from the list
            game_state.player_states[player_id].potential_scouting_groups.remove(potential_scouting_group)
            continue
        # if any of the buildings scouted by this group are in range of the camera
        if not any(filter(lambda unit: unit.id in buildings_in_range_of_camera,
                          potential_scouting_group.units_being_scouted)):
            continue

        actual_scouting_groups.append(potential_scouting_group)

    return [ScoutingInstance(player_id, event.frame, event.frame, scouting_group.base_cluster.center,
                             scouting_group.units_scouting, scouting_group.units_being_scouted,
                             scouting_group.base_cluster.base_type) for scouting_group in actual_scouting_groups]


def handle_game_tick_event(event, game_state):
    for player_id in [1, 2]:
        game_state.update_unit_positions(event.frame)
        units_scouting_base = {}
        for unit_id, unit_state in game_state._unit_states.items():
            if unit_state.pos is None:
                continue
            if not (unit_state.unit_data.is_army or unit_state.unit_data.is_worker):
                continue
            if unit_state.owner.pid != player_id:
                continue
            if player_id == 1:
                opponent_id = 2
            elif player_id == 2:
                opponent_id = 1
            for building_id, location in game_state.player_states[opponent_id].bases[event.frame].items():
                if dist(location, unit_state.pos) < get_unit_vision_radius(unit_state.unit_data.name) * 1.5:
                    if building_id not in units_scouting_base:
                        units_scouting_base[building_id] = []
                    units_scouting_base[building_id].append(unit_state.unit_data)
                    # print("second", cur_frame / 22.4, unit_data.unit.name,
                    #       "views base")
        for building_id, units_scouting in units_scouting_base.items():
            base_cluster = game_state.player_states[opponent_id].base_cluster[event.frame][building_id]
            existing_potential_scouting_groups = game_state.player_states[player_id].potential_scouting_groups
            matching_potential_scouting_groups = list(
                filter(lambda existing_group: existing_group.base_cluster.label == base_cluster.label,
                       existing_potential_scouting_groups))
            for group in matching_potential_scouting_groups:
                existing_potential_scouting_groups.remove(group)
            building_unit = game_state.objects[building_id]
            scouting_group = PotentialScoutingGroup(event.frame, units_scouting, [building_unit], base_cluster)
            existing_potential_scouting_groups.append(scouting_group)
