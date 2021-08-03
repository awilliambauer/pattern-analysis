# A script to track battles and instances of harassing in StarCraft 2
# Alison Cameron
# July 2020
from enum import Enum

import sc2reader
from collections import defaultdict
import os
from collections import namedtuple
from math import dist

from sc2reader.events import UnitBornEvent, UnitDiedEvent, UnitInitEvent
from starcraft.base_plugins import BaseCluster, BaseType
from starcraft.sc2reader.factories.plugins.utils import GameState


class Engagement(namedtuple("Engagement", ("start_time", "end_time", "sides", "base_cluster"))):
    def __str__(self):
        time = str(int(self.start_time // 22.4 // 60)) + ":" + str(int((self.start_time // 22.4) % 60)) + "-" + str(
            int(self.end_time // 22.4 // 60)) + ":" + str(int((self.end_time // 22.4) % 60))

        return time + " " + str(self.base_cluster.player_id) + "'s " + str(self.base_cluster.base_type) + ": " + str(
            self.sides[0]) + ", " + str(self.sides[1])


class EngagementSide(namedtuple("EngagementSide", ("player_id", "total_army_value",
                                                   "total_army_supply", "army_supply_lost", "army_value_lost",
                                                   "total_worker_supply",
                                                   "worker_supply_lost", "total_building_count",
                                                   "building_count_lost", "unit_death_positions"))):
    def __str__(self):
        return str(self.player_id) + ": " + str(self.army_supply_lost) + "/" + str(
            self.total_army_supply) + " lost army, " + str(self.worker_supply_lost) + "/" + str(
            self.total_worker_supply) + " lost worker, " + str(self.building_count_lost) + "/" + str(
            self.total_building_count) + " lost buildings"


class EngagementType(Enum):
    ARMY_VS_WORKER = 0  # small quantity of units, attacking a base, within first 5 minutes


def remove_scouting_during_battles_and_harassment(replay, scouting_instances):
    team1_times = scouting_instances[0]
    team2_times = scouting_instances[1]
    battles, harassment = get_engagements(replay)
    team1_times = remove_scouting_during_battle(battles, team1_times)
    team1_times = remove_scouting_during_battle(harassment, team1_times)
    team2_times = remove_scouting_during_battle(battles, team2_times)
    team2_times = remove_scouting_during_battle(harassment, team2_times)
    return team1_times, team2_times


def track_player_units(replay):
    replay.players[0].current_units = GameState([])
    replay.players[1].current_units = GameState([])
    for event in replay.tracker_events:
        if isinstance(event, UnitBornEvent) or isinstance(event, UnitInitEvent):
            if event.unit is None:
                continue
            if event.control_pid not in [1, 2]:
                continue
            replay.players[event.control_pid - 1].current_units[event.frame].append((event.unit, event.location))
        elif isinstance(event, UnitDiedEvent):
            if event.unit is None or event.unit.owner is None:
                continue
            for unit, location in replay.players[event.unit.owner.pid - 1].current_units[event.frame]:
                if unit == event.unit:
                    replay.players[event.unit.owner.pid - 1].current_units[event.frame].remove((unit, location))
            event.unit.death_position = event.location


def _new_engagement(replay, start_time, base_cluster):
    return Engagement(start_time, start_time,
                      [_new_engagement_side(replay, start_time, 1), _new_engagement_side(replay, start_time, 2)],
                      base_cluster)


def _new_engagement_side(replay, start_time, player_id):
    player = replay.players[player_id - 1]
    current_units = list(map(lambda unit_and_loc: unit_and_loc[0], player.current_units[start_time]))
    army_units = list(filter(lambda unit: unit.is_army, current_units))
    buildings = list(filter(lambda unit: unit.is_building, current_units))
    worker_units = list(filter(lambda unit: unit.is_worker, current_units))
    return EngagementSide(player_id,
                          sum(map(lambda unit: unit.minerals + unit.vespene, army_units)),
                          sum(map(lambda unit: unit.supply, army_units)), 0, 0,
                          sum(map(lambda unit: unit.supply, worker_units)), 0, len(buildings), 0, [])


def get_engagements(replay):
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
        print("replay build is too old for battle detection!")
        return None

    # initializing the list of battles and harssing, where each instance is a tuple that contains
    # the frame that the engagement began and the frame that the engagement ended

    MAX_DEATH_SPACING_FRAMES = 10 * 22.4  # max number of frames between deaths for joining engagements
    MAX_ENGAGEMENT_DISTANCE_FROM_BASE = 30  # max distance between engagement loc and base cluster loc before the
    # engagement is considered definitely not at that base

    owned_units = []
    killed_units = []

    track_player_units(replay)

    for obj in replay.objects.values():
        if obj.owner is not None:
            if (
                    obj.is_worker or obj.is_army or obj.is_building) and obj.minerals is not None and obj.finished_at is not None:
                owned_units.append(obj)
                if obj.died_at is not None:
                    killed_units.append(obj)

    # sorted by frame each unit died at
    killed_units = sorted(killed_units, key=lambda obj: obj.died_at)

    engagements = []
    dead_units = []
    # building the list of engagements
    for unit in killed_units:
        if unit.killing_player is not None and (unit.minerals + unit.vespene > 0) and unit.owner.pid in [1, 2]:
            dead_units.append(unit)
            base_clusters = set(unit.owner.base_cluster[unit.died_at].values())
            base_clusters.update(set(unit.killing_player.base_cluster[
                                         unit.died_at].values()))
            nearest_base_clusters = list(filter(
                lambda base_cluster: dist(base_cluster.center, unit.death_position) < MAX_ENGAGEMENT_DISTANCE_FROM_BASE,
                base_clusters))
            if len(nearest_base_clusters) == 0:
                nearest_base_clusters = [BaseCluster(-1, unit.death_position, BaseType.NONE, -1)]
            matching_engagements = []
            for engagement in engagements:
                if unit.died_at - engagement.end_time > MAX_DEATH_SPACING_FRAMES:
                    # engagement is too temporally separated
                    continue
                if engagement.base_cluster not in nearest_base_clusters:
                    # engagement is not taking place at a base cluster that this could be at
                    continue
                # this engagement is one which our unit could possibly be in
                matching_engagements.append(engagement)
            if len(matching_engagements) == 0:
                closest_matching_engagement = _new_engagement(replay, unit.died_at, min(nearest_base_clusters))
            else:
                closest_matching_engagement = min(matching_engagements,
                                                  key=lambda e: dist(e.base_cluster.center,
                                                                     unit.death_position))
                engagements.remove(closest_matching_engagement)

            side = closest_matching_engagement.sides[unit.owner.pid - 1]
            army_supply_lost = side.army_supply_lost
            army_value_lost = side.army_value_lost
            worker_supply_lost = side.worker_supply_lost
            building_count_lost = side.building_count_lost
            if unit.is_army:
                army_supply_lost += unit.supply
                army_value_lost += unit.minerals + unit.vespene
            elif unit.is_worker:
                worker_supply_lost += unit.supply
            elif unit.is_building:
                building_count_lost += 1
            new_side = EngagementSide(unit.owner.pid, side.total_army_value, side.total_army_supply, army_supply_lost,
                                      army_value_lost, side.total_worker_supply, worker_supply_lost,
                                      side.total_building_count, building_count_lost,
                                      side.unit_death_positions + [unit.death_position])
            other_side = closest_matching_engagement.sides[1 if unit.owner.pid == 1 else 0]
            sides = [new_side, other_side] if new_side.player_id == 1 else [other_side, new_side]
            new_engagement_tuple = Engagement(closest_matching_engagement.start_time, unit.died_at, sides,
                                              closest_matching_engagement.base_cluster)
            engagements.append(new_engagement_tuple)

    return engagements


def remove_scouting_during_battle(battle_list, scouting_list):
    scouting_list_no_battles = []
    for scouting_instance in scouting_list:
        during_battle = False
        for frame in range(scouting_instance.start_time, scouting_instance.end_time + 1, 23):
            if duringBattle(frame, battle_list, 10 * 22.4):
                during_battle = True
                break
        if not during_battle:
            scouting_list_no_battles.append(scouting_instance)
    return scouting_list_no_battles


def duringBattle(frame, battles, margin=10 * 22.4):
    '''duringBattle returns true if a frame takes place during a battle.
    The parameters are a frame of the game and a list of battles returned by
    buildBattleList.'''
    for battle in battles:
        if frame >= (battle[0] - margin) and frame <= (battle[1] + margin):
            return True
