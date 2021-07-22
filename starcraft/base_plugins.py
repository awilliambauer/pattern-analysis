"""
Plugins to track bases and mining locations, based on https://github.com/dsjoerg/ggpyjobs/blob/master/sc2parse/plugins.py
Aaron Bauer
Carleton College
August 2020
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"  # keep sklearn from spawning a bunch of threads

import traceback
from sklearn.cluster import AffinityPropagation
import numpy as np
from math import dist
import time
import warnings
warnings.simplefilter('ignore', UserWarning)
import sc2reader
from sc2reader.log_utils import loggable
from sc2reader.engine.plugins import SelectionTracker, APMTracker
from selection_plugin import ActiveSelection
from typing import NamedTuple, Tuple
from enum import Enum

base_names_tier_one = set(["Hatchery", "Nexus", "CommandCenter"])
base_names = set(["Hatchery", "Lair", "Hive", "Nexus", "CommandCenter", "CommandCenterFlying", "OrbitalCommand", "OrbitalCommandFlying","PlanetaryFortress"])
flying_buildings = ["Barracks", "Factory", "Starport", "CommandCenter"]
land_cc_abils = ['LandOrbitalCommand', 'LandCommandCenter']
cc_names = ['OrbitalCommand', 'CommandCenter']
BaseType = Enum("BaseType", "MAIN EXPANSION PROXY")

class BaseCluster(NamedTuple):
    label: int
    center: Tuple[int, int]
    base_type: BaseType


def is_mining_loc(resource_clusters, location):
    """
    Uses affinity propagation (https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation)
    to cluster resource locations into possible base locations, returns True if location within 8.5 of a center
    """
    # 8.5 threshold set empirically based on 5-replay sample
    # TODO verify with bigger sample
    if len(resource_clusters) == 0:
        return False
    return min([dist(location, center) for center in resource_clusters]) < 8.5


def select_center(cs):
    if len(cs) == 1:
        return cs[0]
    for c in cs:
        assert len([dist(c, x) for x in cs if x != c]) > 0, cs
    return min(cs, key=lambda c: np.mean([dist(c, x) for x in cs if x != c]))


def get_current_name(unit, frame):
    name = unit.name
    for f, t in unit.type_history.items():
        if frame >= f:
            name = t.name
    return name


def get_last_known_loc(unitid, frame, lookup):
    for f in range(frame - 1, 0, -1):
        if f in lookup and unitid in lookup[f]:
            return f, lookup[f][unitid]
    raise ValueError(f"no last known location for {unitid}")


# TODO differentiate between mining bases and other bases (usually turrets/bunkers/nydus worm/proxy barrack or pylon)
@loggable
class BaseTracker(object):
    """
    Tracks a player's current bases
    """

    name = "BaseTracker"


    def handleInitGame(self, event, replay):
        self.resource_locs = None
        try:
            replay.load_map()
            self.map_dim = (replay.map.map_info.width + replay.map.map_info.height) / 2
        except:
            self.map_dim = 150  # go with a reasonable if map can't be loaded
        self.under_construction = {}
        self.building_locs = {}
        self.lookup = {}
        self.keyframes = []
        self.mains = []


    def handleUnitBornEvent(self, event, replay):
        if event.unit.is_building:
            # this must be a starting base
            assert event.unit.name in base_names, "born event for non-main building"
            self.building_locs[event.unit.id] = (event.location, event.unit.owner.team_id, 0, True)
            self.lookup[event.frame] = self.building_locs.copy()
            self.keyframes.append(event.frame)
            self.mains.append(event.location)
            


    def handleUnitInitEvent(self, event, replay):
        # resources should have been created by now, can initialize resoursce locations
        if self.resource_locs is None:
            self.resource_locs = [o.location for o in replay.objects.values() if o.name and ("Mineral" in o.name or "Vespene" in o.name) and hasattr(o, "location")]
            try:
                self.resource_clusters = AffinityPropagation(preference=-1000, random_state=None).fit(self.resource_locs).cluster_centers_
            except:
                print(self.resource_locs)
                traceback.print_exc()
        if event.unit.is_building:
            assert event.unit.id not in self.under_construction, "init event for already under construction unit"
            self.under_construction[event.unit.id] = (event.location, event.unit.owner.team_id, event.unit.finished_at,
                                                      is_mining_loc(self.resource_clusters, event.location) and event.unit.name in base_names)


    def handleUnitDoneEvent(self, event, replay):
        if event.unit.is_building:
            assert event.unit.id in self.under_construction and event.unit.id not in self.building_locs, "done event dictionary problem"
            self.building_locs[event.unit.id] = self.under_construction[event.unit.id]
            del self.under_construction[event.unit.id]
            self.lookup[event.frame] = self.building_locs.copy()
            self.keyframes.append(event.frame)


    def handleUnitDiedEvent(self, event, replay):
        if event.unit.is_building and "Flying" not in get_current_name(event.unit, event.frame):
            if event.unit.id not in self.building_locs and event.unit.id not in self.under_construction:
                self.logger.info(f"dying {event.unit}, built {event.unit.finished_at} missing from dicts as of {event.frame}, probably due to missed landing: {replay.filename}")
            if event.unit.id in self.under_construction:
                del self.under_construction[event.unit.id]
            elif event.unit.id in self.building_locs:
                del self.building_locs[event.unit.id]
            self.lookup[event.frame] = self.building_locs.copy()
            self.keyframes.append(event.frame)


    def handleTargetPointCommandEvent(self, event, replay):
        if event.ability_name.startswith("Land"):
            selects = event.active_selection
            flying_selects = [obj for obj in selects if sorted(obj.type_history.items())[0][1].name in flying_buildings]
            target_type = event.ability_name[4:]
            targets = [obj for obj in flying_selects if get_current_name(obj, event.frame) == f"{target_type}Flying"]
            if len(targets) == 0:
                self.logger.error(f"Landing {target_type} but none selected. event={event}, selects={selects}, names={[get_current_name(s, event.frame) for s in selects]}")
            else:
                # assume the order is given to the first selected building
                # TODO is this true? probably not, selection is sorted by id
                unit = targets[0]
                if unit.id in self.building_locs:
                    self.logger.info(f"\nflying building {unit} should not be in dict: {replay.filename}, {event}")
                    # go back through events and find the type change where the building lifted off
                    for e in [e for e in replay.events[::-1] if e.frame < event.frame]:
                        if isinstance(e, sc2reader.events.UnitTypeChangeEvent) and e.unit == unit and e.unit_type_name == f"{target_type}Flying":
                            # remove flying building from intervening frames
                            for frame in range(event.frame - 1, e.frame - 1, -1):
                                if frame in self.lookup and unit.id in self.lookup[frame]:
                                    del self.lookup[frame][unit.id]
                            self.logger.info(f"CORRECTED via type change event {e}\n")
                            break
                # drop the z-coordinate from the event location
                self.building_locs[unit.id] = (event.location[:2], unit.owner.team_id, unit.finished_at,
                                               is_mining_loc(self.resource_locs, event.location[:2]) and unit.name in base_names)
                self.lookup[event.frame] = self.building_locs.copy()
                self.keyframes.append(event.frame)


    def handleBasicCommandEvent(self, event, replay):
        if event.ability_name.startswith("Lift"):
            selects = event.active_selection
            flying_selects = [obj for obj in selects if sorted(obj.type_history.items())[0][1].name in flying_buildings]
            target_type = event.ability_name[4:]
            targets = [obj for obj in flying_selects if get_current_name(obj, event.frame - 1) == target_type]
            if len(targets) == 0:
                self.logger.error(f"Lifting {target_type}, but none selected. event={event}, selects={selects}, names={[get_current_name(s, event.frame - 1) for s in selects]}")
            else:
                # assume all selected building of the target type lift off
                # TODO is this true?
                for unit in targets:
                    if unit.id not in self.building_locs:
                        self.logger.info(f"\nlanded building {unit} should be in dict at {event.frame}: {replay.filename}, {event}")
                        for e in replay.events[::-1]:
                            if unit.finished_at is not None and unit.finished_at < e.frame < event.frame and isinstance(e, sc2reader.events.UnitTypeChangeEvent) and e.unit == unit and e.unit_type_name == target_type:
                                # most recent landing
                                self.logger.info(f"found recent landing {e}, last known loc = {get_last_known_loc(unit.id, e.frame, self.lookup)}, built at {self.lookup[unit.finished_at][unit.id]}")
                                frame, loc = get_last_known_loc(unit.id, e.frame, self.lookup)
                                for f in range(frame + 1, event.frame):
                                    if f in self.lookup:
                                        self.lookup[f][unit.id] = loc
                                self.logger.info(f"CORRECTED using position as of {frame}\n")
                                break
                    else:
                        del self.building_locs[unit.id]

                    self.lookup[event.frame] = self.building_locs.copy()
                    self.keyframes.append(event.frame)

#     def handleUnitTypeChangeEvent(self, event, replay):
#         # current state is flying or new state is flying
#         if ("Flying" in get_current_name(event.unit, event.frame - 1)) != ("Flying" in event.unit_type_name):
#             unit = event.unit
#             # lifting
#             if "Flying" in event.unit_type_name:
#                 print("lifting", unit)
#                 assert unit.id in self.building_locs, f"landed building {unit} should be in dict: {replay.filename}, {event}"
#                 del self.building_locs[unit.id]
#                 self.lookup[event.frame] = self.building_locs.copy()
#             else:
#                 print("landing", unit)
#                 assert unit.id not in self.building_locs, f"flying building {unit} should not be in dict: {replay.filename}, {event}"
#                 # drop the z-coordinate from the event location
#                 self.building_locs[unit.id] = (event.location[:2], unit.owner.team_id, unit.finished_at,
#                                                is_mining_loc(self.resource_locs, event.location[:2]) and unit.name in base_names)
#                 self.lookup[event.frame] = self.building_locs.copy()


    def handleEndGame(self, event, replay):
        try:
            pdict = {}
            for player in replay.players:
                player.bases = {}
                player.base_cluster = {}
                pdict[player.team_id] = player

            old_frames = {p.pid: 0 for p in replay.players}
            for frame in self.keyframes:
                for player in replay.players:
                    player.bases[frame] = {}
                    player.base_cluster[frame] = {}
                    if frame > 0:
                        for f in range(old_frames[player.pid] + 1, frame):
                            player.bases[f] = player.bases[old_frames[player.pid]]
                            player.base_cluster[f] = player.base_cluster[old_frames[player.pid]]
                        old_frames[player.pid] = frame

                for f, ls in self.lookup.items():
                    if f <= frame:
                        locs, teamids, finishes, prefs = zip(*list(ls.values()))
                        unit_ids = list(ls.keys())
                    else:
                        break
                locs = np.array(locs)
                prefs = np.array(prefs)
                finishes = np.array(finishes)

                af = AffinityPropagation(preference=[0 if p else -5000 for p in prefs], random_state=None).fit(locs)
                cluster_centers_indices = af.cluster_centers_indices_
                centers = af.cluster_centers_.tolist()
                labels = af.labels_
                self.logger.debug(f"(frame {frame}): labels = {labels}")
                n_clusters = len(cluster_centers_indices)

                # mining location? must be separate cluster
                new_centers = []
                for k in range(n_clusters):
                    # mining bases in this cluster
                    mining_locs = [(loc, finish) for loc, finish, pref in zip(locs[labels == k], finishes[labels == k], prefs[labels == k]) if pref]
                    if len(mining_locs) > 1:
                        # split up clusters with more than one mining base
                        self.logger.debug(f"(frame {frame}): mining_locs = {mining_locs}")
                        original = min(filter(lambda x: x is not None, mining_locs), key=lambda x: x[1])
                        to_split = [x for x in mining_locs if x[0].tolist() != original[0].tolist()]
                        self.logger.debug(f"(frame {frame}): original = {original}, to_split = {to_split}")
                        for i, (loc, finish) in enumerate(to_split):
                            new_label = n_clusters + i
                            self.logger.debug(f"(frame {frame}): changing {labels[(locs == loc).all(axis=1).nonzero()]} to {new_label}")
                            labels[(locs == loc).all(axis=1).nonzero()] = new_label
                            members = [(loc, finish) for loc, finish, pref in zip(locs[labels == k], finishes[labels == k], prefs[labels == k]) if not pref]
                            for ml, mf in members:
                                if dist(ml, loc) == min(dist(ml, x[0]) for x in [original] + to_split[:i] + to_split[i + 1:]) and mf >= finish:
                                    self.logger.debug(f"(frame {frame}): changing {labels[(locs == ml).all(axis=1).nonzero()]} to {new_label}")
                                    labels[(locs == ml).all(axis=1).nonzero()] = new_label
                        new_centers.append(loc)

                for c in new_centers:
                    cluster_centers_indices = np.append(cluster_centers_indices, (locs == c).all(axis=1).nonzero())
                    n_clusters += 1

                # maximum distance
                new_centers = []
                for loc in locs:
                    if all(dist(loc, c) / self.map_dim > 0.1 for c in centers):  # too far away from any cluster center, should be split
                        if any(dist(loc, select_center(cs)) / self.map_dim <= 0.1 for cs in new_centers):  # close to an already split building, merge
                            _, i = min((dist(loc, select_center(cs)), i) for i, cs in enumerate(new_centers))
                            labels[(locs == loc).all(axis=1).nonzero()] = n_clusters + i
                            new_centers[i].append(tuple(loc))
                        else:  # start a new cluster
                            labels[(locs == loc).all(axis=1).nonzero()] = n_clusters + len(new_centers)
                            new_centers.append([tuple(loc)])

                for cs in new_centers:
                    central = select_center(cs)
                    cluster_centers_indices = np.append(cluster_centers_indices, (locs == central).all(axis=1).nonzero())
                    n_clusters += 1
                # print(f"(frame {frame}): set(labels) = {set(labels)} center indices = {cluster_centers_indices}")
                base_types = {}
                for loc, label in zip(locs, labels):
                    if any(np.array_equal(loc, m) for m in self.mains):
                        base_types[label] = BaseType.MAIN
                    elif label not in base_types and (is_mining_loc(self.resource_clusters, loc)):
                        base_types[label] = BaseType.EXPANSION
                for unit_id, loc, team_id, label in zip(unit_ids, locs, teamids, labels):
                    pdict[team_id].bases[frame][unit_id] = loc
                    pdict[team_id].base_cluster[frame][unit_id] = BaseCluster(label, locs[cluster_centers_indices[label]], base_types.get(label, BaseType.PROXY))

        except:
            print(locs)
            print(replay.filename)
            traceback.print_exc()

        for player in replay.players:
            if frame < replay.frames:
                for f in range(frame + 1, replay.frames + 1):
                    player.bases[f] = player.bases[frame]
                    player.base_cluster[f] = player.base_cluster[frame]
            assert len(player.bases) == replay.frames + 1, f"{len(player.bases)} base entries, {replay.frames} frames {sorted(player.bases.keys())}"

            # TODO save cluster information for detecting scouting of new bases, etc.
            # for k in range(n_clusters):
            #     class_members = labels == k
            #     cluster_center = locs[cluster_centers_indices[k]]
            #     player = replay.player[teamids[cluster_centers_indices[k]]]
                
if __name__ == "__main__":
    # pro replays over 1000 seconds long
    files = ['replays/spawningtool_52661.SC2Replay',
             'replays/spawningtool_55197.SC2Replay',
             'replays/spawningtool_54185.SC2Replay',
             'replays/spawningtool_56994.SC2Replay',
             'replays/spawningtool_55200.SC2Replay',
             'replays/spawningtool_55942.SC2Replay',
             'replays/spawningtool_52636.SC2Replay',
             'replays/spawningtool_47751.SC2Replay',
             'replays/spawningtool_47709.SC2Replay',
             'replays/spawningtool_52635.SC2Replay',
             'replays/spawningtool_59082.SC2Replay',
             'replays/spawningtool_58796.SC2Replay',
             'replays/spawningtool_52662.SC2Replay',
             'replays/spawningtool_40787.SC2Replay']
    sc2reader.engine.register_plugin(APMTracker())
    sc2reader.engine.register_plugin(SelectionTracker())
    sc2reader.engine.register_plugin(ActiveSelection())
    sc2reader.engine.register_plugin(BaseTracker())

    start = time.perf_counter()
    replays = [sc2reader.load_replay(f) for f in files]
    print(f"loading {len(files)} replays each over 1000 seconds took {time.perf_counter() - start} seconds")

    # fig, ax = plt.subplots()
    # fig.set_tight_layout(True)
    # def update(r, frame):
    #     ax.clear()
    #     colors = " br"
    #     for k in range(len({c.label for c in n_clusters):
    #         class_members = labels == k
    #         cluster_center = locs[cluster_centers_indices[k]]
    #         col = colors[teamids[cluster_centers_indices[k]]]
    #         ax.plot(locs[class_members, 0], locs[class_members, 1], col + '.')
    #         ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #                  markeredgecolor='k', markersize=14)
    #         for loc in locs[class_members]:
    #             ax.plot([cluster_center[0], loc[0]], [cluster_center[1], loc[1]], col)
    #     ax.text(1, 1, f"{frame // 22.4}s          {r.filename}")
    #     return ax
