"""
Plugins to track bases and mining locations, based on https://github.com/dsjoerg/ggpyjobs/blob/master/sc2parse/plugins.py
Aaron Bauer
Carleton College
August 2020
"""

import traceback
from sklearn.cluster import AffinityPropagation
import numpy as np
import warnings
warnings.simplefilter('ignore', UserWarning)

base_names_tier_one = set(["Hatchery", "Nexus", "CommandCenter"])
base_names = set(["Hatchery", "Lair", "Hive", "Nexus", "CommandCenter", "CommandCenterFlying", "OrbitalCommand", "OrbitalCommandFlying","PlanetaryFortress"])
land_cc_abils = ['LandOrbitalCommand', 'LandCommandCenter']
cc_names = ['OrbitalCommand', 'CommandCenter']

def dist(loc1, loc2):
    return ((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)**0.5


def is_mining_loc(resource_locs, location):
    """
    Uses affinity propagation (https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation)
    to cluster resource locations into possible base locations, returns True if location within 8.5 of a center
    """
    try:
        af = AffinityPropagation(preference=-1000).fit(resource_locs)
        cluster_centers = af.cluster_centers_
        # 8.5 threshold set empirically based on 5-replay sample
        # TODO verify with bigger sample
        return min([dist(location, center) for center in cluster_centers]) < 8.5
    except:
        print(resource_locs)
        traceback.print_exc()


def select_center(cs):
    if len(cs) == 1:
        return cs[0]
    return min(cs, key=lambda c: np.mean([dist(c, x) for x in cs if x != c]))


# TODO differentiate between mining bases and other bases (usually turrets/bunkers/nydus worm/proxy barrack or pylon)
class BaseTracker(object):
    """
    Tracks a player's current bases
    """

    name = "BaseTracker"


    def handleInitGame(self, event, replay):
        self.resource_locs = None
        replay.load_map()
        self.map_dim = (replay.map.map_info.width + replay.map.map_info.height) / 2
        self.under_construction = {}
        self.building_locs = {}
        self.lookup = {}


    def handleUnitBornEvent(self, event, replay):
        if event.unit.is_building:
            # this must be a starting base
            assert event.unit.name in base_names, "born event for non-main building"
            self.building_locs[event.unit.id] = (event.location, event.unit.owner.team_id, 0, True)
            self.lookup[event.frame] = self.building_locs.copy()


    def handleUnitInitEvent(self, event, replay):
        # resources should have been created by now, can initialize resoursce locations
        if self.resource_locs is None:
            self.resource_locs = [o.location for o in replay.objects.values() if o.name and ("Mineral" in o.name or "Vespene" in o.name) and hasattr(o, "location")]
        if event.unit.is_building:
            assert event.unit.id not in self.under_construction, "init event for already under construction unit"
            self.under_construction[event.unit.id] = (event.location, event.unit.owner.team_id, event.unit.finished_at,
                                                      is_mining_loc(self.resource_locs, event.location) and event.unit.name in base_names)


    def handleUnitDoneEvent(self, event, replay):
        if event.unit.is_building:
            assert event.unit.id in self.under_construction and event.unit.id not in self.building_locs, "done event dictionary problem"
            self.building_locs[event.unit.id] = self.under_construction[event.unit.id]
            del self.under_construction[event.unit.id]
            self.lookup[event.frame] = self.building_locs.copy()


    def handleUnitDiedEvent(self, event, replay):
        if event.unit.is_building:
            assert event.unit.id in self.building_locs or event.unit.id in self.under_construction, "died event unit mission from dicts"
            if event.unit.id in self.under_construction:
                del self.under_construction[event.unit.id]
            else:
                del self.building_locs[event.unit.id]
            self.lookup[event.frame] = self.building_locs.copy()


    def handleTargetPointCommandEvent(self, event, replay):
        if event.ability_name in land_cc_abils:
            selecteds = event.active_selection
            selected_ccs = [obj for obj in selecteds if sorted(obj.type_history.items())[0][1].name == 'CommandCenter']
            if len(selected_ccs) == 0:
                raise ValueError(f"Landing an OC/CC but none selected. WTF. event={event}, selecteds={selecteds}")
            else:
                # assert len(selected_ccs) == 1, "multiple ccs selected"
                # assume the order is given to the first selected cc
                # TODO is this true?
                unit = selected_ccs[0]
                assert unit.id in self.building_locs, "selected cc missing from dict"
                # drop the z-coordinate from the event location
                self.building_locs[unit.id] = (event.location[:2], unit.owner.team_id, unit.finished_at,
                                               is_mining_loc(self.resource_locs, event.location[:2]) and unit.name in base_names)
                self.lookup[event.frame] = self.building_locs.copy()


    def handleEndGame(self, event, replay):
        try:
            pdict = {}
            for player in replay.players:
                player.bases = {}
                pdict[player.team_id] = player

            step_size = int(20 * 22.4)
            for frame in range(0, replay.frames + 1, step_size):
                for player in replay.players:
                    player.bases[frame] = {}
                    if frame > 0:
                        for f in range(frame - step_size + 1, frame):
                            player.bases[f] = player.bases[frame - step_size]

                for f, ls in self.lookup.items():
                    if f <= frame:
                        locs, teamids, finishes, prefs = zip(*list(ls.values()))
                        unit_ids = list(ls.keys())
                    else:
                        break
                locs = np.array(locs)
                prefs = np.array(prefs)
                finishes = np.array(finishes)

                af = AffinityPropagation(preference=[0 if p else -5000 for p in prefs]).fit(locs)
                cluster_centers_indices = af.cluster_centers_indices_
                centers = af.cluster_centers_.tolist()
                labels = af.labels_
                n_clusters = len(cluster_centers_indices)

                # mining location? must be separate cluster
                new_centers = []
                for k in range(n_clusters):
                    # mining bases in this cluster
                    mining_locs = [(loc, finish) for loc, finish, pref in zip(locs[labels == k], finishes[labels == k], prefs[labels == k]) if pref]
                    if len(mining_locs) > 1:
                        # split up clusters with more than one mining base
                        original = min(mining_locs, key=lambda x: x[1])
                        to_split = [x for x in mining_locs if x[0].tolist() != original[0].tolist()]
                        for i, (loc, finish) in enumerate(to_split):
                            new_label = n_clusters + i
                            labels[(locs == loc).all(axis=1).nonzero()] = new_label
                            members = [(loc, finish) for loc, finish, pref in zip(locs[labels == k], finishes[labels == k], prefs[labels == k]) if not pref]
                            for ml, mf in members:
                                if dist(ml, loc) == min(dist(ml, x[0]) for x in [original] + to_split[:i] + to_split[i + 1:]) and mf >= finish:
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

                for unit_id, loc, team_id in zip(unit_ids, locs, teamids):
                    pdict[team_id].bases[frame][unit_id] = loc
        except:
            print(r.filename)
            traceback.print_exc()

        for player in replay.players:
            if frame < replay.frames:
                for f in range(frame + 1, replay.frames + 1):
                    player.bases[f] = player.bases[frame]
            assert len(player.bases) == replay.frames + 1, f"{len(player.bases)} base entries, {replay.frames} frames {sorted(player.bases.keys())}"

            # TODO save cluster information for detecting scouting of new bases, etc.
            # for k in range(n_clusters):
            #     class_members = labels == k
            #     cluster_center = locs[cluster_centers_indices[k]]
            #     player = replay.player[teamids[cluster_centers_indices[k]]]
                
