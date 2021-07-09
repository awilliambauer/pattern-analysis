from typing import List
from sc2.position import Point2
from collections import deque

movement_speeds = {
    "Overlord": 0.902,
    "Queen": 1.31,
    "LiftedBuilding": 1.31,
    "SpineCrawler": 1.4,
    "SporeCrawler": 1.4,
    "BroodLord": 1.97,
    "BurrowedRoach": 1.97,
    "BurrowedCreepRoach": 2.56,
    "Carrier": 2.62,
    "Mothership": 2.62,
    "High Templar": 2.62,
    "Observer": 2.62,
    "ActivatedUpgradedVoidRay": 2.62,
    "Battlecruiser": 2.62,
    "Thor": 2.62,
    "Locust": 2.62,
    "Overseer": 2.62,
    "UpgradedOverlord": 2.62,
    "BurrowedInfestor": 2.8,
    "ActivatedVoidRay": 2.89,
    "Colossus": 3.15,
    "Disruptor": 3.15,
    "Immortal": 3.15,
    "Sentry": 3.15,
    "Tempest": 3.15,
    "Zealot": 3.15,
    "Hellbat": 3.15,
    "Marauder": 3.15,
    "Marine": 3.15,
    "SiegeTank": 3.15,
    "AssaultViking": 3.15,  # perhaps it has some other internal name?
    "UndisguisedChangeling": 3.15,
    "Hydralisk": 3.15,
    "Infestor": 3.15,
    "Roach": 3.15,
    "SwarmHost": 3.15,
    "Adept": 3.5,
    "Baneling": 3.5,
    "CreepQueen": 3.5,
    "CreepSpineCrawler": 3.5,
    "CreepSporeCrawler": 3.5,
    "BurrowedCreepInfestor": 3.64,
    "CreepLocust": 3.66,
    "VoidRay": 3.85,
    "Banshee": 3.85,
    "FighterViking": 3.85,
    "Ravager": 3.85,
    "Archon": 3.94,
    "DarkTemplar": 3.94,
    "Probe": 3.94,
    "UpgradedObserver": 3.94,
    "Ghost": 3.94,
    "MULE": 3.94,
    "SCV": 3.94,
    "WidowMine": 3.94,
    "Drone": 3.94,
    "UpgradedHydralisk": 3.94,
    "CreepInfestor": 4.09,
    "CreepRoach": 4.09,
    "CreepSwarmHost": 4.09,
    "CreepHydralisk": 4.09,
    "Stalker": 4.13,
    "WarpPrism": 4.13,
    "UpgradedMedivac": 4.13,
    "Raven": 4.13,
    "UpgradedBaneling": 4.13,
    "Lurker": 4.13,
    "Ultralisk": 4.13,
    "Viper": 4.13,
    "Zergling": 4.13,
    "UpgradedRoach": 4.2,
    "CreepBaneling": 4.55,
    "UpgradedLurker": 4.55,
    "UpgradedVoidRay": 4.65,
    "UpgradedZealot": 4.72,
    "Corruptor": 4.72,
    "UpgradedOverseer": 4.72,
    "Cyclone": 4.72,
    "Liberator": 4.72,
    "ActivatedMarauder": 4.72,
    "ActivatedMarine": 4.72,
    "UpgradedUltralisk": 4.95,
    "CreepRavager": 5.0,
    "UpgradedBanshee": 5.25,
    "Reaper": 5.25,
    "UpgradedWarpPrism": 5.36,
    "UpgradedCreepBaneling": 5.37,
    "Broodling": 5.37,
    "CreepLurker": 5.37,
    "CreepUltralisk": 5.37,
    "UpgradedCreepUltralisk": 5.37,
    "CreepZergling": 5.37,
    "UpgradedCreepRoach": 5.46,
    "Shade": 5.6,
    "Oracle": 5.6,
    "Mutalisk": 5.6,
    "UpgradedCreepLurker": 5.91,
    "ActivatedMedivac": 5.94,
    "Phoenix": 5.95,
    "PurificationNova": 5.95,
    "Hellion": 5.95,
    "UpgradedZergling": 6.57,
    "UpgradedCreepZergling": 8.54,
    "UpgradedActivatedZealot": 10.4,
    "Interceptor": 10.5
}


def get_movement_speed(unit_name, **options):
    modified_unit_name = unit_name
    if "Upgraded" in options and options["Upgraded"]:
        modified_unit_name += "Upgraded"
    if "Activated" in options and options["Activated"]:
        modified_unit_name += "Activated"
    if "Creep" in options and options["Creep"]:
        modified_unit_name += "Creep"
    return movement_speeds[unit_name] / 22.4  # convert units per second into units per frame


flying_units = ["Overlord"]


def is_flying_unit(unit_name):
    return unit_name in flying_units


# credit kindall https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator
def window(seq, n=2):
    it = iter(seq)
    win = deque((next(it, None) for _ in range(n)), maxlen=n)
    yield win
    append = win.append
    for e in it:
        append(e)
        yield win


def get_position_estimate_along_path(path: List[Point2], start_time, current_time, unit_speed) -> (float, float):
    estimate_distance_traveled = (current_time - start_time) * unit_speed  # unit speed in units per tick
    current_distance_traveled = 0.0
    if len(path) == 1:
        return path[0]

    for (prev, current) in window(path):
        remaining_distance = estimate_distance_traveled - current_distance_traveled
        distance_between_two = current.distance_to(prev)
        if remaining_distance < distance_between_two:
            # we are somewhere in between the prev and current
            direction = current - prev
            direction = direction.normalized
            return prev + direction * remaining_distance
        current_distance_traveled += distance_between_two
    # if we did not end somewhere in between two of the points we must be at the end
    return path[-1]
