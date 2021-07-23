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
    "VikingAssault": 10,
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
missing_units_vision = set()


def get_unit_vision_radius(unit_name):
    if "Changeling" in unit_name and unit_name != "Changeling":
        return get_unit_vision_radius("Changeling")
    if unit_name not in unit_vision_ranges:
        missing_units_vision.add(unit_name)
        # with open("missing_unit_vision.txt", "a") as f:
        #     try:
        #         already_missing = f.readlines()
        #         if unit_name not in [it.strip() for it in already_missing]:
        #             f.write(unit_name + "\n")
        #     except:
        #         print("missing unit visionx", unit_name)
        return 9  # todo make this unnecessary
    return unit_vision_ranges[unit_name]


movement_speeds = {
    "SiegeTankSieged": 0.0,
    "Larva": 0.0,
    "LurkerEgg": 0.0,
    "LurkerBurrowed": 0.0,
    "BattleHellion": 5.95,
    "WidowMineBurrowed": 0.0,
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
    "HighTemplar": 2.62,
    "Observer": 2.62,
    "ActivatedUpgradedVoidRay": 2.62,
    "Battlecruiser": 2.62,
    "Thor": 2.62,
    "Locust": 2.62,
    "Overseer": 2.62,
    "UpgradedOverlord": 2.62,
    "BurrowedInfestor": 2.8,
    "ActivatedVoidRay": 2.89,
    "LiberatorAG": 0.0,
    "Colossus": 3.15,
    "Disruptor": 3.15,
    "Immortal": 3.15,
    "Sentry": 3.15,
    "Tempest": 3.15,
    "Zealot": 3.15,
    "Hellbat": 3.15,
    "Marauder": 3.15,
    "Marine": 3.15,
    "MarineShield": 3.15,
    "SiegeTank": 3.15,
    "VikingAssault": 3.15,  # perhaps it has some other internal name?
    "Viking": 3.15,
    "MothershipCore": 2.62,
    "OverlordTransport": 0.92,
    "UndisguisedChangeling": 3.15,
    "Changeling": 3.15,
    "Hydralisk": 3.15,
    "Infestor": 3.15,
    "Roach": 3.15,
    "SwarmHost": 3.15,
    "Adept": 3.5,
    "Baneling": 3.5,
    "CreepQueen": 3.5,
    "CreepSpineCrawler": 3.5,
    "CreepSporeCrawler": 3.5,
    "Medivac": 3.5,
    "CreepTumor": 0.0,
    "CreepTumorBurrowed": 0.0,
    "BurrowedCreepInfestor": 3.64,
    "Egg": 0.0,
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
    "ZerglingWings": 6.57,
    "UpgradedCreepZergling": 8.54,
    "UpgradedActivatedZealot": 10.4,
    "Interceptor": 10.5
}

missing_units_movement = set()


def get_unit_movement_speed(unit_name, **options):
    if "Burrowed" in unit_name and ("Roach" not in unit_name and "Infestor" not in unit_name):
        return 0.0
    if "Changeling" in unit_name and unit_name != "Changeling":
        return get_unit_movement_speed(unit_name.replace("Changeling", ""))
    modified_unit_name = unit_name
    if "Upgraded" in options and options["Upgraded"]:
        modified_unit_name += "Upgraded"
    if "Activated" in options and options["Activated"]:
        modified_unit_name += "Activated"
    if "Creep" in options and options["Creep"]:
        modified_unit_name += "Creep"
    if unit_name not in movement_speeds:
        missing_units_movement.add(unit_name)
        # with open("missing_unit_speeds.txt","a") as f:
        #     try:
        #         already_missing = f.readlines()
        #         if unit_name not in [it.strip() for it in already_missing]:
        #             f.write(unit_name + "\n")
        #     except:
        #         print("missing unit",unit_name)
        return 3.5  # todo make this unnecessary!
    return movement_speeds[unit_name] / 22.4  # convert units per second into units per frame


flying_units = ["Observer",
                "WarpPrism",
                "Phoenix",
                "VoidRay",
                "Carrier",
                "Interceptor",
                "Mothership",
                "MothershipCore",
                "Oracle",
                "Tempest",
                "Medivac",
                "Viking",
                "Banshee",
                "Raven",
                "Battlecruiser",
                "PointDefenseDrone",
                "Liberator",
                "Overlord",
                "Overseer",
                "Mutalisk",
                "Corruptor",
                "Brood Lord",
                "Viper"]


def is_flying_unit(unit_name):
    return unit_name in flying_units


def can_produce(building_name, unit_name):
    return False