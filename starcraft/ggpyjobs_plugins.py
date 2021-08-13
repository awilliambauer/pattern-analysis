import sc2reader
from sc2reader.utils import Length


def ZergMacroTracker(replay):
    INJECT_TIME = 40 * 16

    debug_injects = False

    for player in replay.players:
        player.hatches = dict()

    efilter = lambda e: e.name.endswith("TargetUnitCommandEvent") and hasattr(e, "ability") and e.ability_name == "SpawnLarva"
    for event in filter(efilter, replay.events):
        owner = event.player
        target_hatch = event.target
        target_hatch_id = event.target.id
        if target_hatch_id not in owner.hatches:
            target_hatch.injects = [event.frame]
            owner.hatches[target_hatch_id] = target_hatch
        else:
            # If not enough time has passed, the last one didn't happen
            if event.frame - target_hatch.injects[-1] < INJECT_TIME:
                # print "Previous inject on {0} at {1} failed".format(target_hatch, target_hatch.injects[-1])
                target_hatch.injects[-1] = event.frame
            else:
                target_hatch.injects.append(event.frame)

    # Consolidate Lairs and Hives back into the originating Hatcheries
    for player in replay.players:
        if player.play_race != 'Zerg': continue
        player.race_macro_count = 0
        hatches = dict()
        macro_hatches = dict()
        for final_hatch in player.hatches.values():
            hatches[(final_hatch.id,final_hatch.type)] = final_hatch

            if len(final_hatch.injects) == 0:
                continue
            # Throw out injects that don't finish
            final_hatch.injects.sort()
            died_at = final_hatch.died_at if final_hatch.died_at is not None else replay.frames
            if final_hatch.injects[-1]+INJECT_TIME > died_at:
                final_hatch.injects.pop()

            if len(final_hatch.injects) > 1:
                # Should we use the "most upgraded" type here?
                macro_hatches[(final_hatch.id,final_hatch.type)] = final_hatch

                # Calculate Utilization
                final_hatch.inject_time = len(final_hatch.injects) * INJECT_TIME
                final_hatch.active_time = died_at - final_hatch.injects[0]
                final_hatch.utilization = final_hatch.inject_time/float(final_hatch.active_time)
            player.race_macro_count += len(final_hatch.injects)

        player.hatches = hatches
        player.macro_hatches = macro_hatches

        if len(player.macro_hatches) > 0:
            total_inject_time = sum([hatch.inject_time for hatch in player.macro_hatches.values()])
            total_active_time = sum([hatch.active_time for hatch in player.macro_hatches.values()])
            player.race_macro = total_inject_time/float(total_active_time)
        else:
            player.race_macro = 0

        # if debug_injects:
        #     print 'Player {} active total: {}, inject total: {}, race macro: {:.15f}'.format(player, str(Length(seconds=int(total_active_time)/16)), str(Length(seconds=int(total_inject_time)/16)), player.race_macro)
        #     print 'Bases'
        #     print '---------------------------------------'
        #     for hatch in player.hatches.values():
        #         if hasattr(hatch, 'injects'):
        #             if hasattr(hatch, 'active_time'):
        #                 print 'Active:{} Injects:{} Died:{} '.format(str(Length(seconds=int(hatch.active_time)/16)), str(Length(seconds=int(hatch.inject_time)/16)), str(Length(seconds=int(hatch.died_at)/16)))
        #             print "{} Injects: ".format(hatch),
        #             for inject in hatch.injects:
        #                 print str(Length(seconds=int(inject)/16)) + ' ',
        #             print ''
        #     print '---------------------------------------'
        #     print ''
        #     print ''


    return replay

def ProtossTerranMacroTracker(replay):
    """ Implements: protoss/terran max-energy tracking
        Requires: SelectionTracker
                  BaseTracker
    """

    # we simulate nexus/orbital energy for each player's bases,
    # assuming that energy starts accumulating at finished_at.

    # We could have used ordered_at + BASE_BUILD_TIME, but that would
    # be inaccurate and harsh for bases that were built by a worker
    # that had a ways to travel.  finished_at is sometimes later than
    # the true completion time, but let's be generous.  if Nexus/Orbital
    # energy simulates below zero, then we clamp it to zero and the
    # simulation is more accurate going forward from that point.

    # we add the following fields to Nexus/Orbital objects:
    #  as_of_frame        the frame at which the following fields are valid
    #  start_energy_frame the frame at which energy accumulation started
    #  energy             the energy of the Nexus/Orbital at time as_of_frame
    #  maxouts            list of times at which we were at max energy, each being [start_frame, end_frame]
    #  chronoboosts       list of chronoboost frames
    #  mules              list
    #  scans              list
    #  supplydrops        list

    # Algorithm:
    # for each chronoboost/mule/etc, in order
    #  pick which base did it
    #  update that base's stats to the time just before the event
    #  update the base for the event
    # game over, final adjustment to all base stats
    # compute each player's macro score

    CHRONO_COST = 25
    ORBITAL_ABILITY_COST = 50
    ENERGY_REGEN_PER_FRAME = 0.5625 / 16.0
    NEXUS_MAX_ENERGY = 100
    ORBITAL_MAX_ENERGY = 200
    ORBITAL_START_ENERGY = 50

    # pick the closest base with enough energy, as per
    # http://www.teamliquid.net/forum/viewmessage.php?topic_id=406590
    #
    # or if no bases have enough energy, or they dont have locations,
    # return the base with the most energy.
    #
    def which_base(event):
        owner = event.player
        min_base_distance = 999999
        min_base = None
        selected_objects = event.active_selection
        selected_bases = [obj for obj in selected_objects if obj.name in ['Nexus', 'OrbitalCommand', 'OrbitalCommandFlying']]

        # sometimes due to event-ordering madness, we think there are
        # no bases selected at the time of an event.  in those
        # cases, we'll attempt to recover by just looking at all the
        # player's bases.
        if len(selected_bases) == 0:
            selected_bases = event.player.bases

        if len(selected_bases) == 0:
            print("No Bases selected and player has no bases registered yet. I give up")
            return None

        if event.player.play_race == 'Protoss':
            ability_cost = CHRONO_COST
        else:
            ability_cost = ORBITAL_ABILITY_COST

        #print "event {}".format(event)
        #print "which {} {}".format(event.location[0], event.location[1])
        for base in selected_bases:
            update_to_frame(base, event.frame, "which", None)
            if hasattr(base, 'energy') and hasattr(base, 'location') and base.energy > ability_cost:
                if not hasattr(event, 'location'):
                    print("How can this event have no location?!")
                    print("event {}".format(event.__str__().encode('utf-8')))
                    return None
                #print "considering base {} by location/energy".format(base)
                diff_x = base.location[0] - event.location[0]
                diff_y = base.location[1] - event.location[1]
                sqdiff = diff_x * diff_x + diff_y * diff_y
                if sqdiff < min_base_distance:
                    min_base_distance = sqdiff
                    min_base = base
                #print "picking base at location {} {}".format(base.location[0], base.location[1])

        if min_base is not None:
            return min_base

        max_energy = -1
        max_energy_base = None
        for base in selected_bases:
            #print "considering base {} with energy {}".format(base, base.energy if hasattr(base, 'energy') else 'None')
            if hasattr(base, 'energy'):
                if base.energy > max_energy:
                    max_energy = base.energy
                    max_energy_base = base
            #print "picking base with energy {}".format(max_energy)
        # print(event.ability_name, event.active_selection, max_energy_base)
        return max_energy_base


    # roll the base state forward to the given frame. no energy abilities
    # occur in the intervening time.
    def update_to_frame(base, frame, reason, event):
        if frame is None:
            frame = replay.frames
        #print "Updating base {} to frame {}".format(base, frame)
        if not hasattr(base, "start_energy_frame"):
            if getattr(base, 'finished_at', None) != None:
                base.maxouts = []
                if base.name == 'Nexus':
                    base.energy = 0
                    base.chronoboosts = []
                    if base.finished_at == 0:
                        base.start_energy_frame = 0
                    else:
                        base.start_energy_frame = base.finished_at
                elif base.name == 'OrbitalCommand' or base.name == 'OrbitalCommandFlying':
                    base.energy = ORBITAL_START_ENERGY
                    base.mules = []
                    base.scans = []
                    base.supplydrops = []

                    #print "Looking for Orbitals first time as an Orbital, history={}".format(base.type_history.items())

                    # find the first time we changed type to orbital. It is important
                    # to note that we can change to orbital several times as we lift/land
                    for frame, utype in base.type_history.items():
                        if utype.name == 'OrbitalCommand':
                            base.start_energy_frame = frame
                            break

                else:
                    # only Nexus and OrbitalCommand have energy
                    return

                if not hasattr(base, 'start_energy_frame'):
                  print("cant figure out the start-energy frame, not gonna touch it")
                  return
                base.as_of_frame = base.start_energy_frame
            else:
                #print "this base has no finished_at. not gonna touch it. base={}".format(base)
                return
        if frame < base.as_of_frame:
            # a base that is still as-of its start energy frame may
            # receive update_to_frame calls from an earlier
            # time. thats OK.
            #
            # any other out-of-order scenario is a bug.
            #
            if base.as_of_frame != base.start_energy_frame:
                print("update_to_frame called out of time order. {} before {}. reason={}, event={}".format(frame,
                                                                                                           base.as_of_frame,
                                                                                                           reason,
                                                                                                           event))
            return
        if frame == base.as_of_frame:
            #print "no time has passed, nothing to do. now={}".format(frame)
            return

        if base.name == 'Nexus':
            max_energy = NEXUS_MAX_ENERGY
        else:
            max_energy = ORBITAL_MAX_ENERGY
        # energy was already at max. extend the last maxout and we're
        # done here.
        if base.energy == max_energy:
            base.maxouts[-1][1] = frame
            base.as_of_frame = frame
            return

        new_energy = base.energy + ENERGY_REGEN_PER_FRAME * (frame - base.as_of_frame)

        if new_energy < max_energy:
            base.energy = new_energy
        else:
            energy_to_max = float(max_energy - base.energy)
            time_to_max = energy_to_max / ENERGY_REGEN_PER_FRAME
            maxout_start_time = base.as_of_frame + time_to_max
            base.maxouts.append([maxout_start_time, frame])
            base.energy = max_energy

        base.as_of_frame = frame

    def use_ability(base, event):
        update_to_frame(base, event.frame, "use_ability", None)
        cost = ORBITAL_ABILITY_COST
        base.owner.race_macro_count += 1
        if event.ability_name == 'ChronoBoost':
            cost = CHRONO_COST
            base.chronoboosts.append(event.frame)
        elif event.ability_name == 'CalldownMULE':
            base.mules.append(event.frame)
        elif event.ability_name == 'ExtraSupplies':
            base.supplydrops.append(event.frame)
        else:
            base.scans.append(event.frame)


        base.energy = base.energy - cost
        if base.energy < 0:
            #print "base energy at {} at frame {}. base={}".format(base.energy, frame, base)
            base.energy = 0

    for player in replay.players:
        if player.play_race == "Zerg":
            continue
        player.race_macro_count = 0

    efilter = lambda e: hasattr(e, "ability") and e.ability_name in ['ChronoBoost', 'CalldownMULE', 'ExtraSupplies', 'ScannerSweep']
    # TODO also catch OrbitalLand events and update our estimate of the base's location
    for event in filter(efilter, replay.events):
        base = which_base(event)
        if base is None:
            pass # print("Cant figure out which Base this Chronoboost/Scan/etc was for. Ignoring it :(")
        else:
            update_to_frame(base, event.frame, "ability", event)
            use_ability(base, event)

    for player in replay.players:
        if player.play_race == 'Zerg': continue

        total_maxout_time = 0
        total_active_time = 0

        for base in player.units:
            if base.name in ['Nexus', 'OrbitalCommand', 'OrbitalCommandFlying']:
                died_at = base.died_at if base.died_at is not None else replay.frames
                update_to_frame(base, died_at, "final", None)
                if hasattr(base, 'maxouts') and hasattr(base, 'start_energy_frame'):
                    total_maxout_time = total_maxout_time + sum([(maxout[1] - maxout[0]) for maxout in base.maxouts])
                    base.active_time = (died_at - base.start_energy_frame)
                    total_active_time = total_active_time + base.active_time


        if total_active_time > 0:
            player.race_macro = 1.0 - total_maxout_time/float(total_active_time)
        else:
            player.race_macro = 0

            print('Player {} active total: {}, maxout total: {}, race macro: {:.15f}'.format(player, str(Length(
                seconds=int(total_active_time) / 16)), str(Length(seconds=int(total_maxout_time) / 16)),
                                                                                             player.race_macro))
            print('Bases')
            print('---------------------------------------')
            for base in player.bases:
                if hasattr(base, 'maxouts'):
                    print("Player {} Base {} active time {}".format(player, base, Length(
                        seconds=int(getattr(base, 'active_time', 0) / 16))))
                    if base.name == 'Nexus':
                        print("Chronoboosts: ", end=' ')
                        for boost in base.chronoboosts:
                            print(str(Length(seconds=int(boost) / 16)) + ' ', end=' ')
                        print('')
                    else:
                        for name, thelist in [('Scans: ', base.scans), ('MULEs: ', base.mules), ('Supply: ', base.supplydrops)]:
                            print(name)
                            for frame in thelist:
                                print(str(Length(seconds=int(frame) / 16)) + ' ', end=' ')
                            print('')

                    print("Maxouts: ", end=' ')
                    for maxout in base.maxouts:
                        print(str(Length(seconds=int(maxout[0]) / 16)) + '-' + str(
                            Length(seconds=int(maxout[1]) / 16)) + ' ', end=' ')
                    print('')
            print('---------------------------------------')
            print('')
            print('')

    return replay