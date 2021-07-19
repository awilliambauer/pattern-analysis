import sc2reader
from sc2reader.engine.plugins import SelectionTracker, APMTracker
from selection_plugin import ActiveSelection
from collections import Counter
import json
import csv
import numpy as np
import logging
import sys
from modified_rank_plugin import ModifiedRank

root = logging.getLogger()
# root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

with open("lookup.json") as fp:
    lookup = json.load(fp)

sc2reader.engine.register_plugin(SelectionTracker())
sc2reader.engine.register_plugin(ModifiedRank())
sc2reader.engine.register_plugin(APMTracker())
sc2reader.engine.register_plugin(ActiveSelection())

game_ids = np.loadtxt("game_ids.txt", dtype=np.str)
replays = {}
with open("sc2_test_events.csv", 'w') as fp:
    events_out = csv.DictWriter(fp, fieldnames=["game_id", "uid", "frame", "type"])
    events_out.writeheader()
    for game_id in game_ids:
        r = sc2reader.load_replay("replays/gggreplays_{}.SC2Replay".format(game_id))
        print(game_id, r.build, r.datapack.id)
        if hasattr(r, "marked_error") and r.marked_error:
            print("skipping", r.filename, "as it contains errors")
            print(r.filename, "has build", r.build, "but best available datapack is", r.datapack.id)
            continue
        replays[game_id] = r
        commands = [e for e in r.events if "CommandEvent" in e.name and e.ability]
        n_features = len(lookup["features"])

        for player in r.players:
            uid = player.detail_data['bnet']['uid']
            player_commands = [c for c in commands if c.player.uid == player.uid]
            for command in player_commands:
                try:
                    com_type = lookup[command.name][command.ability.name]
                except KeyError:
                    print("lookup doesn't have {} - {} - {}".format(command.ability.name, command.name,
                                                                    command.active_selection))
                    continue
                if com_type is None:
                    continue
                selection = command.active_selection
                if com_type == "Order":
                    selection = [u for u in command.active_selection if
                                 u.name not in lookup["ignoreunits"] and not u.is_building]
                    if len(selection) == 0:
                        continue
                    if all(u.name in lookup["econunits"] for u in selection):
                        com_type = "OrderEcon"
                    else:
                        com_type = "OrderMilitary"
                # if com_type.startswith("Order"):
                #     for _ in range(len(selection)):
                #         events_out.writerow({"game_id": game_id, "uid": uid, "frame": command.frame, "type": com_type})
                if any(u.name == "Larva" for u in selection) and command.ability.name.startswith("Morph"):
                    larva = [u for u in selection if u.name == "Larva"]
                    morphs = [t for t in r.tracker_events if t.name == "UnitTypeChangeEvent" and
                              command.frame <= t.frame <= command.frame + 30 and t.unit in larva]
                    for _ in range(len(morphs)):
                        events_out.writerow({"game_id": game_id, "uid": uid, "frame": command.frame, "type": com_type})
                else:
                    events_out.writerow({"game_id": game_id, "uid": uid, "frame": command.frame, "type": com_type})
