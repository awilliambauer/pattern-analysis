from multiprocessing import Pool, cpu_count
from typing import Dict, List

from load_map_path_data import load_path_data
from generate_replay_info import group_replays_by_map
from collections import namedtuple
import datetime
from functools import partial
import time
import csv


def run(function, replay_filter=lambda x: True, threads=60) -> List[Dict]:
    start_time = time.time()
    results = []
    count = 0
    count_errors = 0
    replay_map_groups = group_replays_by_map(replay_filter)
    replay_count = sum(map(lambda name_replay: len(name_replay[1]), replay_map_groups))
    print(len(replay_map_groups.keys()), "maps", replay_count, "replays")
    with Pool(min(threads, cpu_count())) as pool:
        for map_name_group, replays in replay_map_groups.items():
            if len(replays) == 0:
                continue
            map_path_data = load_path_data(map_name_group)
            if map_path_data is None:
                print("no path data for map", map_name_group)
                continue
            print("loaded path data for map", map_name_group, "with", len(replays), "replays")
            count += len(replays)
            map_time = time.time()
            new_results = pool.map(partial(function, map_path_data=map_path_data), replays)
            results += new_results
            count_errors_this_map = 0
            for result in new_results:
                if result is None:
                    count_errors_this_map += 1
            count_errors += count_errors_this_map
            print(time.time() - map_time, "s to finish processing map", map_name_group, len(replays), "replays,",
                  count_errors_this_map, "errors", count, "/", replay_count, " done")
    deltatime = time.time() - start_time
    print("Analyzed", replay_count, "replays, Run time: ", "{:2d}".format(int(deltatime // 60)), "minutes and",
          "{:05.2f}".format(deltatime % 60),
          "seconds")
    print("Time per replay:", replay_count / deltatime)
    return results


def save(results, output_file):
    output_file_adjusted = output_file[:-4] if output_file.endswith(".csv") else output_file
    output_file_adjusted += datetime.datetime.now().date()
    output_file_adjusted += ".csv"
    first_result_fields = results[0]._fields
    # assume that every result has the same fields
    with open(output_file_adjusted, 'w', newline='') as my_csv:
        events_out = csv.DictWriter(my_csv, fieldnames=first_result_fields)
        events_out.writeheader()
        for result in results:
            if result:
                if not isinstance(result, list):
                    rows = [result]
                else:
                    rows = result
                events_out.writerows(map(lambda tuple: tuple._asdict(), rows))
