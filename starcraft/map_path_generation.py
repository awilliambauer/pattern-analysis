import os
import random
from multiprocessing import Pool, cpu_count
import math
from typing import List, Optional
import sc2
from sc2.position import Point3, Point2
import traceback
from loguru import logger
import time
import pickle
import lzma
from names_to_hashes import get_hashes
from hashes_to_names import get_names
from MapAnalyzer.MapData import MapData
from MapAnalyzer.utils import import_bot_instance

import file_locations

GREEN = Point3((0, 255, 0))
RED = Point3((255, 0, 0))
BLUE = Point3((0, 0, 255))
BLACK = Point3((0, 0, 0))


class MATester(sc2.BotAI):

    def __init__(self):
        super().__init__()
        self.map_data = None
        # local settings for easy debug
        self.target = None
        self.base = None
        self.sens = 4
        self.hero_tag = None
        self.p0 = None
        self.p1 = None
        self.influence_grid = None
        self.ramp = None
        self.influence_points = None
        self.path = None
        logger.remove()  # avoid duplicate logging

    async def on_start(self):
        self.map_data = MapData(self, loglevel="DEBUG", arcade=True)

        base = self.townhalls[0]
        self.base = reg_start = self.map_data.where_all(base.position_tuple)[0]
        reg_end = self.map_data.where_all(self.enemy_start_locations[0].position)[0]
        self.p0 = reg_start.center
        self.p1 = reg_end.center
        self.influence_grid = self.map_data.get_pyastar_grid()
        ramps = reg_end.region_ramps
        logger.info(ramps)
        if len(ramps) > 1:
            if self.map_data.distance(ramps[0].top_center, reg_end.center) < self.map_data.distance(ramps[1].top_center,
                                                                                                    reg_end.center):
                self.ramp = ramps[0]
            else:
                self.ramp = ramps[1]
        else:
            self.ramp = ramps[0]

        self.influence_points = [(self.ramp.top_center, 2), (Point2((66, 66)), 18)]

        self.influence_points = self._get_random_influence(25, 5)
        """Uncomment this code block to add random costs and make the path more complex"""
        # for tup in self.influence_points:
        #     p = tup[0]
        #     r = tup[1]
        #     self.map_data.add_cost(p, r=r, arr=self.influence_grid)

        self.path = self.map_data.pathfind(start=self.p0, goal=self.p1, grid=self.influence_grid, sensitivity=self.sens)
        self.hero_tag = self.workers[0].tag

    def get_random_point(self, minx, maxx, miny, maxy):
        return (random.randint(minx, maxx), random.randint(miny, maxy))

    def _get_random_influence(self, n, r):
        pts = []
        for i in range(n):
            pts.append(
                (Point2(self.get_random_point(50, 130, 25, 175)), r))
        return pts

    def _draw_point_list(self, point_list: List = None, color=None, text=None, box_r=None) -> bool:
        if not color:
            color = GREEN
        h = self.get_terrain_z_height(self.townhalls[0])
        for p in point_list:
            p = Point2(p)

            pos = Point3((p.x, p.y, h))
            if box_r:
                p0 = Point3((pos.x - box_r, pos.y - box_r, pos.z + box_r)) + Point2((0.5, 0.5))
                p1 = Point3((pos.x + box_r, pos.y + box_r, pos.z - box_r)) + Point2((0.5, 0.5))
                self.client.debug_box_out(p0, p1, color=color)
            if text:
                self.client.debug_text_world(
                    "\n".join([f"{text}", ]), pos, color=color, size=30,
                )

    def _draw_path_box(self, p, color):
        h = self.get_terrain_z_height(p)
        pos = Point3((p.x, p.y, h))
        box_r = 1
        p0 = Point3((pos.x - box_r, pos.y - box_r, pos.z + box_r)) + Point2((0.5, 0.5))
        p1 = Point3((pos.x + box_r, pos.y + box_r, pos.z - box_r)) + Point2((0.5, 0.5))
        self.client.debug_box_out(p0, p1, color=color)

    async def on_step(self, iteration: int):

        pos = self.map_data.bot.townhalls.ready.first.position
        areas = self.map_data.where_all(pos)
        # logger.debug(areas) # uncomment this to get the areas of starting position
        region = areas[0]
        # logger.debug(region)
        # logger.debug(region.points)
        list_points = list(region.points)
        logger.debug(type(list_points))  # uncomment this to log the region points Type
        logger.debug(list_points)  # uncomment this to log the region points
        hero = self.workers.by_tag(self.hero_tag)
        dist = 1.5 * hero.calculate_speed() * 1.4
        if self.target is None:
            self.target = self.path.pop(0)
        logger.info(f"Distance to next step : {self.map_data.distance(hero.position, self.target)}")
        if self.map_data.distance(hero.position, self.target) > 1:
            hero.move(self.target)

        if self.map_data.distance(hero.position, self.target) <= dist:
            if len(self.path) > 0:
                self.target = self.path.pop(0)
            else:
                logger.info("Path Complete")
        self._draw_path_box(p=self.target, color=RED)  # draw scouting SCV next move point in the path
        self._draw_path_box(p=hero.position, color=GREEN)  # draw scouting SCV position
        self.client.debug_text_world(
            "\n".join([f"{hero.position}", ]), hero.position, color=BLUE, size=30,
        )

        self.client.debug_text_world(
            "\n".join([f"start {self.p0}", ]), Point2(self.p0), color=BLUE, size=30,
        )
        self.client.debug_text_world(
            "\n".join([f"end {self.p1}", ]), Point2(self.p1), color=RED, size=30,
        )

        """
        Drawing Buildable points of our home base ( Region ) 
        Feel free to try lifting the Command Center,  or building structures to see how it updates
        """
        self._draw_point_list(self.base.buildables.points, text='*')

        """
        Drawing the path for our scouting SCV  from our base to enemy's Main
        """
        self._draw_point_list(self.path, text='*', color=RED)


PATH_ROW_CHUNK_SIZE = 10
PATH_RESOLUTION = 5


def lower_resolution(path):
    if path is None:
        return None
    new_path = []
    current_idx = 0
    current_point = None
    while current_point != path[-1]:
        current_point = path[current_idx]
        adjusted_point = ((current_point[0] // PATH_RESOLUTION) * PATH_RESOLUTION,
                          (current_point[1] // PATH_RESOLUTION) * PATH_RESOLUTION)
        if len(new_path) == 0 or new_path[-1] != adjusted_point:
            new_path.append(adjusted_point)
        current_idx += 1
    return new_path


def generate_paths(file):
    try:
        start_time = time.time()
        with lzma.open("map_data/" + file, "rb") as f:
            raw_game_data, raw_game_info, raw_observation = pickle.load(f)

            bot = import_bot_instance(raw_game_data, raw_game_info, raw_observation)

            map_data = MapData(bot)

        print("Generating paths for map", file, "at resolution", PATH_RESOLUTION, "with chunk size",
              PATH_ROW_CHUNK_SIZE, "units")
        astar_grid = map_data.get_pyastar_grid()
        chunk_count = math.ceil(astar_grid.shape[1] / PATH_ROW_CHUNK_SIZE)
        # we will split the map into chunks based on y value so that we don't have to keep all the paths in memory
        for chunk in range(0, chunk_count):
            print("Map", file, ", chunk", chunk)
            paths = {}
            y_offset = chunk * PATH_ROW_CHUNK_SIZE
            for x1 in range(0, astar_grid.shape[0], PATH_RESOLUTION):
                for y1_relative in range(0, PATH_ROW_CHUNK_SIZE, PATH_RESOLUTION):
                    y1 = y1_relative + y_offset
                    for x2 in range(0, astar_grid.shape[0], PATH_RESOLUTION):
                        for y2 in range(0, astar_grid.shape[1], PATH_RESOLUTION):
                            if x1 != x2 or y1 != y2:  # if equal, unit is already there
                                best_path = lower_resolution(map_data.pathfind((x1, y1), (x2, y2), astar_grid))
                                if best_path is not None:
                                    if best_path[0] == (x1, y1):
                                        if len(best_path) != 1:
                                            paths[(x1, y1, x2, y2)] = best_path[1]
                                    else:
                                        paths[(x1, y1, x2, y2)] = best_path[0]
            with open("map_path_data_better_optimized/" + file[:-3] + str(chunk).zfill(2) + ".pkl", "wb") as f:
                pickle.dump(paths, f)
                f.close()
                print("Saving path data chunk", chunk, "for map", file)
        print("Generated paths for map", map_data.map_name, "in", time.time() - start_time, "seconds")
    except Exception as e:
        print("Exception generating path data for map", file, e)
        traceback.print_exc()
        return


def create_path_data():
    files = os.listdir(file_locations.PICKLED_MAP_DATA_DIRECTORY)
    # files = files[:8]  # just limiting map count for debugging
    # files = [
    #     "TritonLE.xz",
    #     "LightshadeLE.xz",
    #     "RomanticideLE.xz",
    #     "OxideLE.xz",
    #     "Turbocruise84LE.xz",
    #     "PurityandIndustryLE.xz",
    #     "BlackburnLE.xz",
    #     "BeckettIndustriesLE.xz",
    #     "StasisLE.xz"
    # ]
    pool = Pool(min(cpu_count(), 9))
    results = pool.map(generate_paths, files)
    pool.close()
    pool.join()


class MapPathData:
    def __init__(self, map_name, chunks):
        self.map_name = map_name
        self.chunks = chunks

    def get_path(self, source, dest) -> Optional[List[Point2]]:
        print("getting path")
        nearest_source_x = round(source[0] / PATH_RESOLUTION) * PATH_RESOLUTION
        nearest_source_y = round(source[1] / PATH_RESOLUTION) * PATH_RESOLUTION
        nearest_dest_x = round(dest[0] / PATH_RESOLUTION) * PATH_RESOLUTION
        nearest_dest_y = round(dest[1] / PATH_RESOLUTION) * PATH_RESOLUTION
        if nearest_dest_x == nearest_source_x and nearest_dest_y == nearest_source_y:
            return None
        start_chunk_idx = int(nearest_source_y // PATH_ROW_CHUNK_SIZE)  # is int() here redundant?
        path = []
        if (nearest_source_x, nearest_source_y, nearest_dest_x, nearest_dest_y) not in self.chunks[start_chunk_idx]:
            return None
        next_point = self.chunks[start_chunk_idx][(nearest_source_x, nearest_source_y, nearest_dest_x, nearest_dest_y)]
        while next_point[0] != nearest_dest_x or next_point[1] != nearest_dest_y:
            print("this point:",next_point)
            path.append(Point2(next_point))
            next_chunk_idx = int(next_point[1] // PATH_ROW_CHUNK_SIZE)
            if (next_point[0], next_point[1], nearest_dest_x, nearest_dest_y) not in self.chunks[next_chunk_idx]:
                return path
            next_point = self.chunks[next_chunk_idx][(next_point[0], next_point[1], nearest_dest_x, nearest_dest_y)]
            print("next point:",next_point)
        if len(path) == 0:
            return None
        print("got path")
        return path


# credit timgeb https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
def is_iterable(thing):
    try:
        iter(thing)
        return True
    except TypeError:
        return False


def load_path_data_chunk(path_chunk_file_name):
    print("loading chunk", path_chunk_file_name)
    with open("map_path_data_better_optimized" + "/" + path_chunk_file_name, "rb") as r:
        return pickle.load(r)


def load_path_data(map_file_name):
    print("loading map", map_file_name)
    files = os.listdir("map_path_data_better_optimized")
    chunk_file_names = set()
    for file in files:
        if isinstance(map_file_name, list) or isinstance(map_file_name, tuple):
            for possible_name in map_file_name:
                if possible_name.lower() in file.lower():
                    chunk_file_names.add(file)
                    break
        elif map_file_name in file:
            chunk_file_names.add(file)
    if len(chunk_file_names) == 0:
        return None
    chunks = []
    for chunk_file_name in sorted(chunk_file_names):
        chunks.append(load_path_data_chunk(chunk_file_name))
    return MapPathData(map_file_name, chunks)


def get_all_path_generated_maps():
    files = []
    for file in os.listdir("map_path_data"):
        if file.endswith("00.xz"):
            files.append(file[:-5])
    return files


def get_all_possible_names(map_name):
    hashes = get_hashes(map_name)
    names = []
    for hash in hashes:
        for name in get_names(hash):
            if name not in names:
                names.append(name)
    return names


def optimize_map(map):
    print("optimizing", map)
    map_path_data = load_path_data(map)
    idx = 0
    for chunk in map_path_data.chunks:
        paths = {}
        for start_pos, dict_1 in chunk.items():
            for dest_pos, path in dict_1.items():
                if path is not None:
                    if start_pos not in paths:
                        paths[start_pos] = {}
                    paths[start_pos][dest_pos] = path

        with open("map_path_data_optimized/" + map + str(idx).zfill(2) + ".pkl", "wb") as f:
            pickle.dump(paths, f)
            f.close()
            print("Saving path data chunk", idx, "for map", map)
        idx += 1


def test():
    maps = get_all_path_generated_maps()
    with Pool(min(cpu_count(), 60)) as pool:
        pool.map(optimize_map, maps)


if __name__ == "__main__":
    test()
    # load_path_data("InterloperLE")
    # print(load_path_data("16-BitLE").get_path((25, 25), (75, 75)))
