# used to generate static map path data from map terrain data
# ZL June 2021

import os
import random
from multiprocessing import Pool, cpu_count
import math
from typing import List
import sc2
from sc2.position import Point3, Point2
import traceback
from loguru import logger
import time
import sys
from itertools import repeat
import numpy as np
import heapq
import pickle
import lzma
try:
    from MapAnalyzer.MapData import MapData
    from MapAnalyzer.utils import import_bot_instance
except ImportError:
    pass

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


PATH_ROW_CHUNK_SIZE = 12
PATH_RESOLUTION = 4


def _shortest_paths(grid, source):
    # Initialize the distance of all the nodes from the source node to infinity
    distance = np.full(grid.shape, np.inf)
    # Distance of source node to itself is 0
    distance[source] = 0
    parents = {}

    # Create a dictionary of { node, distance_from_source }
    queue = [(0, source)]
    heapq.heapify(queue)

    while queue:

        # Get the key for the smallest value in the dictionary
        # i.e Get the node with the shortest distance from the source
        dist, current = heapq.heappop(queue)
        for x in range(-1, 2):
            for y in range(-1, 2):
                if x == 0 and y == 0:
                    continue
                adj = (x + current[0], y + current[1])
                if not ((0 < adj[0] < grid.shape[0]) and 0 < adj[1] < grid.shape[1]):
                    continue

                cost_to_adj = (1.414 if abs(x) == abs(y) else 1) + grid[adj[0]][adj[1]]
                if cost_to_adj != np.inf and distance[adj[0]][adj[1]] > distance[current[0]][current[1]] + cost_to_adj:
                    distance[adj[0]][adj[1]] = distance[current] + cost_to_adj
                    parents[adj] = current
                    heapq.heappush(queue, (distance[adj[0]][adj[1]], adj))
    return parents


def _generate_chunk(lowres_grid, chunk):
    paths = {}
    y_offset = chunk * PATH_ROW_CHUNK_SIZE // PATH_RESOLUTION
    for x1 in range(0, lowres_grid.shape[0]):
        for y1_relative in range(0, PATH_ROW_CHUNK_SIZE // PATH_RESOLUTION):
            y1 = y1_relative + y_offset
            if y1 >= lowres_grid.shape[1]:
                continue
            if lowres_grid[(x1, y1)] == np.inf:
                continue
            pointers = _shortest_paths(lowres_grid, (x1, y1))

            for parent, child in pointers.items():
                highres_parent = (parent[0] * PATH_RESOLUTION, parent[1] * PATH_RESOLUTION)
                highres_child = (child[0] * PATH_RESOLUTION, child[1] * PATH_RESOLUTION)
                paths[
                    (highres_parent[0], highres_parent[1], x1 * PATH_RESOLUTION,
                     y1 * PATH_RESOLUTION)] = highres_child
    return chunk, paths


def generate_paths(file):
    """
    Generates and saves the static optimal path data for the given map terrain data file. First, the map terrain grid
    is lowered in resolution by a factor of PATH_RESOLUTION. Then, we run Djikstra's on every point in the map.
    Djikstra(x,y) gives us a dictionary mapping (a,b): (c,d) such that the best step towards (x,y) from (a,b) is (c,d).
    Thus, after running it for every point on the map, we can construct a dictionary of (a,b,x,y):(c,d), so that for
    every point pair (a,b),(x,y), we know the next step we need to take on the optimal path. We then split this
    dictionary up by the y-coordinate of the goal every PATH_ROW_CHUNK_SIZE units and save each of these chunks in
    files. These can then be loaded later by calling load_path_data in load_map_path_data.py.
    """
    try:
        start_time = time.time()
        with lzma.open(file_locations.MAP_TERRAIN_DATA_DIRECTORY + "/" + file, "rb") as f:
            raw_game_data, raw_game_info, raw_observation = pickle.load(f)

            bot = import_bot_instance(raw_game_data, raw_game_info, raw_observation)

            map_data = MapData(bot)

        print("Generating paths for map", file, "at resolution", PATH_RESOLUTION, "with chunk size",
              PATH_ROW_CHUNK_SIZE, "units")
        highres_grid = map_data.get_pyastar_grid()
        lowres_grid = np.ndarray(
            shape=(highres_grid.shape[0] // PATH_RESOLUTION, highres_grid.shape[1] // PATH_RESOLUTION))
        for x in range(0, highres_grid.shape[0], PATH_RESOLUTION):
            for y in range(0, highres_grid.shape[1], PATH_RESOLUTION):
                lowres_grid[x // PATH_RESOLUTION][y // PATH_RESOLUTION] = highres_grid[x][y]

        chunk_count = math.ceil(highres_grid.shape[1] / PATH_ROW_CHUNK_SIZE)
        with Pool(min(cpu_count(), min(chunk_count, 20))) as pool:
            chunk_data = pool.starmap(_generate_chunk, zip(repeat(lowres_grid), range(0, chunk_count)))
            for chunk, datum in chunk_data:
                with open(file_locations.MAP_PATH_DATA_DIRECTORY + "/" + file[:-3] + str(chunk).zfill(2) + ".pkl",
                          "wb") as f:
                    pickle.dump(datum, f)
                    f.close()
                    print("Saving path data chunk", chunk, "for map", file)
        # we will split the map into chunks based on y value so that we don't have to keep all the paths in memory
        print("Generated paths for map", map_data.map_name, "in", time.time() - start_time, "seconds")
    except Exception as e:
        print("Exception generating path data for map", file, e)
        traceback.print_exc()
        return


def create_path_data():
    """
    Creates and saves all path data for all maps which have terrain data.
    """
    files = os.listdir(file_locations.MAP_TERRAIN_DATA_DIRECTORY)
    for file in files:
        generate_paths(file)


if __name__ == "__main__":
    create_path_data()
