# used to load static map path data from files in file_locations.MAP_PATH_DATA_DIRECTORY
# ZL June 2021

import os
from multiprocessing import Pool, cpu_count
from typing import List, Optional
from sc2.position import Point2
import pickle
import file_locations
from names_to_hashes import get_hashes
from hashes_to_names import get_names
from generate_map_path_data import PATH_RESOLUTION, PATH_ROW_CHUNK_SIZE
import time


def _get_grid_points_in_circle(radius, center):
    X = int(radius)  # R is the radius
    for x in range(-X, X + 1):
        Y = int((radius * radius - x * x) ** 0.5)  # bound for y given x
        for y in range(-Y, Y + 1):
            yield x + center[0], y + center[1]


class MapPathData:
    """
    Stores the loaded optimal path data for a map in chunks and can construct optimal paths from that data.
    """

    def __init__(self, map_name, chunks):
        self.map_name = map_name
        self.chunks = chunks

    def find_eligible_dest(self, source, dest):
        """
        Finds the nearest destination to dest for which there exists a path between source and that destination.
        This is found by iterating in a circle of radius 2 around the destination until either a valid path is found or
        there is none.
        :return: the nearest eligible destination or None if there is none.
        """
        untried_dests = list(map(lambda pt: (pt[0] * PATH_RESOLUTION, pt[1] * PATH_RESOLUTION),
                                 _get_grid_points_in_circle(2,
                                                            (dest[0] // PATH_RESOLUTION, dest[1] // PATH_RESOLUTION))))
        for dest in untried_dests:
            chunk_idx = dest[1] // PATH_ROW_CHUNK_SIZE
            if (source[0], source[1], dest[0], dest[1]) in self.chunks[chunk_idx]:
                return dest
        return None

    def get_path(self, source, dest) -> Optional[List[Point2]]:
        """
        Constructs and returns a path, if one exists, from source to dest (or a nearby eligible dest, if dest is not
        pathable).
        :return: A path in the form of a list of waypoints.
        """
        nearest_source = (
            round(source[0] / PATH_RESOLUTION) * PATH_RESOLUTION, round(source[1] / PATH_RESOLUTION) * PATH_RESOLUTION)
        nearest_dest = (
            round(dest[0] / PATH_RESOLUTION) * PATH_RESOLUTION, round(dest[1] / PATH_RESOLUTION) * PATH_RESOLUTION)
        if nearest_source == nearest_dest:
            # you are close enough that we consider you already there
            return None
        nearest_eligible_dest = self.find_eligible_dest(nearest_source, nearest_dest)
        if nearest_eligible_dest is None:
            # no destination in the neighborhood of dest which we can path to
            return None

        chunk_idx = int(nearest_eligible_dest[1] // PATH_ROW_CHUNK_SIZE)  # is int() here redundant?
        path = []
        next_point = nearest_source
        # because we don't store the whole path in the file, we just store the next step, we need to walk
        # along these steps and save them in a list. This becomes the path
        while next_point[0] != nearest_eligible_dest[0] or next_point[1] != nearest_eligible_dest[1]:
            path.append(Point2(next_point))
            if (next_point[0], next_point[1], nearest_eligible_dest[0], nearest_eligible_dest[1]) not in self.chunks[
                chunk_idx]:
                # should hopefully be impossible. This would mean that at one point it thought there was a path from
                # the start to the dest, but as it followed it it suddenly stopped being possible.
                return None
            next_point = self.chunks[chunk_idx][
                (next_point[0], next_point[1], nearest_eligible_dest[0], nearest_eligible_dest[1])]
        if len(path) == 0:
            return None
        return path


def _load_path_data_chunk(path_chunk_file_name):
    print("Loading chunk", path_chunk_file_name)
    with open(file_locations.MAP_PATH_DATA_DIRECTORY + "/" + path_chunk_file_name, "rb") as r:
        return path_chunk_file_name, pickle.load(r)


def load_path_data(map_file_name):
    """
    loads the path data for the given map.
    :param map_file_name: the file name of the map, excluding the file type (i.e. .pkl or .xz or whatever). this
    can either be a str or a list/tuple of strings. if it is a string, it will simply load any chunks that match
    that string. if it is a list/tuple, it will load any chunks that match any of the strings inside that list/tuple.
    not case sensitive.
    :return: a mappathdata instance
    """
    print("Loading path data for map", map_file_name)
    t = time.time()
    files = os.listdir(file_locations.MAP_PATH_DATA_DIRECTORY)
    chunk_file_names = set()
    for file in files:
        if isinstance(map_file_name, list) or isinstance(map_file_name, tuple):
            # in this case, we want to match any file whose name is in the file name group
            for possible_name in map_file_name:
                if possible_name.lower() in file.lower():
                    chunk_file_names.add(file)
                    break
        elif map_file_name.lower() in file.lower():
            chunk_file_names.add(file)
    if len(chunk_file_names) == 0:
        return None
    with Pool(min(cpu_count(), min(20, len(chunk_file_names)))) as pool:
        chunks = pool.map(_load_path_data_chunk, chunk_file_names)
    chunks = [chunk for name, chunk in sorted(chunks, key=lambda name_and_chunk: name_and_chunk[0])]
    print("Time:", time.time() - t,"s")
    return MapPathData(map_file_name, chunks)


def get_all_path_generated_maps():
    """
    :return: a list of all maps for which there exists terrain data. The map names are in English and in filename-style.
    """
    files = []
    for file in os.listdir(file_locations.MAP_PATH_DATA_DIRECTORY):
        if file.endswith("00.pkl"):
            files.append(file.replace("00.pkl", ""))
    return files


def get_all_possible_names(map_name):
    """
    :return: all names associated with the same hashes that the given map_name has been associated with
    """
    hashes = get_hashes(map_name)
    names = []
    for hash in hashes:
        for name in get_names(hash):
            if name not in names:
                names.append(name)
    return names
