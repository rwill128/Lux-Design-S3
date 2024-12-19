import unittest
import numpy as np

from run_match import *


class TestIsTilePassable(unittest.TestCase):
    def setUp(self):
        self.known_asteroids = {(5,5)}
        self.sensor_mask = np.zeros((10,10), dtype=bool)
        self.tile_type_map = np.zeros((10,10), dtype=int)  # all empty by default

    def test_known_asteroid(self):
        # Tile is known asteroid
        self.assertFalse(is_tile_passable(5,5,self.sensor_mask,self.tile_type_map,self.known_asteroids))

    def test_visible_asteroid(self):
        # Visible asteroid but not in known set
        self.sensor_mask[3,3] = True
        self.tile_type_map[3,3] = 2
        self.assertFalse(is_tile_passable(3,3,self.sensor_mask,self.tile_type_map,set()))

    def test_visible_empty(self):
        self.sensor_mask[2,2] = True
        self.tile_type_map[2,2] = 0
        self.assertTrue(is_tile_passable(2,2,self.sensor_mask,self.tile_type_map,set()))

    def test_invisible_unknown(self):
        # Not visible, assume passable
        self.assertTrue(is_tile_passable(9,9,self.sensor_mask,self.tile_type_map,set()))
