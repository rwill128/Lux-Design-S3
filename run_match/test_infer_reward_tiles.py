import unittest
from run_match import *

class TestInferRewardTiles(unittest.TestCase):
    def test_no_extra_gain(self):
        last_unit_positions = [(1,1)]
        relic_tile_data = {
            (10,10): {(1,1): {"tested":False,"reward_tile":False}}
        }
        # No points difference
        updated = infer_reward_tiles(last_unit_positions, relic_tile_data, 100, 100)
        self.assertFalse(updated[(10,10)][(1,1)]["reward_tile"])

    def test_extra_gain_single_tile(self):
        last_unit_positions = [(1,1)]
        relic_tile_data = {
            (10,10): {(1,1): {"tested":False,"reward_tile":False}}
        }
        # Gained one extra point
        updated = infer_reward_tiles(last_unit_positions, relic_tile_data, 100, 101)
        self.assertTrue(updated[(10,10)][(1,1)]["reward_tile"])
        self.assertTrue(updated[(10,10)][(1,1)]["tested"])

    def test_extra_gain_multiple_tiles(self):
        last_unit_positions = [(1,1),(2,2),(3,3)]
        relic_tile_data = {
            (10,10): {
                (1,1): {"tested":False,"reward_tile":False},
                (2,2): {"tested":False,"reward_tile":False},
                (3,3): {"tested":False,"reward_tile":False}
            }
        }
        # Gained 2 extra points
        updated = infer_reward_tiles(last_unit_positions, relic_tile_data, 100, 102)
        reward_count = sum(d["reward_tile"] for d in updated[(10,10)].values())
        self.assertEqual(reward_count,2)
