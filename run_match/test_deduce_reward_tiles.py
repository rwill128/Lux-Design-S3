import unittest
import json
import os
import numpy as np
from run_match import BestAgentAttacker


class TestDeduceRewardTiles(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.agent = BestAgentAttacker("player_0", {})
        self.test_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")

    def test_confidence_persistence(self):
        """Test that confidence values persist between agent instances"""
        # First agent instance
        agent1 = BestAgentAttacker("player_0", {})
        
        # Simulate a tile with positive confidence
        test_tile = (5, 5)
        agent1.tile_confidence[test_tile] = 1.5
        BestAgentAttacker._global_tile_confidence[test_tile] = 1.5

        # Create second agent instance
        agent2 = BestAgentAttacker("player_0", {})

        # Check confidence decay
        expected_confidence = 1.5 * agent2.CONFIDENCE_DECAY
        self.assertIn(test_tile, agent2.tile_confidence)
        self.assertAlmostEqual(
            agent2.tile_confidence[test_tile],
            expected_confidence
        )

    def test_multi_unit_confidence_distribution(self):
        """Test confidence distribution with multiple units"""
        # Mock observation with multiple units on unknown tiles
        obs = {
            "team_points": {0: 2},  # Gained 2 points
            "units": {"position": {0: np.array([(1, 1), (2, 2)])}},
            "units_mask": {0: np.array([True, True])},
            "steps": 1
        }
        
        # Add tiles to unknown set
        self.agent.unknown_tiles.add((1, 1))
        self.agent.unknown_tiles.add((2, 2))
        
        # Run deduction
        self.agent.deduce_reward_tiles(obs)
        
        # Check confidence distribution
        self.assertIn((1, 1), self.agent.tile_confidence)
        self.assertIn((2, 2), self.agent.tile_confidence)
        # Each tile should get half the gain rate
        self.assertAlmostEqual(self.agent.tile_confidence[(1, 1)], 1.0)
        self.assertAlmostEqual(self.agent.tile_confidence[(2, 2)], 1.0)

    def test_negative_confidence_threshold(self):
        """Test that tiles are marked as non-rewards when confidence drops"""
        test_tile = (3, 3)
        
        # Initialize tile as unknown with low confidence
        self.agent.unknown_tiles.add(test_tile)
        self.agent.tile_confidence[test_tile] = -1.5

        # Mock observation with no point gain
        obs = {
            "team_points": {0: 0},
            "units": {"position": {0: np.array([(3, 3)])}},
            "units_mask": {0: np.array([True])},
            "steps": 1
        }
        
        # Run deduction
        self.agent.deduce_reward_tiles(obs)

        # Check tile was marked as non-reward
        self.assertIn(test_tile, self.agent.not_reward_tiles)
        self.assertNotIn(test_tile, self.agent.unknown_tiles)
        self.assertNotIn(test_tile, self.agent.known_reward_tiles)

    def test_pattern_persistence(self):
        """Test that relic patterns persist between agent instances"""
        # First agent instance
        agent1 = BestAgentAttacker("player_0", {})
        relic_pos = (10, 10)
        test_pattern = {(0, 0), (1, 0), (0, 1)}
        
        # Set up pattern data - use frozenset since sets aren't hashable
        pattern_set = {frozenset(test_pattern)}
        agent1.relic_patterns[relic_pos] = pattern_set
        BestAgentAttacker._global_relic_patterns[relic_pos] = pattern_set
        
        # Create second agent instance
        agent2 = BestAgentAttacker("player_0", {})
        
        # Check pattern persistence
        self.assertIn(relic_pos, agent2.relic_patterns)
        self.assertEqual(agent2.relic_patterns[relic_pos], pattern_set)

    def load_scenario(self, scenario_file):
        """Load a test scenario from JSON file and convert arrays to numpy format."""
        with open(os.path.join(self.test_data_dir, scenario_file), 'r') as f:
            scenario = json.load(f)
            
        # Convert observation arrays to numpy format
        obs = scenario["obs_before"]
        if "units_mask" in obs:
            # Handle nested array structure
            obs["units_mask"] = {
                0: np.array(obs["units_mask"][0], dtype=bool),
                1: np.array(obs["units_mask"][1], dtype=bool)
            }
        if "units" in obs and "position" in obs["units"]:
            # Handle nested array structure for positions
            obs["units"]["position"] = {
                0: np.array([pos for pos in obs["units"]["position"][0] if pos[0] != -1]),
                1: np.array([pos for pos in obs["units"]["position"][1] if pos[0] != -1])
            }
        if "team_points" in obs:
            obs["team_points"] = {
                0: obs["team_points"][0],
                1: obs["team_points"][1]
            }
                    
        return scenario
            
    def test_confidence_threshold_reward_marking(self):
        """Test that tiles are marked as rewards above confidence threshold"""
        test_tile = (4, 4)
        
        # Initialize tile as unknown with high confidence
        self.agent.unknown_tiles.add(test_tile)
        self.agent.tile_confidence[test_tile] = 1.5

        # Mock observation with point gain
        obs = {
            "team_points": {0: 1},
            "units": {"position": {0: np.array([(4, 4)])}},
            "units_mask": {0: np.array([True])},
            "steps": 1
        }
        
        # Run deduction
        self.agent.deduce_reward_tiles(obs)

        # Check tile was marked as reward
        self.assertIn(test_tile, self.agent.known_reward_tiles)
        self.assertNotIn(test_tile, self.agent.unknown_tiles)
        self.assertNotIn(test_tile, self.agent.not_reward_tiles)


    def test_scenario_eliminated_tiles_with_multiple_units(self):
        """Test scenario where multiple tiles are eliminated as non-rewards.
        
        This test verifies that the agent correctly identifies tiles as non-rewards
        when units visit them and gain no points, even with multiple units active.
        The scenario involves 14 units exploring the map, with 2 tiles being
        eliminated as non-rewards through direct visits.
        
        Scenario details:
        - 14 active units exploring the map
        - Tiles (12,4) and (12,2) visited with no point gain
        - Both tiles correctly marked as non-rewards
        - No false positives in reward tile identification
        """
        scenario = self.load_scenario("test_scenario_42.json")
        
        # Set up agent state from scenario
        self.agent.possible_reward_tiles = set(map(tuple, scenario["state_before"]["possible_reward_tiles"]))
        self.agent.unknown_tiles = set(map(tuple, scenario["state_before"]["unknown_tiles"]))
        self.agent.not_reward_tiles = set(map(tuple, scenario["state_before"]["not_reward_tiles"]))
        self.agent.known_reward_tiles = set(map(tuple, scenario["state_before"]["known_reward_tiles"]))
        self.agent.last_reward_occupied = set(map(tuple, scenario["state_before"]["last_reward_occupied"]))
        self.agent.last_unknown_occupied = set(map(tuple, scenario["state_before"]["last_unknown_occupied"]))
        self.agent.last_team_points = scenario["state_before"]["last_team_points"]
        self.agent.last_gain = scenario["state_before"]["last_gain"]
        self.agent.last_unit_positions = list(map(tuple, scenario["state_before"]["last_unit_positions"]))
        
        # Run deduction
        self.agent.deduce_reward_tiles(scenario["obs_before"])
        
        # Verify state matches scenario
        self.assertEqual(
            set(map(tuple, scenario["state_after"]["not_reward_tiles"])),
            self.agent.not_reward_tiles,
            "Non-reward tiles don't match expected state"
        )
        self.assertEqual(
            set(map(tuple, scenario["state_after"]["unknown_tiles"])),
            self.agent.unknown_tiles,
            "Unknown tiles don't match expected state"
        )
        self.assertEqual(
            set(map(tuple, scenario["state_after"]["known_reward_tiles"])),
            self.agent.known_reward_tiles,
            "Known reward tiles don't match expected state"
        )
        
    def test_scenario_gained_points_with_multiple_units(self):
        """Test scenario where points are gained with multiple units active.
        
        This test verifies that the agent correctly attributes point gains to the
        appropriate tiles when multiple units are active. The scenario involves
        12 units exploring the map, with a 6-point gain being correctly attributed
        to reward tiles.
        
        Scenario details:
        - 12 active units on the map
        - 6 points gained in a single turn
        - Multiple units occupying reward tiles simultaneously
        - Proper attribution of points to correct tiles
        - No incorrect reward tile identifications
        """
        scenario = self.load_scenario("test_scenario_123.json")
        
        # Set up agent state from scenario
        self.agent.possible_reward_tiles = set(map(tuple, scenario["state_before"]["possible_reward_tiles"]))
        self.agent.unknown_tiles = set(map(tuple, scenario["state_before"]["unknown_tiles"]))
        self.agent.not_reward_tiles = set(map(tuple, scenario["state_before"]["not_reward_tiles"]))
        self.agent.known_reward_tiles = set(map(tuple, scenario["state_before"]["known_reward_tiles"]))
        self.agent.last_reward_occupied = set(map(tuple, scenario["state_before"]["last_reward_occupied"]))
        self.agent.last_unknown_occupied = set(map(tuple, scenario["state_before"]["last_unknown_occupied"]))
        self.agent.last_team_points = scenario["state_before"]["last_team_points"]
        self.agent.last_gain = scenario["state_before"]["last_gain"]
        self.agent.last_unit_positions = list(map(tuple, scenario["state_before"]["last_unit_positions"]))
        
        # Run deduction
        self.agent.deduce_reward_tiles(scenario["obs_before"])
        
        # Verify state matches scenario
        self.assertEqual(
            set(map(tuple, scenario["state_after"]["not_reward_tiles"])),
            self.agent.not_reward_tiles,
            "Non-reward tiles don't match expected state"
        )
        self.assertEqual(
            set(map(tuple, scenario["state_after"]["unknown_tiles"])),
            self.agent.unknown_tiles,
            "Unknown tiles don't match expected state"
        )
        self.assertEqual(
            set(map(tuple, scenario["state_after"]["known_reward_tiles"])),
            self.agent.known_reward_tiles,
            "Known reward tiles don't match expected state"
        )
        
    def test_scenario_large_point_gain_with_many_units(self):
        """Test scenario with a large point gain and many active units.
        
        This test verifies that the agent correctly handles scenarios with many
        units (16) and large point gains (11 points). It checks that the agent
        can properly track multiple reward tiles and unit positions while
        maintaining accurate state transitions.
        
        Scenario details:
        - 16 active units (maximum unit count)
        - 11 points gained in a single turn
        - Complex mix of known rewards and unknown tiles
        - Multiple units visiting reward tiles simultaneously
        - Proper handling of large state transitions
        """
        scenario = self.load_scenario("test_scenario_456.json")
        
        # Set up agent state from scenario
        self.agent.possible_reward_tiles = set(map(tuple, scenario["state_before"]["possible_reward_tiles"]))
        self.agent.unknown_tiles = set(map(tuple, scenario["state_before"]["unknown_tiles"]))
        self.agent.not_reward_tiles = set(map(tuple, scenario["state_before"]["not_reward_tiles"]))
        self.agent.known_reward_tiles = set(map(tuple, scenario["state_before"]["known_reward_tiles"]))
        self.agent.last_reward_occupied = set(map(tuple, scenario["state_before"]["last_reward_occupied"]))
        self.agent.last_unknown_occupied = set(map(tuple, scenario["state_before"]["last_unknown_occupied"]))
        self.agent.last_team_points = scenario["state_before"]["last_team_points"]
        self.agent.last_gain = scenario["state_before"]["last_gain"]
        self.agent.last_unit_positions = list(map(tuple, scenario["state_before"]["last_unit_positions"]))
        
        # Run deduction
        self.agent.deduce_reward_tiles(scenario["obs_before"])
        
        # Verify state matches scenario
        self.assertEqual(
            set(map(tuple, scenario["state_after"]["not_reward_tiles"])),
            self.agent.not_reward_tiles,
            "Non-reward tiles don't match expected state"
        )
        self.assertEqual(
            set(map(tuple, scenario["state_after"]["unknown_tiles"])),
            self.agent.unknown_tiles,
            "Unknown tiles don't match expected state"
        )
        self.assertEqual(
            set(map(tuple, scenario["state_after"]["known_reward_tiles"])),
            self.agent.known_reward_tiles,
            "Known reward tiles don't match expected state"
        )

if __name__ == '__main__':
    unittest.main()
