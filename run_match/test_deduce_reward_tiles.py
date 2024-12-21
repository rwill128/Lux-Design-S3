import unittest
import numpy as np
from run_match import BestAgentAttacker


class TestDeduceRewardTiles(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.agent = BestAgentAttacker("player_0", {})

    def test_confidence_persistence(self):
        """Test that confidence values persist between agent instances"""
        # First agent instance
        agent1 = BestAgentAttacker("player_0", {})
        
        # Simulate a tile with positive confidence
        test_tile = (5, 5)
        agent1.tile_confidence[test_tile] = 1.5
        agent1._persistent_tile_confidence[test_tile] = 1.5

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
        agent1.relic_patterns[relic_pos] = {frozenset(test_pattern)}
        agent1._persistent_relic_patterns[relic_pos] = {frozenset(test_pattern)}
        
        # Create second agent instance
        agent2 = BestAgentAttacker("player_0", {})
        
        # Check pattern persistence
        self.assertIn(relic_pos, agent2.relic_patterns)
        self.assertIn(test_pattern, agent2.relic_patterns[relic_pos])

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


if __name__ == '__main__':
    unittest.main()
