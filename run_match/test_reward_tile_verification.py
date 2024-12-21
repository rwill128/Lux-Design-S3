import json
import logging
import numpy as np
import flax
from luxai_s3.wrappers import RecordEpisode, LuxAIS3GymEnv
from luxai_s3.state import EnvState
from submissions.best_agent_attacker import BestAgentAttacker, NumpyEncoder
from submissions.best_agent_better_shooter import BestAgentBetterShooter

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='test_reward_tile_verification.log'
)

class TestRewardTileVerification:
    def setup_method(self):
        # Initialize base environment first to get config
        base_env = LuxAIS3GymEnv(numpy_output=True)
        env_params = base_env.env_params
        
        # Convert env_params to dict for agent initialization
        env_cfg = {
            'max_units': env_params.max_units,
            'map_width': env_params.map_width,
            'map_height': env_params.map_height,
            'unit_move_cost': env_params.unit_move_cost,
            'unit_sap_cost': env_params.unit_sap_cost,
            'unit_sap_range': env_params.unit_sap_range,
            'unit_sensor_range': env_params.unit_sensor_range,
        }
        
        # Log environment configuration
        logging.debug("Environment params: %s", 
                     json.dumps(env_cfg, cls=NumpyEncoder, indent=2))
        
        # Wrap environment with recording
        self.env = RecordEpisode(
            base_env,
            save_dir="test_data/reward_verification"
        )
        
        # Initialize agents with proper string player IDs
        self.attacker = BestAgentAttacker(player="player_0", env_cfg=env_cfg)
        self.shooter = BestAgentBetterShooter(player="player_1", env_cfg=env_cfg)
        
        # Store env_params for later use
        self.env_params = env_params
        
        logging.debug("Agents initialized successfully")
        
    def get_actual_reward_tiles(self, state):
        """Extract actual reward tiles from environment state.
        
        Args:
            state: EnvState object containing environment state information
            
        Returns:
            set: Set of (x, y) tuples representing reward tile positions
        """
        reward_tiles = set()
        
        try:
            # Convert state to serialized dictionary format for logging
            state_dict = flax.serialization.to_state_dict(state)
            logging.debug("State dict keys: %s", list(state_dict.keys()))
            
            # Extract relic nodes and their configurations using state attributes
            active_relic_nodes = state.relic_nodes[state.relic_nodes_mask]
            active_configs = state.relic_node_configs[state.relic_nodes_mask]
            
            logging.debug("Found %d active relic nodes", len(active_relic_nodes))
            
            # Process each active relic node
            for relic_pos, config in zip(active_relic_nodes, active_configs):
                rx, ry = relic_pos
                
                # Check 5x5 grid around relic for reward tiles
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        x, y = rx + dx, ry + dy
                        # Check if position is valid and is a reward tile
                        if (0 <= x < self.env_params.map_width and 
                            0 <= y < self.env_params.map_height and
                            config[dx+2][dy+2]):
                            reward_tiles.add((int(x), int(y)))
            
            logging.debug("Found reward tiles: %s", reward_tiles)
            
        except Exception as e:
            logging.error("Error extracting reward tiles: %s", str(e))
            logging.error("State attributes: %s", dir(state))
            raise
        
        return reward_tiles
        
    def test_reward_tile_deduction_accuracy(self):
        """Test that reward tile deduction produces no false positives.
        
        This test runs a full game and verifies at each step that any tiles
        we have deduced to be reward tiles are actually reward tiles according
        to the environment state.
        """
        obs, info = self.env.reset(seed=42)
        
        # Format initial observations for agents
        attacker_obs = obs["player_0"]
        shooter_obs = obs["player_1"]
        
        # Log initial state keys and structure
        logging.debug("Initial state info keys: %s", list(info.keys()))
        logging.debug("Initial observation keys: %s", list(attacker_obs.keys()))
        if "final_state" in info:
            logging.debug("Final state attributes: %s", dir(info["final_state"]))
        
        done = False
        step = 0
        
        while not done:
            # Format observations for agents
            attacker_obs = obs["player_0"]
            shooter_obs = obs["player_1"]
            
            # Get actions from both agents
            attacker_action = self.attacker.act(step, attacker_obs, 0)
            shooter_action = self.shooter.act(step, shooter_obs, 0)
            actions = {"player_0": attacker_action, "player_1": shooter_action}
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(actions)
            done = terminated or truncated
            
            # Get actual reward tiles from environment state
            actual_reward_tiles = self.get_actual_reward_tiles(info["final_state"])
            
            # Get deduced reward tiles from attacker agent
            deduced_reward_tiles = self.attacker.known_reward_tiles
            
            # Log state for debugging
            logging.debug("Step %d state:", step)
            logging.debug("Actual reward tiles: %s", actual_reward_tiles)
            logging.debug("Deduced reward tiles: %s", deduced_reward_tiles)
            if "team_points" in obs:
                logging.debug("Current points: %s", obs.get("team_points"))
            if "units" in obs and "position" in obs["units"]:
                logging.debug("Unit positions: %s", 
                            obs["units"]["position"].get(self.attacker.team_id, []))
            
            # Verify no false positives in deduction
            assert deduced_reward_tiles.issubset(actual_reward_tiles), \
                f"False positives in deduced reward tiles: {deduced_reward_tiles - actual_reward_tiles}"
            
            step += 1
