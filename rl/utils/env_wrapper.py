"""Environment wrapper for Lux AI Season 3 reinforcement learning.

This module provides a wrapper around the LuxAIS3GymEnv that implements:
1. Flattened and normalized observation space
2. Simplified action space mapping
3. Reward shaping for RL training
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from luxai_s3.state import EnvState, EnvObs
from luxai_s3.wrappers import LuxAIS3GymEnv


class LuxRLWrapper(gym.Wrapper):
    """Wrapper around LuxAIS3GymEnv for reinforcement learning."""
    
    def __init__(self, env: LuxAIS3GymEnv):
        """Initialize the wrapper with a LuxAIS3GymEnv instance.
        
        Args:
            env: The LuxAIS3GymEnv instance to wrap
        """
        super().__init__(env)
        
        # Cache map dimensions for faster access
        self.map_size = 24  # Fixed size for Lux AI S3
        self.max_units = 16  # Maximum units per player (from environment)
        
        # Create observation and action spaces
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        
        # Initialize episode tracking
        self.current_step = 0
        self.max_steps = 100  # Fixed for Lux AI S3
        
    def _create_observation_space(self):
        """Create a flattened and normalized observation space.
        
        The observation space includes:
        - Map features (24x24x4): Terrain, resources, units, etc.
        - Unit states (4x6): Position, energy, etc. for each unit
        - Global state (4): Game step, team points
        
        Returns:
            gym.spaces.Dict: The observation space
        """
        return spaces.Dict({
            # Map features: terrain, resources, units, vision
            'map_features': spaces.Box(
                low=0,
                high=1,
                shape=(self.map_size, self.map_size, 4),
                dtype=np.float32
            ),
            # Unit states: pos_x, pos_y, energy, action_queue_length
            'unit_states': spaces.Box(
                low=0,
                high=1,
                shape=(self.max_units, 6),
                dtype=np.float32
            ),
            # Global state: step, points
            'global_state': spaces.Box(
                low=0,
                high=1,
                shape=(4,),
                dtype=np.float32
            )
        })
        
    def _create_action_space(self):
        """Create an action space matching the environment's expectations.
        
        The action space includes for each unit:
        - Action type (0-6): NONE=0, MOVE_{UP,RIGHT,DOWN,LEFT}=1-4, SAP=5
        - Target position (2): x, y coordinates for sap, range [-unit_sap_range, unit_sap_range]
        
        Returns:
            gym.spaces.Dict: The action space
        """
        # Match environment's action space
        low = np.zeros((16, 3), dtype=np.int16)  # Using env's max_units=16
        low[:, 1:] = -5  # -unit_sap_range
        high = np.ones((16, 3), dtype=np.int16) * 6  # 6 action types
        high[:, 1:] = 5  # unit_sap_range
        
        return spaces.Dict({
            'unit_actions': spaces.Box(
                low=low,
                high=high,
                dtype=np.int16
            )
        })
        
    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation.
        
        Args:
            seed: Random seed for environment
            options: Additional options for reset
            
        Returns:
            dict: Processed observation
        """
        obs, info = super().reset(seed=seed, options=options)
        self.current_step = 0
        # Only return the processed observation, not the info dict
        return self._process_observation(obs)
        
    def step(self, action):
        """Take a step in the environment using the given action.
        
        Args:
            action: Dict containing unit actions
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Convert normalized actions to environment format
        env_action = self._process_action(action)
        
        # Take step in environment
        obs, reward, terminated, truncated, info = super().step(env_action)
        self.current_step += 1
        
        # Process observation and shape reward
        processed_obs = self._process_observation(obs)
        shaped_reward = self._shape_reward(reward, info)
        
        return processed_obs, shaped_reward, terminated, truncated, info
        
    def _process_observation(self, obs):
        """Process and normalize the raw observation from the environment.
        
        Args:
            obs: Raw observation from environment (dict or EnvObs)
            
        Returns:
            dict: Processed observation
        """
        # Debug: Print observation structure
        if isinstance(obs, dict):
            print("Observation keys:", obs.keys())
            for key in obs:
                if isinstance(obs[key], dict):
                    print(f"{key} subkeys:", obs[key].keys())
                else:
                    print(f"{key} type:", type(obs[key]))
        
        # Initialize processed observation with zeros
        processed = {
            'map_features': np.zeros((self.map_size, self.map_size, 4), dtype=np.float32),
            'unit_states': np.zeros((self.max_units, 6), dtype=np.float32),
            'global_state': np.zeros(4, dtype=np.float32)
        }
        
        # Extract player_0's observation
        if isinstance(obs, dict) and 'player_0' in obs:
            player_obs = obs['player_0']
        else:
            player_obs = obs
            
        # Process EnvObs instance
        try:
            if isinstance(player_obs, EnvObs):
                # Extract features from EnvObs
                map_features = np.array(player_obs.map_features.tile_type, dtype=np.float32)
                
                # Handle unit positions - extract player 0's units and reshape
                raw_positions = player_obs.units.position
                if len(raw_positions.shape) == 3:  # Shape is (2, 16, 2)
                    unit_positions = raw_positions[0].astype(np.float32)  # Take player 0's units
                else:
                    unit_positions = raw_positions.astype(np.float32)
                    
                # Handle unit energy similarly
                raw_energy = player_obs.units.energy
                if len(raw_energy.shape) == 2:  # Shape might be (2, 16)
                    unit_energy = raw_energy[0].reshape(-1, 1).astype(np.float32)  # Take player 0's units
                else:
                    unit_energy = raw_energy.reshape(-1, 1).astype(np.float32)
                    
                # Handle units mask
                raw_mask = player_obs.units_mask
                if len(raw_mask.shape) == 2:  # Shape might be (2, 16)
                    units_mask = raw_mask[0].astype(np.float32)  # Take player 0's mask
                else:
                    units_mask = raw_mask.astype(np.float32)
                    
                team_points = player_obs.team_points.astype(np.float32)
                match_steps = float(player_obs.match_steps)
            else:
                # Fallback to zeros if structure is unexpected
                print("Warning: Unexpected observation structure")
                map_features = np.zeros((self.map_size, self.map_size))
                unit_positions = np.zeros((self.max_units, 2))
                unit_energy = np.zeros((self.max_units, 1))
                units_mask = np.zeros(self.max_units)
                team_points = np.zeros(2)
                match_steps = 0.0
        except Exception as e:
            print("Error processing observation:", e)
            print("Observation structure:", obs)
            raise
        
        # Normalize and assign features
        processed['map_features'][:, :, 0] = map_features / 2.0  # Normalize by max tile type (2)
        processed['unit_states'][:, :2] = unit_positions / self.map_size  # Normalize positions
        processed['unit_states'][:, 2] = np.squeeze(unit_energy) / 100.0  # Normalize energy and ensure 1D array
        processed['unit_states'][:, 3] = np.squeeze(units_mask)  # Add unit mask and ensure 1D array
        
        # Set global state
        processed['global_state'][0] = self.current_step / self.max_steps
        processed['global_state'][1] = team_points[0] / 100.0
        processed['global_state'][2] = team_points[1] / 100.0
        processed['global_state'][3] = match_steps / self.max_steps
        
        return processed
        
    def _process_action(self, action):
        """Convert actions to environment format.
        
        Args:
            action: Actions from policy
            
        Returns:
            dict: Actions in environment format with player keys
        """
        # Convert action to numpy array if it's a dictionary
        if isinstance(action, dict):
            action = action.get('unit_actions', np.zeros((16, 3), dtype=np.int16))
        
        # Ensure action is a numpy array with correct dtype and shape
        action = np.array(action, dtype=np.int16)
        if action.shape != (16, 3):
            # Pad or truncate to match expected shape
            padded_action = np.zeros((16, 3), dtype=np.int16)
            padded_action[:min(action.shape[0], 16), :min(action.shape[1], 3)] = action[:min(action.shape[0], 16), :min(action.shape[1], 3)]
            action = padded_action
        
        # Create empty action array for opponent
        opponent_action = np.zeros((16, 3), dtype=np.int16)
        
        # Format actions with player keys
        env_action = {
            'player_0': action,
            'player_1': opponent_action
        }
        return env_action
        
    def _shape_reward(self, reward, info):
        """Shape the reward to encourage desired behavior.
        
        Args:
            reward: Original reward from environment
            info: Additional information from step
            
        Returns:
            float: Shaped reward
        """
        shaped_reward = reward
        
        # Add reward shaping components
        # TODO: Implement reward shaping
        
        return shaped_reward
