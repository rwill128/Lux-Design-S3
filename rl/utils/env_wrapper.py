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
        self.max_units = 4  # Maximum units per player
        
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
        """Create a simplified action space for each unit.
        
        The action space includes:
        - Movement direction (5): center, up, right, down, left
        - Action type (2): move, sap
        - Target position (2): x, y coordinates for sap
        
        Returns:
            gym.spaces.Dict: The action space
        """
        return spaces.Dict({
            'unit_actions': spaces.Box(
                low=0,
                high=1,
                shape=(self.max_units, 4),  # direction, action_type, target_x, target_y
                dtype=np.float32
            )
        })
        
    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation.
        
        Args:
            seed: Random seed for environment
            options: Additional options for reset
            
        Returns:
            tuple: (observation, info)
        """
        obs, info = super().reset(seed=seed, options=options)
        self.current_step = 0
        return self._process_observation(obs), info
        
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
        
        # Extract data based on observation type
        if isinstance(obs, dict):
            # Process dictionary observation
            try:
                # Try to access observation fields with error handling
                if 'board' in obs:
                    map_features = np.array(obs['board'].get('tile_type', np.zeros((self.map_size, self.map_size))), dtype=np.float32)
                else:
                    map_features = np.zeros((self.map_size, self.map_size))
                
                if 'units' in obs:
                    unit_data = obs['units']
                    unit_positions = np.array(unit_data.get('position', np.zeros((self.max_units, 2))), dtype=np.float32)
                    unit_energy = np.array(unit_data.get('energy', np.zeros((self.max_units, 1))), dtype=np.float32)
                else:
                    unit_positions = np.zeros((self.max_units, 2))
                    unit_energy = np.zeros((self.max_units, 1))
                
                units_mask = np.array(obs.get('units_mask', np.zeros(self.max_units)), dtype=np.float32)
                team_points = np.array(obs.get('team_points', [0, 0]), dtype=np.float32)
                match_steps = float(obs.get('match_steps', 0))
            except Exception as e:
                print("Error processing observation:", e)
                print("Observation structure:", obs)
                raise
        else:
            # Process EnvObs instance
            map_features = np.array(obs.map_features.tile_type, dtype=np.float32)
            unit_positions = obs.units.position.astype(np.float32)
            unit_energy = obs.units.energy.astype(np.float32)
            units_mask = obs.units_mask.astype(np.float32)
            team_points = obs.team_points.astype(np.float32)
            match_steps = float(obs.match_steps)
        
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
        """Convert normalized actions to environment format.
        
        Args:
            action: Normalized actions from policy
            
        Returns:
            dict: Actions in environment format
        """
        # TODO: Implement action processing
        return {}
        
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
