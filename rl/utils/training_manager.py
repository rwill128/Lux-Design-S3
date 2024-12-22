"""Training manager for Lux AI Season 3 reinforcement learning.

This module implements a training manager that handles:
1. Best-of-5 match series training
2. Opponent pool management
3. Training metrics tracking
4. Checkpoint management
"""

import numpy as np
import torch
from typing import List, Dict, Any
import json
import os
from datetime import datetime
from collections import deque

class SeriesTrainingManager:
    """Manager for training RL agents in best-of-5 match series."""
    
    def __init__(
        self,
        env,
        agent,
        opponent_pool: List[str],
        save_dir: str = "checkpoints",
        log_dir: str = "logs",
        series_length: int = 5,
        max_steps: int = 100
    ):
        """Initialize the training manager.
        
        Args:
            env: LuxAIS3GymEnv instance
            agent: RL agent instance
            opponent_pool: List of opponent agent names
            save_dir: Directory for saving checkpoints
            log_dir: Directory for saving logs
            series_length: Number of matches per series (default 5)
            max_steps: Maximum steps per match (default 100)
        """
        self.env = env
        self.agent = agent
        self.opponent_pool = opponent_pool
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.series_length = series_length
        self.max_steps = max_steps
        
        # Create directories if they don't exist
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics = {
            'series_wins': 0,
            'match_wins': 0,
            'total_reward': 0,
            'episodes': 0,
            'steps': 0
        }
        
        # Track recent performance
        self.recent_rewards = deque(maxlen=100)
        self.recent_wins = deque(maxlen=100)
        
    def run_training_series(
        self,
        num_series: int = 1000,
        eval_interval: int = 10,
        save_interval: int = 100,
        curriculum_interval: int = 200
    ):
        """Run multiple training series.
        
        Args:
            num_series: Number of series to run
            eval_interval: Interval for evaluation
            save_interval: Interval for saving checkpoints
            curriculum_interval: Interval for updating opponent pool
        """
        for series_idx in range(num_series):
            # Select opponent
            opponent = np.random.choice(self.opponent_pool)
            
            # Run single series
            series_metrics = self._run_single_series(opponent)
            
            # Update metrics
            self._update_metrics(series_metrics)
            
            # Log progress
            if series_idx % eval_interval == 0:
                self._log_progress(series_idx)
            
            # Save checkpoint
            if series_idx % save_interval == 0:
                self._save_checkpoint(series_idx)
            
            # Update curriculum
            if series_idx % curriculum_interval == 0:
                self._update_curriculum()
                
    def _run_single_series(self, opponent: str) -> Dict[str, Any]:
        """Run a single best-of-5 series against an opponent.
        
        Args:
            opponent: Name of the opponent agent
            
        Returns:
            dict: Series metrics
        """
        series_metrics = {
            'matches_won': 0,
            'total_reward': 0,
            'total_steps': 0
        }
        
        matches_needed = (self.series_length // 2) + 1
        matches_played = 0
        
        while matches_played < self.series_length:
            # Run single match
            match_metrics = self._run_single_match(opponent)
            matches_played += 1
            
            # Update series metrics
            series_metrics['matches_won'] += int(match_metrics['won'])
            series_metrics['total_reward'] += match_metrics['reward']
            series_metrics['total_steps'] += match_metrics['steps']
            
            # Check if series is decided
            if series_metrics['matches_won'] >= matches_needed:
                break
            if (matches_played - series_metrics['matches_won']) >= matches_needed:
                break
                
        return series_metrics
        
    def _run_single_match(self, opponent: str) -> Dict[str, Any]:
        """Run a single match against an opponent.
        
        Args:
            opponent: Name of the opponent agent
            
        Returns:
            dict: Match metrics
        """
        obs = self.env.reset()
        done_value = False
        truncated_value = False
        total_reward = 0
        steps = 0
        
        # Lists to store experiences
        observations = []
        next_observations = []
        actions = []
        rewards = []
        dones = []
        
        # Convert JAX arrays to numpy arrays if needed
        def convert_to_numpy(x):
            try:
                if hasattr(x, 'numpy'):  # JAX array
                    arr = x.numpy()
                    # Handle scalar JAX arrays
                    if not arr.shape:  # scalar array
                        return np.array([float(arr)], dtype=np.float32)
                    return arr.astype(np.float32)
                if isinstance(x, (bool, int, float)):
                    return np.array([float(x)], dtype=np.float32)
                if isinstance(x, np.ndarray):
                    return x.astype(np.float32)
                return x
            except Exception as e:
                print(f"[training_manager] Error converting to numpy: {e}")
                print(f"[training_manager] Input type: {type(x)}")
                print(f"[training_manager] Input value: {x}")
                raise
        
        # Collect experience batch
        batch_size = 8  # Collect 8 steps before training
        while not (done_value or truncated_value) and steps < self.max_steps:
            # Convert current observation to numpy
            obs_numpy = {k: convert_to_numpy(v) for k, v in obs.items()}
            
            # Get action from agent
            action, _ = self.agent.predict(obs)
            action_numpy = convert_to_numpy(action)
            
            # Take step in environment
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # Debug environment step outputs
            print(f"[training_manager] reward type: {type(reward)}")
            print(f"[training_manager] reward value: {reward}")
            print(f"[training_manager] done type: {type(done)}")
            print(f"[training_manager] done value: {done}")
            
            # Extract scalar values
            if isinstance(reward, dict):
                reward_value = reward.get('player_0', 0.0)
            else:
                reward_value = float(reward)
                
            if isinstance(done, dict):
                done_value = done.get('player_0', False)
            else:
                done_value = bool(done)
                
            if isinstance(truncated, dict):
                truncated_value = truncated.get('player_0', False)
            else:
                truncated_value = bool(truncated)
            
            # Convert next observation to numpy
            next_obs_numpy = {k: convert_to_numpy(v) for k, v in next_obs.items()}
            
            # Store experiences
            observations.append(obs_numpy)
            next_observations.append(next_obs_numpy)
            actions.append(action_numpy)
            rewards.append(reward_value)
            dones.append(done_value or truncated_value)
            
            # Update for next step
            obs = next_obs
            total_reward += reward_value
            steps += 1
            
            # Train if we have enough experiences
            if len(observations) >= batch_size or done_value or truncated_value:
                # Convert lists to numpy arrays with proper shapes
                rewards_array = np.array(rewards, dtype=np.float32).reshape(-1, 1)
                dones_array = np.array(dones, dtype=np.float32).reshape(-1, 1)
                actions_array = np.stack(actions)
                
                # Stack observations into proper format
                stacked_obs = {}
                for key in observations[0].keys():
                    # Stack arrays for each observation key
                    stacked_obs[key] = np.stack([obs[key] for obs in observations])
                
                stacked_next_obs = {}
                for key in next_observations[0].keys():
                    # Stack arrays for each next observation key
                    stacked_next_obs[key] = np.stack([obs[key] for obs in next_observations])
                
                print(f"[training_manager] Training with batch size: {len(observations)}")
                print(f"[training_manager] rewards shape: {rewards_array.shape}")
                print(f"[training_manager] dones shape: {dones_array.shape}")
                print(f"[training_manager] actions shape: {actions_array.shape}")
                print(f"[training_manager] observations shape: {[(k, v.shape) for k, v in stacked_obs.items()]}")
                
                # Train on collected experiences
                self.agent.train_step(
                    stacked_obs,
                    actions_array,
                    rewards_array,
                    dones_array,
                    stacked_next_obs
                )
                
                # Clear experience buffers
                observations = []
                next_observations = []
                actions = []
                rewards = []
                dones = []
            
            # Debug step info
            print(f"[training_manager] total_reward: {total_reward}")
            
        return {
            'won': info.get('winner', 0) == 1,  # Assuming player 1 is our agent
            'reward': total_reward,
            'steps': steps
        }
        
    def _update_metrics(self, series_metrics: Dict[str, Any]):
        """Update tracking metrics with results from a series.
        
        Args:
            series_metrics: Metrics from the series
        """
        self.metrics['series_wins'] += int(series_metrics['matches_won'] > self.series_length // 2)
        self.metrics['match_wins'] += series_metrics['matches_won']
        self.metrics['total_reward'] += series_metrics['total_reward']
        self.metrics['episodes'] += self.series_length
        self.metrics['steps'] += series_metrics['total_steps']
        
        # Update recent performance tracking
        self.recent_rewards.append(series_metrics['total_reward'])
        self.recent_wins.append(series_metrics['matches_won'])
        
    def _log_progress(self, series_idx: int):
        """Log training progress.
        
        Args:
            series_idx: Current series index
        """
        metrics = {
            'series': series_idx,
            'avg_reward': np.mean(self.recent_rewards),
            'avg_wins': np.mean(self.recent_wins),
            'total_series_wins': self.metrics['series_wins'],
            'total_match_wins': self.metrics['match_wins'],
            'total_episodes': self.metrics['episodes'],
            'total_steps': self.metrics['steps']
        }
        
        # Save metrics to log file
        log_path = os.path.join(
            self.log_dir,
            f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(log_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    def _save_checkpoint(self, series_idx: int):
        """Save a checkpoint of the agent.
        
        Args:
            series_idx: Current series index
        """
        checkpoint = {
            'series': series_idx,
            'model_state': self.agent.policy.state_dict(),
            'optimizer_state': self.agent.optimizer.state_dict(),
            'metrics': self.metrics
        }
        
        checkpoint_path = os.path.join(
            self.save_dir,
            f'checkpoint_{series_idx}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
    def _update_curriculum(self):
        """Update the training curriculum.
        
        This method can be extended to implement curriculum learning by:
        1. Adjusting opponent pool based on agent performance
        2. Modifying environment parameters
        3. Changing reward scaling
        """
        # Calculate recent performance
        win_rate = np.mean(self.recent_wins) / self.series_length
        
        # Example curriculum adjustment (can be extended)
        if win_rate > 0.7:
            # Agent is doing well, could increase difficulty
            pass
        elif win_rate < 0.3:
            # Agent is struggling, could decrease difficulty
            pass
