"""PPO agent implementation for Lux AI Season 3.

This module implements a PPO (Proximal Policy Optimization) agent using PyTorch,
specifically designed for the Lux AI Season 3 environment. It includes both actor
and critic networks with customizable architectures.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class LuxPPOPolicy(nn.Module):
    """Neural network policy for PPO algorithm."""
    
    def __init__(self, observation_space: Dict, action_space: Dict):
        """Initialize the policy networks.
        
        Args:
            observation_space: Dict of observation spaces
            action_space: Dict of action spaces
        """
        super().__init__()
        
        # Calculate input dimensions
        self.map_feature_dim = int(np.prod(observation_space['map_features'].shape))
        self.unit_state_dim = int(np.prod(observation_space['unit_states'].shape))
        self.global_state_dim = int(np.prod(observation_space['global_state'].shape))
        
        # Calculate total input dimension
        self.input_dim = self.map_feature_dim + self.unit_state_dim + self.global_state_dim
        
        # Calculate output dimension for actor
        self.action_dim = int(np.prod(action_space['unit_actions'].shape))
        
        # Feature extraction layers
        self.map_features = nn.Sequential(
            nn.Linear(self.map_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.unit_features = nn.Sequential(
            nn.Linear(self.unit_state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.global_features = nn.Sequential(
            nn.Linear(self.global_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(224, 256),  # 128 + 64 + 32 = 224
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(224, 256),  # 128 + 64 + 32 = 224
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def _extract_features(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from different observation components.
        
        Args:
            obs: Dictionary containing different observation components
            
        Returns:
            torch.Tensor: Concatenated feature vector
        """
        # Flatten and process map features
        map_features = obs['map_features'].reshape(-1, self.map_feature_dim)
        map_features = self.map_features(map_features)
        
        # Process unit states
        unit_states = obs['unit_states'].reshape(-1, self.unit_state_dim)
        unit_features = self.unit_features(unit_states)
        
        # Process global state
        global_state = obs['global_state'].reshape(-1, self.global_state_dim)
        global_features = self.global_features(global_state)
        
        # Concatenate all features
        return torch.cat([map_features, unit_features, global_features], dim=1)
        
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both actor and critic networks.
        
        Args:
            obs: Dictionary containing different observation components
            
        Returns:
            tuple: (action_logits, value_estimate)
        """
        features = self._extract_features(obs)
        return self.actor(features), self.critic(features)
        
    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for training.
        
        Args:
            obs: Dictionary containing different observation components
            actions: Actions to evaluate
            
        Returns:
            tuple: (action_log_probs, values, entropy)
        """
        action_logits, values = self.forward(obs)
        
        # Create action distribution
        action_dist = torch.distributions.Normal(action_logits, 1.0)
        
        
        # Compute log probabilities, entropy
        log_probs = action_dist.log_prob(actions).sum(dim=-1)
        entropy = action_dist.entropy().mean()
        
        return log_probs, values.squeeze(), entropy

class LuxPPOAgent:
    """PPO agent for Lux AI Season 3."""
    
    def __init__(
        self,
        observation_space: Dict,
        action_space: Dict,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the PPO agent.
        
        Args:
            observation_space: Dict of observation spaces
            action_space: Dict of action spaces
            learning_rate: Learning rate for optimizer
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size for training
            n_epochs: Number of epoch when optimizing the surrogate loss
            gamma: Discount factor
            gae_lambda: Factor for trade-off of bias vs variance for GAE
            clip_range: Clipping parameter for PPO
            device: Device to run the model on
        """
        self.device = torch.device(device)
        self.policy = LuxPPOPolicy(observation_space, action_space).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # PPO parameters
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        
    def predict(
        self,
        obs: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Predict action based on observation.
        
        Args:
            obs: Dictionary containing different observation components
            deterministic: Whether to use deterministic action selection
            
        Returns:
            tuple: (actions, None)
        """
        # Convert JAX arrays to numpy if needed
        def to_numpy(x):
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
                print(f"[predict] Error converting to numpy: {e}")
                print(f"[predict] Input type: {type(x)}")
                print(f"[predict] Input value: {x}")
                raise

        # Convert inputs to numpy arrays first
        obs_numpy = {k: to_numpy(v) for k, v in obs.items()}

        # Convert numpy arrays to tensors
        obs_tensor = {
            k: torch.FloatTensor(v).to(self.device)
            for k, v in obs_numpy.items()
        }
        
        with torch.no_grad():
            action_logits, _ = self.policy(obs_tensor)
            
            if deterministic:
                actions = action_logits
            else:
                # Add noise for exploration
                action_dist = torch.distributions.Normal(action_logits, 1.0)
                actions = action_dist.sample()
            
        return actions.cpu().numpy(), None
        
    def train_step(
        self,
        obs: Dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_obs: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Perform a training step using PPO algorithm.
        
        Args:
            obs: Dictionary of observations
            actions: Actions taken
            rewards: Rewards received
            dones: Done flags
            next_obs: Next observations
            
        Returns:
            dict: Training metrics
        """
        # Convert JAX arrays to numpy if needed
        def to_numpy(x):
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
                print(f"[train_step] Error converting to numpy: {e}")
                print(f"[train_step] Input type: {type(x)}")
                print(f"[train_step] Input value: {x}")
                raise

        # Debug info for inputs
        print(f"[train_step] dones type: {type(dones)}")
        if hasattr(dones, 'numpy'):
            print(f"[train_step] dones JAX shape: {dones.shape if hasattr(dones, 'shape') else 'no shape'}")

        # Convert inputs to numpy arrays first
        obs_numpy = {k: to_numpy(v) for k, v in obs.items()}
        actions_numpy = to_numpy(actions)
        rewards_numpy = to_numpy(rewards)
        dones_numpy = to_numpy(dones)

        # Debug info for numpy arrays
        print(f"[train_step] dones_numpy type: {type(dones_numpy)}")
        print(f"[train_step] dones_numpy shape: {dones_numpy.shape if hasattr(dones_numpy, 'shape') else 'no shape'}")
        print(f"[train_step] dones_numpy value: {dones_numpy}")

        # Ensure dones is properly shaped for tensor conversion
        if isinstance(dones_numpy, np.ndarray) and not dones_numpy.shape:
            dones_numpy = np.array([float(dones_numpy)], dtype=np.float32)
        elif not isinstance(dones_numpy, np.ndarray):
            dones_numpy = np.array([float(dones_numpy)], dtype=np.float32)

        # Convert numpy arrays to tensors
        obs_tensor = {
            k: torch.FloatTensor(v).to(self.device)
            for k, v in obs_numpy.items()
        }
        actions_tensor = torch.FloatTensor(actions_numpy).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards_numpy).to(self.device)
        dones_tensor = torch.FloatTensor(dones_numpy).to(self.device)

        # Debug info for tensors
        print(f"[train_step] dones_tensor shape: {dones_tensor.shape}")
        print(f"[train_step] dones_tensor value: {dones_tensor}")
        
        # Compute advantages using GAE
        with torch.no_grad():
            _, values = self.policy(obs_tensor)
            # Convert next_obs to numpy first
            next_obs_numpy = {k: to_numpy(v) for k, v in next_obs.items()}
            _, next_values = self.policy({
                k: torch.FloatTensor(v).to(self.device)
                for k, v in next_obs_numpy.items()
            })
            
            # Debug shapes before GAE
            print(f"[train_step] rewards_tensor shape: {rewards_tensor.shape}")
            print(f"[train_step] values shape before GAE: {values.shape}")
            print(f"[train_step] next_values shape before GAE: {next_values.shape}")
            print(f"[train_step] dones_tensor shape before GAE: {dones_tensor.shape}")
            
            # Ensure tensors have proper batch dimension
            if len(rewards_tensor.shape) == 1:
                rewards_tensor = rewards_tensor.unsqueeze(1)
            if len(values.shape) == 1:
                values = values.unsqueeze(1)
            if len(next_values.shape) == 1:
                next_values = next_values.unsqueeze(1)
            if len(dones_tensor.shape) == 1:
                dones_tensor = dones_tensor.unsqueeze(1)
            
            advantages = self._compute_gae(
                rewards_tensor,
                values,
                next_values,
                dones_tensor
            )
            
            # Debug shapes after GAE
            print(f"[train_step] advantages shape: {advantages.shape}")
            print(f"[train_step] values shape after GAE: {values.shape}")
            
            returns = advantages + values
            print(f"[train_step] returns shape: {returns.shape}")
            
        # PPO training loop
        for _ in range(self.n_epochs):
            # Get action log probs and value estimates
            log_probs, values, entropy = self.policy.evaluate_actions(
                obs_tensor,
                actions_tensor
            )
            
            # Compute PPO loss
            ratio = torch.exp(log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
            
            # Ensure tensors have matching shapes for loss calculation
            if values.shape != returns.shape:
                values = values.view(returns.shape)
            
            # Debug shapes before loss calculation
            print(f"[train_step] values shape before loss: {values.shape}")
            print(f"[train_step] returns shape before loss: {returns.shape}")
            
            # Calculate losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(returns, values)
            entropy_loss = -entropy.mean()
            
            # Compute total loss
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": loss.item()
        }
        
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Rewards tensor
            values: Values tensor
            next_values: Next values tensor
            dones: Done flags tensor
            
        Returns:
            torch.Tensor: Advantages
        """
        # Debug input shapes
        print(f"[compute_gae] rewards shape: {rewards.shape}")
        print(f"[compute_gae] values shape: {values.shape}")
        print(f"[compute_gae] next_values shape: {next_values.shape}")
        print(f"[compute_gae] dones shape: {dones.shape}")
        
        # Handle empty tensors
        if rewards.shape[0] == 0:
            print("[compute_gae] Warning: Empty tensors received")
            return torch.zeros((1, 1), device=self.device)
            
        # Ensure all tensors have same batch size
        batch_size = rewards.shape[0]
        if values.shape[0] != batch_size or next_values.shape[0] != batch_size or dones.shape[0] != batch_size:
            raise ValueError(f"Tensor batch sizes don't match: rewards={rewards.shape}, values={values.shape}, next_values={next_values.shape}, dones={dones.shape}")
            
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            
        print(f"[compute_gae] advantages shape: {advantages.shape}")
        return advantages
