"""Training script for Lux AI Season 3 reinforcement learning agent.

This script ties together the custom components:
1. LuxRLWrapper for environment handling
2. PPO agent for policy learning
3. SeriesTrainingManager for training coordination
"""

import os
import argparse
import torch
from typing import Dict, Any
import yaml

from luxai_s3.wrappers import LuxAIS3GymEnv
from rl.utils.env_wrapper import LuxRLWrapper
from rl.agents.ppo_agent import LuxPPOAgent
from rl.utils.training_manager import SeriesTrainingManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RL agent for Lux AI S3')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--opponent-pool',
        type=str,
        nargs='+',
        default=['best_agent_attacker'],
        help='List of opponent agents to train against'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_environment(config: Dict[str, Any]) -> LuxRLWrapper:
    """Set up the Lux environment with wrapper.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LuxRLWrapper: Wrapped environment
    """
    env = LuxAIS3GymEnv()
    return LuxRLWrapper(env)


def setup_agent(
    env: LuxRLWrapper,
    config: Dict[str, Any],
    checkpoint_path: str = None
) -> LuxPPOAgent:
    """Set up the PPO agent.
    
    Args:
        env: Wrapped environment
        config: Configuration dictionary
        checkpoint_path: Optional path to checkpoint
        
    Returns:
        LuxPPOAgent: Initialized agent
    """
    agent = LuxPPOAgent(
        env.observation_space,
        env.action_space,
        learning_rate=config['ppo']['learning_rate'],
        n_steps=config['ppo']['n_steps'],
        batch_size=config['ppo']['batch_size'],
        n_epochs=config['ppo']['n_epochs'],
        gamma=config['ppo']['gamma'],
        gae_lambda=config['ppo']['gae_lambda'],
        clip_range=config['ppo']['clip_range'],
        device=config['training']['device']
    )
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        agent.policy.load_state_dict(checkpoint['model_state'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
    return agent


def main():
    """Main training loop."""
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    
    # Set up environment and agent
    env = setup_environment(config)
    agent = setup_agent(env, config, args.checkpoint)
    
    # Create training manager
    manager = SeriesTrainingManager(
        env=env,
        agent=agent,
        opponent_pool=args.opponent_pool,
        save_dir=config['training']['save_dir'],
        log_dir=config['training']['log_dir'],
        series_length=config['training']['series_length'],
        max_steps=config['training']['max_steps']
    )
    
    # Run training
    manager.run_training_series(
        num_series=config['training']['num_series'],
        eval_interval=config['training']['eval_interval'],
        save_interval=config['training']['save_interval'],
        curriculum_interval=config['training']['curriculum_interval']
    )


if __name__ == '__main__':
    main()
