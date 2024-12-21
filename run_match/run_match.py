import os
import sys
import logging
import argparse
from collections import defaultdict

from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

import numpy as np
from scipy.optimize import linear_sum_assignment

from submissions.attacker import BestAgentAttacker

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from submissions.best_agent_better_shooter import BestAgentBetterShooter

def setup_logging(log_file, log_level):
    """Configure logging to both file and console."""
    # Set root logger to WARNING to filter out JAX logs
    logging.getLogger().setLevel(logging.WARNING)
    
    # Create our custom logger for deduce_reward_tiles
    logger = logging.getLogger('deduce_reward_tiles')
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.propagate = False  # Don't propagate to root logger
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def evaluate_agents(agent_1_cls, agent_2_cls, seed=45, games_to_play=3, max_steps=1000, replay_save_dir="replays", log_level="DEBUG", log_file="logs/matches/match.log"):
    # Ensure directories exist
    os.makedirs(replay_save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Setup logging
    logger = setup_logging(log_file, log_level)
    logger.info(f"Starting evaluation with seed {seed} and max_steps {max_steps}")

    # Create an environment wrapped to record episodes
    gym_env = LuxAIS3GymEnv(numpy_output=True)  # Initialize environment
    gym_env.render_mode = "human"
    env = RecordEpisode(
        gym_env, save_on_close=True, save_on_reset=True, save_dir=replay_save_dir
    )

    for i in range(games_to_play):
        # Reset the environment for each game
        obs, info = env.reset(seed=seed + i)  # changing seed each game
        env_cfg = info["params"]  # game parameters that agents can see

        player_0 = agent_1_cls("player_0", env_cfg)
        player_1 = agent_2_cls("player_1", env_cfg)

        game_done = False
        step = 0
        print(f"Running game {i + 1}/{games_to_play}")
        while not game_done:
            actions = {
                "player_0": player_0.act(step=step, obs=obs["player_0"]),
                "player_1": player_1.act(step=step, obs=obs["player_1"]),
            }

            obs, reward, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] or truncated[k] for k in terminated}
            if dones["player_0"] or dones["player_1"] or step >= max_steps:
                game_done = True
            step += 1

    env.close()  # saves the replay of the last game and frees resources
    print(f"Finished {games_to_play} games. Replays saved to {replay_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_games', type=int, default=1, help='Number of games to play')
    parser.add_argument('--max_steps', type=int, default=50, help='Maximum steps per game')
    parser.add_argument('--log_level', default='DEBUG', help='Logging level')
    parser.add_argument('--log_file', default='logs/matches/match.log', help='Log file path')
    parser.add_argument('--replay_dir', default='logs/matches', help='Directory to save replay files')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(args.replay_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(args.log_file, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting match with seed {args.seed}")
    
    # Run games
    evaluate_agents(
        BestAgentAttacker,
        BestAgentBetterShooter,
        seed=args.seed,
        games_to_play=args.num_games,
        max_steps=args.max_steps,
        replay_save_dir=args.replay_dir,
        log_level=args.log_level,
        log_file=args.log_file
    )
    
    logger.info("Match completed")
