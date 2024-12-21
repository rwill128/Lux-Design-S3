import os
from collections import defaultdict

from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

import numpy as np
from scipy.optimize import linear_sum_assignment

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from submissions.best_agent_attacker_no_nebula_different_explore_no_attack import \
    BestAgentAttackerNoNebulaDifferentExploreNoAttack
from submissions.best_agent_better_shooter import BestAgentBetterShooter

def evaluate_agents(agent_1_cls, agent_2_cls, seed=45, games_to_play=3, replay_save_dir="replays"):
    # Ensure the replay directory exists
    os.makedirs(replay_save_dir, exist_ok=True)

    # Create an environment wrapped to record episodes
    gym_env = LuxAIS3GymEnv(numpy_output=True, max_episode_length=100)  # Limit episode length
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
            if dones["player_0"] or dones["player_1"]:
                game_done = True
            step += 1

    env.close()  # saves the replay of the last game and frees resources
    print(f"Finished {games_to_play} games. Replays saved to {replay_save_dir}")


if __name__ == "__main__":
    # Run evaluation with the dummy Agent against itself
    # Import agents
    from submissions.best_agent_attacker import BestAgentAttacker
    from submissions.best_agent_better_shooter import BestAgentBetterShooter
    
    # Run three games with different seeds to capture diverse scenarios
    # Game 1: Baseline scenario
    evaluate_agents(BestAgentAttacker, BestAgentBetterShooter, seed=42, games_to_play=1,
                   replay_save_dir="replays/game_42")
    
    # Game 2: Different seed for variety
    evaluate_agents(BestAgentAttacker, BestAgentBetterShooter, seed=123, games_to_play=1,
                   replay_save_dir="replays/game_123")
    
    # Game 3: Another seed for more scenarios
    evaluate_agents(BestAgentAttacker, BestAgentBetterShooter, seed=456, games_to_play=1,
                   replay_save_dir="replays/game_456")

    # After running, you can check the "replays" directory for saved replay files.
    # You can set breakpoints anywhere in this file or inside the Agent class.
