import numpy as np
import os
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

from agents.baseline_agent.baselineagent import BaselineAgent

class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

    def count_visible_tiles(obs, team_id):
        # sensor_mask is shape (W, H). True means visible.
        sensor_mask = obs["sensor_mask"]
        # Count how many tiles are visible to the given team.
        # Note: sensor_mask is currently 2D, but per the docs it's for each team.
        # In some versions, sensor_mask might be shaped differently or per-team.
        # Adjust indexing accordingly if needed.
        visible_count = np.sum(sensor_mask)
        return visible_count

    def total_team_energy(obs, team_id):
        # obs["units"]["energy"] is shape (T, N) or (T, N, 1). Check actual shape.
        # In your snippet it's shape (T, N) after extracting.
        team_energy = obs["units"]["energy"][team_id]
        # Ignore -1 (means no unit)
        valid_energy = team_energy[team_energy != -1]
        return np.sum(valid_energy)

    def points_delta(obs, prev_obs, team_id):
        if prev_obs is None:
            return 0
        return obs["team_points"][team_id] - prev_obs["team_points"][team_id]

    def find_high_value_tiles(obs, threshold=5):
        # obs["map_features"]["energy"] is shape (W, H)
        energy_map = obs["map_features"]["energy"]
        # Find coordinates where energy > threshold
        coords = np.argwhere(energy_map > threshold)
        return coords

    def unit_positions(obs, team_id):
        # obs["units"]["position"] is shape (T, N, 2)
        positions = obs["units"]["position"][team_id]
        # Filter out invalid (-1, -1) positions
        valid_pos = positions[~(positions == -1).any(axis=1)]
        return valid_pos

    def average_distance_to_high_value_tiles(units_pos, high_value_tiles):
        if len(high_value_tiles) == 0 or len(units_pos) == 0:
            return None
        # Compute pairwise distances and return average of nearest distances
        avg_dist = 0
        for pos in units_pos:
            distances = np.sum((high_value_tiles - pos)**2, axis=1)**0.5
            avg_dist += np.min(distances)
        avg_dist /= len(units_pos)
        return avg_dist

    def act(self, step: int, obs, remainingOverageTime: int = 60):
            """
            Implement your logic here. This is just a dummy agent that does nothing.
            """
            unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (max_units, )
            actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
            # Insert your decision logic here if desired
            return actions

def evaluate_agents(agent_1_cls, agent_2_cls, seed=42, games_to_play=3, replay_save_dir="replays"):
    # Ensure the replay directory exists
    os.makedirs(replay_save_dir, exist_ok=True)

    # Create an environment wrapped to record episodes
    env = RecordEpisode(
        LuxAIS3GymEnv(numpy_output=True), save_on_close=True, save_on_reset=True, save_dir=replay_save_dir
    )

    for i in range(games_to_play):
        # Reset the environment for each game
        obs, info = env.reset(seed=seed+i)  # changing seed each game
        env_cfg = info["params"]  # game parameters that agents can see

        player_0 = agent_1_cls("player_0", env_cfg)
        player_1 = agent_2_cls("player_1", env_cfg)

        game_done = False
        step = 0
        print(f"Running game {i+1}/{games_to_play}")
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
    evaluate_agents(BaselineAgent, Agent, games_to_play=3, replay_save_dir="replays/" + BaselineAgent.__name__ + "_" + Agent.__name__)

    # After running, you can check the "replays" directory for saved replay files.
    # You can set breakpoints anywhere in this file or inside the Agent class.
