import numpy as np
import os
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

from agents.baseline_agent.baselineagent import BaselineAgent
from lux.utils import direction_to


class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (max_units,)
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape (max_units, 2)
        sensor_mask = obs["sensor_mask"]  # shape (W, H)
        energy_map = obs["map_features"]["energy"]  # shape (W, H)

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # Identify available units
        available_unit_ids = np.where(unit_mask)[0]

        # Apply the sensor mask to the energy map
        visible_energy = np.where(sensor_mask, energy_map, -9999)

        # Find all visible tiles sorted by energy in descending order
        # Flatten the array, get indices, then unflatten
        flat_energy = visible_energy.flatten()
        # Get indices of sorted tiles by energy descending
        sorted_indices = np.argsort(flat_energy)[::-1]

        # Filter out tiles that have negative (invisible) energy
        sorted_indices = sorted_indices[flat_energy[sorted_indices] > -1]

        # If no visible tile, do nothing
        if len(sorted_indices) == 0:
            return actions

        # We'll assign each unit a tile in descending order of energy.
        # If more units than tiles, some units will do nothing at the end.
        # If more tiles than units, we just ignore extra tiles.
        num_units = len(available_unit_ids)
        num_tiles = len(sorted_indices)
        assign_count = min(num_units, num_tiles)

        # Assign top assign_count tiles to units
        for i in range(assign_count):
            unit_id = available_unit_ids[i]
            tile_idx = sorted_indices[i]
            w = energy_map.shape[1]  # width
            h = energy_map.shape[0]  # height

            # Convert flat index back to (y, x)
            tile_y = tile_idx // w
            tile_x = tile_idx % w

            unit_x, unit_y = unit_positions[unit_id]
            # If unit is already on the tile, do nothing
            if unit_x == tile_x and unit_y == tile_y:
                actions[unit_id] = [0, 0, 0]
            else:
                # Move unit towards assigned tile
                direction = direction_to(np.array([unit_x, unit_y]), np.array([tile_x, tile_y]))
                actions[unit_id] = [direction, 0, 0]

        # Remaining units (if any) do nothing
        for i in range(assign_count, num_units):
            unit_id = available_unit_ids[i]
            actions[unit_id] = [0, 0, 0]

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
