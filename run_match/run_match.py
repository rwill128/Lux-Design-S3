import numpy as np
import os
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

from agents.baseline_agent.baselineagent import BaselineAgent
from lux.utils import direction_to

# You can install scipy if not already available:
# !pip install scipy
from scipy.optimize import linear_sum_assignment

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

        # Flatten and find indices of visible tiles with energy >= 0
        flat_energy = visible_energy.flatten()
        valid_indices = np.where(flat_energy > -1)[0]

        # If no visible tile, do nothing
        if len(valid_indices) == 0:
            return actions

        # Extract tile coordinates and their values
        w = energy_map.shape[1]  # width
        tile_coords = []
        tile_values = []
        for idx in valid_indices:
            tile_y = idx // w
            tile_x = idx % w
            tile_coords.append((tile_x, tile_y))
            tile_values.append(visible_energy[tile_y, tile_x])

        # Sort tiles by energy descending
        tile_coords = [t for _, t in sorted(zip(tile_values, tile_coords), key=lambda x: x[0], reverse=True)]

        # Now we have tile_coords sorted by value descending
        # If fewer tiles than units, not all units will be assigned
        # If more tiles than units, not all tiles get a unit
        num_units = len(available_unit_ids)
        num_tiles = len(tile_coords)
        assign_count = min(num_units, num_tiles)

        # Create cost matrix for assignment:
        # Rows: units, Cols: tiles, cost = Manhattan distance (or Euclidean if you prefer)
        cost_matrix = np.zeros((assign_count, assign_count), dtype=int)

        # If we have fewer tiles than units or vice versa, we'll limit to assign_count for both
        # and ignore extra units or tiles
        selected_units = available_unit_ids[:assign_count]
        selected_tiles = tile_coords[:assign_count]

        for i, unit_id in enumerate(selected_units):
            ux, uy = unit_positions[unit_id]
            for j, (tx, ty) in enumerate(selected_tiles):
                # Manhattan distance
                dist = abs(ux - tx) + abs(uy - ty)
                cost_matrix[i, j] = dist

        # Solve the assignment problem
        # linear_sum_assignment returns the optimal row->col mapping
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # row_ind[i], col_ind[i] means selected_units[row_ind[i]] assigned to selected_tiles[col_ind[i]]

        # Create a mapping of unit_id to tile coordinate
        unit_to_tile = {}
        for r, c in zip(row_ind, col_ind):
            unit_id = selected_units[r]
            tile_x, tile_y = selected_tiles[c]
            unit_to_tile[unit_id] = (tile_x, tile_y)

        # Units not assigned remain unassigned
        unassigned_units = set(available_unit_ids) - set(unit_to_tile.keys())

        # Move assigned units towards their assigned tiles
        for unit_id, (tile_x, tile_y) in unit_to_tile.items():
            ux, uy = unit_positions[unit_id]
            if ux == tile_x and uy == tile_y:
                # Already there, do nothing
                actions[unit_id] = [0, 0, 0]
            else:
                direction = direction_to(np.array([ux, uy]), np.array([tile_x, tile_y]))
                actions[unit_id] = [direction, 0, 0]

        # Unassigned units do nothing
        for unit_id in unassigned_units:
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
