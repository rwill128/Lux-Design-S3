import numpy as np
import os
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

from agents.baseline_agent.baselineagent import BaselineAgent
import numpy as np
from lux.utils import direction_to
from scipy.optimize import linear_sum_assignment

class BalancedAgent:
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
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        available_unit_ids = np.where(unit_mask)[0]
        num_units = len(available_unit_ids)

        if num_units == 0:
            # No units available
            return actions

        map_width = self.env_cfg["map_width"]
        map_height = self.env_cfg["map_height"]
        energy_map = obs["map_features"]["energy"]  # shape (W, H)

        # Determine a grid size for distribution
        rows = int(np.floor(np.sqrt(num_units)))
        cols = int(np.ceil(num_units / rows))
        while rows * cols < num_units:
            cols += 1

        # Compute cell size
        cell_width = map_width / cols
        cell_height = map_height / rows

        # We'll assign at most one unit per cell center, just like before
        assigned_cell_count = min(num_units, rows * cols)
        cell_centers = []
        for i in range(assigned_cell_count):
            r = i // cols
            c = i % cols
            cell_center_x = int((c + 0.5) * cell_width)
            cell_center_y = int((r + 0.5) * cell_height)
            cell_center_x = min(cell_center_x, map_width - 1)
            cell_center_y = min(cell_center_y, map_height - 1)
            cell_centers.append((cell_center_x, cell_center_y))

        # Calculate energy in each cell region
        # One approach: sum up all visible energy tiles within the cell's rectangular area
        # For simplicity, just sum up energy in a bounding box around the cell center.
        # The "area" of each cell is approximately cell_width x cell_height.
        # We'll round down to get the top-left corner and up for the bottom-right corner.
        cell_energies = []
        for (cx, cy) in cell_centers:
            # cell boundaries
            left = int(cy - cell_height/2)
            right = int(cy + cell_height/2)
            top = int(cx - cell_width/2)
            bottom = int(cx + cell_width/2)

            # Ensure boundaries are within the map
            left = max(0, left)
            right = min(map_height - 1, right)
            top = max(0, top)
            bottom = min(map_width - 1, bottom)

            # energy_map is indexed [y][x], so be careful with indexing
            # Summation over the cell region
            # Slice: energy_map[left:right+1, top:bottom+1]
            # Remember: W, H from obs might be (W,H) indexing differently. Usually energy_map is (H,W).
            # From your obs structure: "map_features" arrays are shape (H, W), so indexing is [y, x]
            cell_energy = np.sum(energy_map[left:right+1, top:bottom+1])
            # If no visible energy (or negative?), but we do not mask here. Negative means no tile?
            # If the environment uses -1 for "no tile," filter those out:
            if cell_energy < 0:
                # replace -1 tiles with 0
                region = energy_map[left:right+1, top:bottom+1]
                cell_energy = np.sum(region[region > -1])  # sum only valid energies
            cell_energies.append(cell_energy if cell_energy > 0 else 0)

        # Create cost matrix:
        # cost = distance / (1 + cell_energy)
        # If cell_energy is large, cost is lower, so units prefer that cell.
        selected_units = available_unit_ids[:assigned_cell_count]
        cost_matrix = np.zeros((assigned_cell_count, assigned_cell_count), dtype=float)
        for i, unit_id in enumerate(selected_units):
            ux, uy = unit_positions[unit_id]
            for j, (tx, ty) in enumerate(cell_centers):
                dist = abs(ux - tx) + abs(uy - ty)
                cell_energy = cell_energies[j]
                cost = dist / (1 + cell_energy)
                cost_matrix[i, j] = cost

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Assign units
        unit_to_target = {}
        for r, c in zip(row_ind, col_ind):
            unit_id = selected_units[r]
            tx, ty = cell_centers[c]
            unit_to_target[unit_id] = (tx, ty)

        assigned_units = set(unit_to_target.keys())
        unassigned_units = set(available_unit_ids) - assigned_units

        # Move assigned units towards targets
        for unit_id, (tx, ty) in unit_to_target.items():
            ux, uy = unit_positions[unit_id]
            if ux == tx and uy == ty:
                actions[unit_id] = [0, 0, 0]
            else:
                direction = direction_to(np.array([ux, uy]), np.array([tx, ty]))
                actions[unit_id] = [direction, 0, 0]

        # Unassigned units do nothing
        for unit_id in unassigned_units:
            actions[unit_id] = [0, 0, 0]

        return actions

class VisionAgent:
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
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        available_unit_ids = np.where(unit_mask)[0]
        num_units = len(available_unit_ids)

        if num_units == 0:
            # No units available
            return actions

        map_width = self.env_cfg["map_width"]
        map_height = self.env_cfg["map_height"]

        # Determine a grid size for distribution
        # A simple approach: try to make a square grid close to sqrt(num_units)
        rows = int(np.floor(np.sqrt(num_units)))
        cols = int(np.ceil(num_units / rows))

        # If there's a mismatch, it's possible rows*cols < num_units, adjust if needed
        while rows * cols < num_units:
            cols += 1

        # Now we have a rows x cols grid that can hold at least num_units positions.
        # We'll assign each unit to one cell in this grid.
        # Some cells might remain unused if rows*cols > num_units.

        # Compute cell size
        cell_width = map_width / cols
        cell_height = map_height / rows

        # Generate target positions (cell centers) for the top num_units cells
        targets = []
        assigned_cell_count = min(num_units, rows*cols)
        for i in range(assigned_cell_count):
            r = i // cols
            c = i % cols
            # center of this cell
            cell_center_x = int((c + 0.5) * cell_width)
            cell_center_y = int((r + 0.5) * cell_height)
            # Ensure we don't go out of bounds
            cell_center_x = min(cell_center_x, map_width - 1)
            cell_center_y = min(cell_center_y, map_height - 1)
            targets.append((cell_center_x, cell_center_y))

        # Create cost matrix (units x targets)
        # Use Manhattan distance for simplicity
        cost_matrix = np.zeros((num_units, assigned_cell_count), dtype=int)
        for i, unit_id in enumerate(available_unit_ids):
            ux, uy = unit_positions[unit_id]
            for j, (tx, ty) in enumerate(targets):
                dist = abs(ux - tx) + abs(uy - ty)
                cost_matrix[i, j] = dist

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # row_ind[i] is a unit index, col_ind[i] is a target index

        # Assign each matched unit to the corresponding target
        unit_to_target = {}
        for r, c in zip(row_ind, col_ind):
            unit_id = available_unit_ids[r]
            tx, ty = targets[c]
            unit_to_target[unit_id] = (tx, ty)

        # Units not in unit_to_target remain idle
        assigned_units = set(unit_to_target.keys())
        unassigned_units = set(available_unit_ids) - assigned_units

        # Move assigned units towards targets
        for unit_id, (tx, ty) in unit_to_target.items():
            ux, uy = unit_positions[unit_id]
            if ux == tx and uy == ty:
                # Already there
                actions[unit_id] = [0, 0, 0]
            else:
                direction = direction_to(np.array([ux, uy]), np.array([tx, ty]))
                actions[unit_id] = [direction, 0, 0]

        # Unassigned units do nothing
        for unit_id in unassigned_units:
            actions[unit_id] = [0, 0, 0]

        return actions

class EnergyAgent:
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
    evaluate_agents(BaselineAgent, BalancedAgent, games_to_play=3,
                    replay_save_dir="replays/" + BaselineAgent.__name__ + "_" + BalancedAgent.__name__)

    # After running, you can check the "replays" directory for saved replay files.
    # You can set breakpoints anywhere in this file or inside the Agent class.
