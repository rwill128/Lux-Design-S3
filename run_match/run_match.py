import numpy as np
import os
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

from agents.baseline_agent.baselineagent import BaselineAgent
import numpy as np
from lux.utils import direction_to
from scipy.optimize import linear_sum_assignment

import numpy as np
from lux.utils import direction_to
from scipy.optimize import linear_sum_assignment

import numpy as np
from lux.utils import direction_to
from scipy.optimize import linear_sum_assignment

from scipy.optimize import linear_sum_assignment
import numpy as np
import math

def direction_to(from_pos, to_pos):
    # Simple directional logic: return an action code that moves from_pos closer to to_pos
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    # Prioritize horizontal or vertical moves
    # Example action encoding: 1=UP,2=RIGHT,3=DOWN,4=LEFT,0=STAY
    # You may need to adjust this depending on your environment's move coding
    if abs(dx) > abs(dy):
        return 2 if dx > 0 else 4  # move horizontally
    else:
        return 3 if dy > 0 else 1  # move vertically

class RelicHuntingShootingAgent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        # Attributes for relic logic
        self.last_team_points = 0
        self.last_relic_gain = 0
        self.relic_allocation = 20
        self.current_tester_tile = None

        self.last_team_points = 0
        self.relic_tile_data = {}
        # relic_tile_data[relic_position] = {
        #    (tile_x, tile_y): {"tested": bool, "reward_tile": bool}
        # }


    def update_tile_results(self, current_points):
        gain = current_points - self.last_team_points

        # If we had a tester tile assigned last turn:
        if self.current_tester_tile and self.current_tester_tile_relic:
            # If gain increased by 1 compared to expected baseline,
            # we mark this tile as reward_tile = True
            if gain > self.expected_baseline_gain:
                self.relic_tile_data[self.current_tester_tile_relic][self.current_tester_tile]["reward_tile"] = True
            else:
                # no additional reward from that tile
                self.relic_tile_data[self.current_tester_tile_relic][self.current_tester_tile]["reward_tile"] = False
            self.relic_tile_data[self.current_tester_tile_relic][self.current_tester_tile]["tested"] = True
            self.current_tester_tile = None

    def select_tiles_for_relic(self, relic_pos):
        # known reward tiles:
        reward_tiles = [t for t, d in self.relic_tile_data[relic_pos].items() if d["reward_tile"]]
        # unknown tiles:
        untested_tiles = [t for t, d in self.relic_tile_data[relic_pos].items() if not d["tested"]]

        return reward_tiles, untested_tiles

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (T, N) but we index with self.team_id
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape (N, 2)
        unit_energy = np.array(obs["units"]["energy"][self.team_id])  # shape (N,)

        opp_positions = np.array(obs["units"]["position"][self.opp_team_id])  # (N, 2)
        opp_energy = np.array(obs["units"]["energy"][self.opp_team_id])       # (N,)

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        available_unit_ids = np.where(unit_mask)[0]
        num_units = len(available_unit_ids)

        current_team_points = obs["team_points"][self.team_id]

        # Update results from last turn's test
        self.update_tile_results(current_team_points)

        # Discover visible relic nodes and initialize their tile data if needed
        relic_nodes_mask = obs["relic_nodes_mask"]
        relic_nodes_positions = obs["relic_nodes"][relic_nodes_mask]

        for (rx, ry) in relic_nodes_positions:
            if (rx, ry) not in self.relic_tile_data:
                # Initialize the 5x5 area around (rx, ry)
                self.relic_tile_data[(rx, ry)] = {}
                for bx in range(rx-2, rx+3):
                    for by in range(ry-2, ry+3):
                        if 0 <= bx < self.env_cfg["map_width"] and 0 <= by < self.env_cfg["map_height"]:
                            self.relic_tile_data[(rx, ry)][(bx, by)] = {"tested": False, "reward_tile": False}

        # Assign units:
        # 1) Always occupy known reward tiles.
        # 2) If there are untested tiles and spare units, pick one tile to test this turn.

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        available_unit_ids = np.where(obs["units_mask"][self.team_id])[0]

        # First fill reward tiles:
        used_units = set()
        for (rx, ry) in relic_nodes_positions:
            reward_tiles, untested_tiles = self.select_tiles_for_relic((rx, ry))
            # Place units on reward tiles first
            for tile in reward_tiles:
                if len(available_unit_ids) == 0:
                    break
                u = available_unit_ids[0]
                available_unit_ids = available_unit_ids[1:]
                # Move unit u to tile (prefer minimal distance)
                ux, uy = obs["units"]["position"][self.team_id][u]
                tx, ty = tile
                if (ux, uy) != (tx, ty):
                    direction = direction_to(np.array([ux, uy]), np.array([tx, ty]))
                    actions[u] = [direction, 0, 0]
                else:
                    actions[u] = [0,0,0]
                used_units.add(u)

            # If we have untested tiles and still have units left, test one tile this turn
            # but only one tile at a time for clear inference.
            if untested_tiles and len(available_unit_ids) > 0:
                test_tile = untested_tiles[0]
                u = available_unit_ids[0]
                available_unit_ids = available_unit_ids[1:]
                # Move unit u to test_tile
                ux, uy = obs["units"]["position"][self.team_id][u]
                tx, ty = test_tile
                if (ux, uy) != (tx, ty):
                    direction = direction_to(np.array([ux, uy]), np.array([tx, ty]))
                    actions[u] = [direction, 0, 0]
                else:
                    actions[u] = [0,0,0]
                used_units.add(u)
                # Record this testing action for next turn's inference
                self.current_tester_tile = test_tile
                self.current_tester_tile_relic = (rx, ry)
                # Expected baseline gain is the sum of currently known reward tiles occupied
                self.expected_baseline_gain = len(reward_tiles)  # since each yields 1 point presumably
                # Once we place a tester, break out if you only want to test one tile total
                break

        if num_units == 0:
            # No units available
            return actions

        map_width = self.env_cfg["map_width"]
        map_height = self.env_cfg["map_height"]

        # Sap parameters
        sap_range = self.env_cfg.get("unit_sap_range", 1)
        sap_cost = self.env_cfg.get("unit_sap_cost", 10)

        # Identify visible opponent units
        opp_visible_mask = (opp_positions[:,0] != -1) & (opp_positions[:,1] != -1)
        visible_opp_ids = np.where(opp_visible_mask)[0]

        # Enemy positions dictionary for sapping logic
        enemy_positions = {}
        for oid in visible_opp_ids:
            ex, ey = opp_positions[oid]
            if (ex, ey) not in enemy_positions:
                enemy_positions[(ex, ey)] = []
            enemy_positions[(ex, ey)].append(oid)

        # Determine visible relics
        # obs["relic_nodes_mask"] = shape (R,)
        # obs["relic_nodes"] = shape (R, 2)
        relic_nodes_mask = obs["relic_nodes_mask"]
        relic_nodes_positions = obs["relic_nodes"][relic_nodes_mask]  # This gives only visible relics
        relics = relic_nodes_positions.tolist()  # list of (x, y) for visible relics

        # Calculate current relic gain based on team points difference
        current_team_points = obs["team_points"][self.team_id]
        current_relic_gain = current_team_points - self.last_team_points

        # Adjust relic_allocation
        # if len(relics) > 0:
        #     pass
            # if current_relic_gain > self.last_relic_gain:
            #     self.relic_allocation = min(self.relic_allocation + 1, num_units)
            # else:
            #     self.relic_allocation = max(self.relic_allocation - 1, 1)
        # else:
        #     self.relic_allocation = 0

        self.last_relic_gain = current_relic_gain

        # Step 1: Attempt sapping
        sap_done = set()
        for unit_id in available_unit_ids:
            ux, uy = unit_positions[unit_id]
            uenergy = unit_energy[unit_id]

            if uenergy > sap_cost:
                found_target = False
                for dx in range(-sap_range, sap_range + 1):
                    for dy in range(-sap_range, sap_range + 1):
                        tx = ux + dx
                        ty = uy + dy
                        if tx < 0 or tx >= map_width or ty < 0 or ty >= map_height:
                            continue

                        # Count enemies in the 3x3 area centered on (tx, ty)
                        center_count = 0
                        adjacent_count = 0
                        for adjx in [-1, 0, 1]:
                            for adjy in [-1, 0, 1]:
                                cx = tx + adjx
                                cy = ty + adjy
                                if (cx, cy) in enemy_positions:
                                    ccount = len(enemy_positions[(cx, cy)])
                                    if adjx == 0 and adjy == 0:
                                        center_count = ccount
                                    else:
                                        adjacent_count += ccount

                        if center_count > 0 and adjacent_count > 0:
                            actions[unit_id] = [5, dx, dy]
                            sap_done.add(unit_id)
                            found_target = True
                            break
                    if found_target:
                        break

        # Units that didn't sap proceed with relic or vision logic
        remaining_units = [u for u in available_unit_ids if u not in sap_done]
        if len(remaining_units) == 0:
            # Update last_team_points before returning
            self.last_team_points = current_team_points
            return actions

        # If we have relics and an allocation, choose some units to go to relic areas
        relic_targets = []
        if len(relics) > 0:
            # Pick a pattern around each relic - a 5x5 block centered on the relic
            block_radius = 2
            for (rx, ry) in relics:
                for bx in range(rx - block_radius, rx + block_radius + 1):
                    for by in range(ry - block_radius, ry + block_radius + 1):
                        if 0 <= bx < map_width and 0 <= by < map_height:
                            relic_targets.append((bx, by))
            # Deduplicate
            relic_targets = list(set(relic_targets))

        # Relic assignment
        if self.relic_allocation > 0 and len(relic_targets) > 0:
            relic_units_count = min(self.relic_allocation, len(remaining_units), len(relic_targets))
            relic_cost_matrix = np.zeros((len(remaining_units), len(relic_targets)), dtype=int)
            for i, u in enumerate(remaining_units):
                ux, uy = unit_positions[u]
                for j, (tx, ty) in enumerate(relic_targets):
                    dist = abs(ux - tx) + abs(uy - ty)
                    relic_cost_matrix[i, j] = dist

            row_ind, col_ind = linear_sum_assignment(relic_cost_matrix)
            pairs = sorted(zip(row_ind, col_ind), key=lambda rc: relic_cost_matrix[rc[0], rc[1]])

            assigned_to_relic = set()
            relic_unit_to_target = {}
            for r, c in pairs[:relic_units_count]:
                u = remaining_units[r]
                relic_unit_to_target[u] = relic_targets[c]
                assigned_to_relic.add(u)

            remaining_units = [u for u in remaining_units if u not in assigned_to_relic]

            for u, (tx, ty) in relic_unit_to_target.items():
                ux, uy = unit_positions[u]
                if ux == tx and uy == ty:
                    actions[u] = [0, 0, 0]
                else:
                    direction = direction_to(np.array([ux, uy]), np.array([tx, ty]))
                    actions[u] = [direction, 0, 0]

        # Vision assignment for remaining units
        if len(remaining_units) > 0:
            still_num_units = len(remaining_units)
            rows = int(np.floor(np.sqrt(still_num_units)))
            cols = int(np.ceil(still_num_units / rows))
            while rows * cols < still_num_units:
                cols += 1

            cell_width = self.env_cfg["map_width"] / cols
            cell_height = self.env_cfg["map_height"] / rows
            assigned_cell_count = min(still_num_units, rows*cols)
            targets = []
            for i in range(assigned_cell_count):
                r = i // cols
                c = i % cols
                cell_center_x = int((c + 0.5) * cell_width)
                cell_center_y = int((r + 0.5) * cell_height)
                cell_center_x = min(cell_center_x, map_width - 1)
                cell_center_y = min(cell_center_y, map_height - 1)
                targets.append((cell_center_x, cell_center_y))

            num_remaining = still_num_units
            used_cell_count = min(num_remaining, assigned_cell_count)
            if used_cell_count == 0:
                for u in remaining_units:
                    actions[u] = [0, 0, 0]
            else:
                cost_matrix = np.zeros((num_remaining, used_cell_count), dtype=int)
                for i, unit_id in enumerate(remaining_units):
                    ux, uy = unit_positions[unit_id]
                    for j in range(used_cell_count):
                        tx, ty = targets[j]
                        dist = abs(ux - tx) + abs(uy - ty)
                        cost_matrix[i, j] = dist

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                unit_to_target = {}
                for r, c in zip(row_ind, col_ind):
                    unit_id = remaining_units[r]
                    tx, ty = targets[c]
                    unit_to_target[unit_id] = (tx, ty)

                assigned_units = set(unit_to_target.keys())
                unassigned_units = set(remaining_units) - assigned_units

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

        # Update last_team_points for next turn
        self.last_team_points = current_team_points

        return actions

class BetterShootingVisionAgent:
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
        unit_energy = np.array(obs["units"]["energy"][self.team_id])  # shape (max_units,)

        opp_positions = np.array(obs["units"]["position"][self.opp_team_id])  # (max_units, 2)
        opp_energy = np.array(obs["units"]["energy"][self.opp_team_id])       # (max_units,)

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        available_unit_ids = np.where(unit_mask)[0]
        num_units = len(available_unit_ids)

        if num_units == 0:
            # No units available
            return actions

        map_width = self.env_cfg["map_width"]
        map_height = self.env_cfg["map_height"]

        # Sap parameters
        sap_range = self.env_cfg.get("unit_sap_range", 1)
        sap_cost = self.env_cfg.get("unit_sap_cost", 10)
        dropoff_factor = self.env_cfg.get("unit_sap_dropoff_factor", 1.0)  # if needed

        # Identify visible opponent units (opp_positions != -1)
        opp_visible_mask = (opp_positions[:,0] != -1) & (opp_positions[:,1] != -1)
        visible_opp_ids = np.where(opp_visible_mask)[0]

        # Convert visible opponent positions into a quick lookup for checking enemy counts
        # A simple approach is to build a set of positions occupied by enemies
        enemy_positions = {}
        for oid in visible_opp_ids:
            ex, ey = opp_positions[oid]
            if (ex, ey) not in enemy_positions:
                enemy_positions[(ex, ey)] = []
            enemy_positions[(ex, ey)].append(oid)

        sap_done = set()  # units that have performed a sap action

        # Sapping logic:
        # Only sap if we can hit multiple enemy units. That means we should find a target tile
        # within sap_range that contains at least 1 enemy unit and at least 1 other enemy in one
        # of its 8 adjacent cells. Essentially, we want at least two enemies in the 3x3 area.
        # We'll iterate over potential target tiles within range and check enemy clustering.

        for unit_id in available_unit_ids:
            ux, uy = unit_positions[unit_id]
            uenergy = unit_energy[unit_id]

            if uenergy > sap_cost:
                # Try all tiles within sap_range
                found_target = False
                for dx in range(-sap_range, sap_range + 1):
                    for dy in range(-sap_range, sap_range + 1):
                        tx = ux + dx
                        ty = uy + dy
                        # Check map bounds
                        if tx < 0 or tx >= map_width or ty < 0 or ty >= map_height:
                            continue

                        # Count enemies in the 3x3 area centered on (tx, ty)
                        # Center tile: (tx, ty)
                        # Adjacent tiles: (tx+adjx, ty+adjy) for adjx in [-1,0,1], adjy in [-1,0,1]
                        # We'll check if there's at least 1 enemy in (tx, ty) and at least 1 enemy in any adjacent tile.
                        center_count = 0
                        adjacent_count = 0
                        for adjx in [-1, 0, 1]:
                            for adjy in [-1, 0, 1]:
                                cx = tx + adjx
                                cy = ty + adjy
                                if (cx, cy) in enemy_positions:
                                    ccount = len(enemy_positions[(cx, cy)])
                                    if adjx == 0 and adjy == 0:
                                        center_count = ccount  # enemies on the target tile
                                    else:
                                        adjacent_count += ccount
                        # Now we have how many enemies are in the center tile and how many in adjacent tiles
                        # We only sap if there's at least 1 enemy in the center tile and at least 1 enemy in adjacent tiles
                        if center_count > 0 and adjacent_count > 0:
                            # We found a good target to sap
                            actions[unit_id] = [5, dx, dy]
                            sap_done.add(unit_id)
                            found_target = True
                            break
                    if found_target:
                        break

        # Units that didn't sap proceed with the normal "vision" logic
        remaining_units = [u for u in available_unit_ids if u not in sap_done]
        if len(remaining_units) == 0:
            # All units sapped, no need for assignment
            return actions

        # Compute grid assignment for exploration
        rows = int(np.floor(np.sqrt(num_units)))
        cols = int(np.ceil(num_units / rows))
        while rows * cols < num_units:
            cols += 1

        cell_width = self.env_cfg["map_width"] / cols
        cell_height = self.env_cfg["map_height"] / rows
        assigned_cell_count = min(num_units, rows*cols)
        targets = []
        for i in range(assigned_cell_count):
            r = i // cols
            c = i % cols
            cell_center_x = int((c + 0.5) * cell_width)
            cell_center_y = int((r + 0.5) * cell_height)
            cell_center_x = min(cell_center_x, map_width - 1)
            cell_center_y = min(cell_center_y, map_height - 1)
            targets.append((cell_center_x, cell_center_y))

        # Filter cost matrix and assignment only for units that did not sap
        num_remaining = len(remaining_units)
        used_cell_count = min(num_remaining, assigned_cell_count)
        if used_cell_count == 0:
            # No cells or no units left
            # The units that didn't sap and we couldn't assign just stay put
            for u in remaining_units:
                actions[u] = [0, 0, 0]
            return actions

        cost_matrix = np.zeros((num_remaining, used_cell_count), dtype=int)
        for i, unit_id in enumerate(remaining_units):
            ux, uy = unit_positions[unit_id]
            for j in range(used_cell_count):
                tx, ty = targets[j]
                dist = abs(ux - tx) + abs(uy - ty)
                cost_matrix[i, j] = dist

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        unit_to_target = {}
        for r, c in zip(row_ind, col_ind):
            unit_id = remaining_units[r]
            tx, ty = targets[c]
            unit_to_target[unit_id] = (tx, ty)

        assigned_units = set(unit_to_target.keys())
        unassigned_units = set(remaining_units) - assigned_units

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
    evaluate_agents(BaselineAgent, RelicHuntingShootingAgent, games_to_play=3,
                    replay_save_dir="replays/" + BaselineAgent.__name__ + "_" + RelicHuntingShootingAgent.__name__)

    # After running, you can check the "replays" directory for saved replay files.
    # You can set breakpoints anywhere in this file or inside the Agent class.
