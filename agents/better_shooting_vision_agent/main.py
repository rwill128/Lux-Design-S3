import json
from argparse import Namespace

import numpy as np
from scipy.optimize import linear_sum_assignment

from lux.kit import from_json
from lux.utils import direction_to


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


### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = dict() # store potentially multiple dictionaries as kaggle imports code directly
agent_prev_obs = dict()
def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    global agent_dict
    obs = observation.obs
    if type(obs) == str:
        obs = json.loads(obs)
    step = observation.step
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        agent_dict[player] = BetterShootingVisionAgent(player, configurations["env_cfg"])
    agent = agent_dict[player]
    actions = agent.act(step, from_json(obs), remainingOverageTime)
    return dict(action=actions.tolist())
if __name__ == "__main__":
    
    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)
    step = 0
    player_id = 0
    env_cfg = None
    i = 0
    while True:
        inputs = read_input()
        raw_input = json.loads(inputs)
        observation = Namespace(**dict(step=raw_input["step"], obs=raw_input["obs"], remainingOverageTime=raw_input["remainingOverageTime"], player=raw_input["player"], info=raw_input["info"]))
        if i == 0:
            env_cfg = raw_input["info"]["env_cfg"]
            player_id = raw_input["player"]
        i += 1
        actions = agent_fn(observation, dict(env_cfg=env_cfg))
        # send actions to engine
        print(json.dumps(actions))