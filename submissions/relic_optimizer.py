import json
from argparse import Namespace

import numpy as np
from scipy.optimize import linear_sum_assignment

def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1

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


    def update_tile_results(self, current_points, obs):

        # If we had a tester tile assigned last turn:
        if self.current_tester_tile and self.current_tester_tile_relic:
            # Check if a friendly unit is currently standing on the tile we tested last turn.
            tile_occupied = False
            for uid in np.where(obs["units_mask"][self.team_id])[0]:
                ux, uy = obs["units"]["position"][self.team_id][uid]
                if (ux, uy) == self.current_tester_tile:
                    tile_occupied = True
                    break

            if tile_occupied:
                # Proceed with the gain comparison logic
                gain = current_points - self.last_team_points
                if gain > self.expected_baseline_gain:
                    self.relic_tile_data[self.current_tester_tile_relic][self.current_tester_tile]["reward_tile"] = True
                else:
                    self.relic_tile_data[self.current_tester_tile_relic][self.current_tester_tile]["reward_tile"] = False
                self.relic_tile_data[self.current_tester_tile_relic][self.current_tester_tile]["tested"] = True
            else:
                # The unit has not reached the tile yet, do not mark it tested or non-reward.
                # Instead, keep self.current_tester_tile and retry next turn.
                pass

            if tile_occupied:
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
        self.update_tile_results(current_team_points, obs)

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

def from_json(state):
    if isinstance(state, list):
        return np.array(state)
    elif isinstance(state, dict):
        out = {}
        for k in state:
            out[k] = from_json(state[k])
        return out
    else:
        return state

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
        agent_dict[player] = RelicHuntingShootingAgent(player, configurations["env_cfg"])
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