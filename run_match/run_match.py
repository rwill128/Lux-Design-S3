import os
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

from agents.baseline_agent.baselineagent import BaselineAgent

import numpy as np
from scipy.optimize import linear_sum_assignment


class RelicHuntingShootingAgent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        self.last_team_points = 0
        self.last_relic_gain = 0
        self.relic_allocation = 20
        self.current_tester_tile = None
        self.current_tester_tile_relic = None
        self.expected_baseline_gain = 0

        self.relic_tile_data = {}
        self.end_of_match_printed = False
        self.last_unit_positions = []  # store positions of units from previous turn

        self.possible_reward_tiles = set()
        self.unknown_tiles = set()
        self.not_reward_tiles = set()
        self.known_reward_tiles = set()
        self.last_gain = 0
        self.last_unknown_occupied = set()

        # New attribute to store known relic locations across games
        self.known_relic_positions = []  # list of (x, y) relic coordinates known from previous games

    def simple_heuristic_move(self, from_pos, to_pos):
        # ... unchanged ...
        (fx, fy) = from_pos
        (tx, ty) = to_pos
        dx = tx - fx
        dy = ty - fy
        if abs(dx) > abs(dy):
            return 2 if dx > 0 else 4  # move right or left
        else:
            return 3 if dy > 0 else 1  # move down or up

    def dxdy_to_action(self, dx, dy):
        # ... unchanged ...
        if dx > 0:
            return 2  # right
        elif dx < 0:
            return 4  # left
        elif dy > 0:
            return 3  # down
        elif dy < 0:
            return 1  # up
        return 0  # stay

    def bfs_pathfind(self, map_width, map_height, start, goal, obs):
        # ... unchanged ...
        sensor_mask = obs["sensor_mask"]
        tile_type_map = obs["map_features"]["tile_type"]

        def is_passable(x, y):
            if x < 0 or x >= map_width or y < 0 or y >= map_height:
                return False
            if sensor_mask[x, y]:
                # We can see the tile, check if asteroid
                if tile_type_map[x, y] == 2:
                    return False
                return True
            else:
                return True

        from collections import deque
        queue = deque([start])
        came_from = {start: None}

        while queue:
            current = queue.popleft()
            if current == goal:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            (cx, cy) = current
            for dx, dy in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < map_width and 0 <= ny < map_height:
                    if is_passable(nx, ny) and (nx, ny) not in came_from:
                        came_from[(nx, ny)] = (cx, cy)
                        queue.append((nx, ny))
        return None

    def get_direction_via_pathfinding(self, from_pos, to_pos, obs):
        # ... unchanged ...
        if from_pos == to_pos:
            return 0  # Already at target
        path = self.bfs_pathfind(self.env_cfg["map_width"], self.env_cfg["map_height"], from_pos, to_pos, obs)
        if path is None or len(path) < 2:
            return self.simple_heuristic_move(from_pos, to_pos)
        next_step = path[1]
        dx = next_step[0] - from_pos[0]
        dy = next_step[1] - from_pos[1]
        return self.dxdy_to_action(dx, dy)

    def update_tile_results(self, current_points, obs):
        # ... unchanged ...
        baseline = 0
        for (x, y) in self.last_unit_positions:
            for relic_pos, tiles_data in self.relic_tile_data.items():
                if (x, y) in tiles_data and tiles_data[(x, y)]["reward_tile"]:
                    baseline += 1

        gain = current_points - self.last_team_points
        extra = gain - baseline
        if extra <= 0:
            return

        unknown_tiles_occupied_last_turn = []
        for (x, y) in self.last_unit_positions:
            for relic_pos, tiles_data in self.relic_tile_data.items():
                if (x, y) in tiles_data:
                    tile_info = tiles_data[(x, y)]
                    if tile_info["tested"] == False and tile_info["reward_tile"] == False:
                        unknown_tiles_occupied_last_turn.append((relic_pos, (x, y)))

        if len(unknown_tiles_occupied_last_turn) > 0:
            selected_candidates = unknown_tiles_occupied_last_turn[:extra]
            for relic_pos, tile_pos in selected_candidates:
                self.relic_tile_data[relic_pos][tile_pos]["reward_tile"] = True
                self.relic_tile_data[relic_pos][tile_pos]["tested"] = True
            self.current_tester_tile = None
            self.current_tester_tile_relic = None

    def select_tiles_for_relic(self, relic_pos):
        # ... unchanged ...
        reward_tiles = [t for t, d in self.relic_tile_data[relic_pos].items() if d["reward_tile"]]
        untested_tiles = [t for t, d in self.relic_tile_data[relic_pos].items() if not d["tested"]]
        return reward_tiles, untested_tiles

    def deduce_reward_tiles(self, obs):
        # Current points
        current_team_points = obs["team_points"][self.team_id]
        # If current_team_points is a scalar array, convert to python int
        if hasattr(current_team_points, 'item'):
            current_team_points = current_team_points.item()
        gain = current_team_points - self.last_team_points

        # Occupied unknown tiles this turn
        unit_positions = obs["units"]["position"][self.team_id]
        unit_mask = obs["units_mask"][self.team_id].astype(bool)
        occupied_this_turn = set()
        for uid in np.where(unit_mask)[0]:
            x,y = unit_positions[uid]
            if (x,y) in self.unknown_tiles:
                occupied_this_turn.add((x,y))

        # Determine if gain rate went down compared to last turn's gain
        last_gain = getattr(self, 'last_gain', 0)  # if not set, assume 0 from previous turn
        gain_drop = gain < last_gain

        # Case 1: No or Negative Gain
        if gain <= 0:
            # No point gain means no unknown tile contributed points this turn
            # Thus all unknown tiles occupied last turn are not reward
            self.not_reward_tiles.update(self.last_unknown_occupied)
            # If currently occupied unknown tiles were also occupied last turn with no gain,
            # they are also not reward
            self.not_reward_tiles.update(occupied_this_turn.intersection(self.last_unknown_occupied))
            self.unknown_tiles -= self.not_reward_tiles

        else:
            # gain > 0
            newly_occupied = occupied_this_turn - self.last_unknown_occupied
            if len(newly_occupied) == 1:
                # Exactly one new tile caused the gain
                self.known_reward_tiles.update(newly_occupied)
                self.unknown_tiles -= newly_occupied
            elif len(newly_occupied) > 1:
                # More than one new unknown tile is occupied, can't deduce which is reward
                pass
            else:
                # gain > 0 but no new tiles were occupied?
                # This should not happen if our logic relies on new occupancy for gain
                pass
                # assert False, "Points went up but no new unknown tile was occupied."

        # Additional logic: If gain rate dropped compared to last turn
        if gain_drop:
            # We had fewer points gained this turn than last turn
            # This suggests we lost a reward tile occupant
            # Tiles that were occupied last turn but not this turn:
            newly_unoccupied = self.last_unknown_occupied - occupied_this_turn
            if len(newly_unoccupied) == 1:
                # Exactly one tile was vacated and gain rate dropped
                # That tile must have been a reward tile that we lost
                self.known_reward_tiles.update(newly_unoccupied)
                self.unknown_tiles -= newly_unoccupied
            elif len(newly_unoccupied) > 1:
                # More than one tile vacated - can't deduce which one caused the drop
                pass

        # Update tracking
        self.last_unknown_occupied = occupied_this_turn
        self.last_team_points = current_team_points
        self.last_gain = gain  # Store current gain for next turn

        # Print current categorization results
        print("Possible Tiles:", len(self.possible_reward_tiles))
        print("Unknown Tiles:", len(self.unknown_tiles))
        print("Not Reward Tiles:", len(self.not_reward_tiles))
        print("Known Reward Tiles:", len(self.known_reward_tiles))
        if len(self.known_reward_tiles) > 0:
            print("Known Rewards:", self.known_reward_tiles)

    def update_possible_reward_tiles(self, obs):
        # ... unchanged ...
        relic_nodes_mask = obs["relic_nodes_mask"]
        relic_nodes = obs["relic_nodes"][relic_nodes_mask]

        new_possible = set()
        block_radius = 2
        map_width = self.env_cfg["map_width"]
        map_height = self.env_cfg["map_height"]
        for (rx, ry) in relic_nodes:
            for bx in range(rx - block_radius, rx + block_radius + 1):
                for by in range(ry - block_radius, ry + block_radius + 1):
                    if 0 <= bx < map_width and 0 <= by < map_height:
                        new_possible.add((bx, by))

        self.possible_reward_tiles = new_possible
        currently_unknown = self.possible_reward_tiles - self.known_reward_tiles - self.not_reward_tiles
        self.unknown_tiles = currently_unknown

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energy = np.array(obs["units"]["energy"][self.team_id])

        opp_positions = np.array(obs["units"]["position"][self.opp_team_id])

        available_unit_ids = np.where(unit_mask)[0]
        num_units = len(available_unit_ids)

        current_team_points = obs["team_points"][self.team_id]

        # 1) Update possible reward tiles
        self.update_possible_reward_tiles(obs)

        # 2) Deduce reward tiles based on occupancy and point gains
        self.deduce_reward_tiles(obs)

        # 3) Update global tile results
        self.update_tile_results(current_team_points, obs)

        relic_nodes_mask = obs["relic_nodes_mask"]
        relic_nodes_positions = obs["relic_nodes"][relic_nodes_mask]

        # Add newly discovered relics to known_relic_positions
        for (rx, ry) in relic_nodes_positions:
            if (rx, ry) not in self.relic_tile_data:
                self.relic_tile_data[(rx, ry)] = {}
                for bx in range(rx-2, rx+3):
                    for by in range(ry-2, ry+3):
                        if 0 <= bx < self.env_cfg["map_width"] and 0 <= by < self.env_cfg["map_height"]:
                            self.relic_tile_data[(rx, ry)][(bx, by)] = {"tested": False, "reward_tile": False}
            # If it's a new relic not seen in previous games, store it
            if (rx, ry) not in self.known_relic_positions:
                self.known_relic_positions.append((rx, ry))

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        available_unit_ids = np.where(obs["units_mask"][self.team_id])[0]

        if num_units == 0:
            self.last_team_points = current_team_points
            self.last_unit_positions = []
            for uid in np.where(obs["units_mask"][self.team_id])[0]:
                ux, uy = obs["units"]["position"][self.team_id][uid]
                self.last_unit_positions.append((ux, uy))
            return actions

        map_width = self.env_cfg["map_width"]
        map_height = self.env_cfg["map_height"]

        sap_range = self.env_cfg.get("unit_sap_range", 1)
        sap_cost = self.env_cfg.get("unit_sap_cost", 10)

        opp_visible_mask = (opp_positions[:,0] != -1) & (opp_positions[:,1] != -1)
        visible_opp_ids = np.where(opp_visible_mask)[0]

        enemy_positions = {}
        for oid in visible_opp_ids:
            ex, ey = opp_positions[oid]
            if (ex, ey) not in enemy_positions:
                enemy_positions[(ex, ey)] = []
            enemy_positions[(ex, ey)].append(oid)

        relics = relic_nodes_positions.tolist()

        current_relic_gain = current_team_points - self.last_team_points
        self.last_relic_gain = current_relic_gain

        # Attempt sap action
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

        remaining_units = [u for u in available_unit_ids if u not in sap_done]
        if len(remaining_units) == 0:
            self.last_team_points = current_team_points
            self.last_unit_positions = []
            for uid in np.where(obs["units_mask"][self.team_id])[0]:
                ux, uy = obs["units"]["position"][self.team_id][uid]
                self.last_unit_positions.append((ux, uy))
            return actions

        relic_targets = list(self.known_relic_positions)
        if len(relics) > 0:
            block_radius = 2
            for (rx, ry) in relics:
                for bx in range(rx - block_radius, rx + block_radius + 1):
                    for by in range(ry - block_radius, ry + block_radius + 1):
                        if 0 <= bx < map_width and 0 <= by < map_height:
                            relic_targets.append((bx, by))
            relic_targets = list(set(relic_targets))

        relic_targets = list(set(list(set(relic_targets) - set(self.not_reward_tiles)) + list(self.known_reward_tiles)))

        # Prioritization constants
        REWARD_BONUS = -50
        NON_REWARD_PENALTY = 1000

        if self.relic_allocation > 0 and len(relic_targets) > 0:
            relic_units_count = min(self.relic_allocation, len(remaining_units), len(relic_targets))
            relic_cost_matrix = np.zeros((len(remaining_units), len(relic_targets)), dtype=int)
            for i, u in enumerate(remaining_units):
                ux, uy = unit_positions[u]
                for j, (tx, ty) in enumerate(relic_targets):
                    dist = abs(ux - tx) + abs(uy - ty)
                    cost = dist
                    if (tx, ty) in self.known_reward_tiles:
                        cost += REWARD_BONUS
                        if cost < 0:
                            cost = 0
                    if (tx, ty) in self.not_reward_tiles:
                        cost += NON_REWARD_PENALTY
                    relic_cost_matrix[i, j] = cost

            row_ind, col_ind = linear_sum_assignment(relic_cost_matrix)
            pairs = sorted(zip(row_ind, col_ind), key=lambda rc: relic_cost_matrix[rc[0], rc[1]])

            assigned_to_relic = set()
            relic_unit_to_target = {}
            used_positions = set()
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
                    direction = self.get_direction_via_pathfinding((ux, uy), (tx, ty), obs)
                    actions[u] = [direction, 0, 0]
                used_positions.add((tx, ty))

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
                        cost = dist
                        if (tx, ty) in self.known_reward_tiles:
                            cost += REWARD_BONUS
                            if cost < 0:
                                cost = 0
                        if (tx, ty) in self.not_reward_tiles:
                            cost += NON_REWARD_PENALTY
                        cost_matrix[i, j] = cost

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                unit_to_target = {}
                for r, c in zip(row_ind, col_ind):
                    unit_id = remaining_units[r]
                    tx, ty = targets[c]
                    unit_to_target[unit_id] = (tx, ty)

                assigned_units = set(unit_to_target.keys())
                unassigned_units = set(remaining_units) - assigned_units

                for unit_id, (tx, ty) in unit_to_target.items():
                    ux, uy = unit_positions[unit_id]
                    if ux == tx and uy == ty:
                        actions[unit_id] = [0, 0, 0]
                    else:
                        direction = self.get_direction_via_pathfinding((ux, uy), (tx, ty), obs)
                        actions[unit_id] = [direction, 0, 0]

                for unit_id in unassigned_units:
                    actions[unit_id] = [0, 0, 0]

        self.last_team_points = current_team_points
        self.last_unit_positions = []
        for uid in np.where(obs["units_mask"][self.team_id])[0]:
            ux, uy = obs["units"]["position"][self.team_id][uid]
            self.last_unit_positions.append((ux, uy))

        # If the match ended, print known relic positions and reward tiles
        # Do not clear self.known_relic_positions here, so it's usable next game
        if obs["steps"] == 500 and not self.end_of_match_printed:
            all_reward_tiles = []
            for relic_pos, tiles_data in self.relic_tile_data.items():
                for tile_pos, tile_info in tiles_data.items():
                    if tile_info["reward_tile"]:
                        all_reward_tiles.append((relic_pos, tile_pos))

            print("Known relic positions across games:", self.known_relic_positions)
            print("Known reward tiles at end of match:")
            for relic_pos, tile_pos in all_reward_tiles:
                print(f"Relic: {relic_pos}, Reward Tile: {tile_pos}")

            self.end_of_match_printed = True

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
    evaluate_agents(BaselineAgent, RelicHuntingShootingAgent, games_to_play=5,
                    replay_save_dir="replays/" + BaselineAgent.__name__ + "_" + RelicHuntingShootingAgent.__name__)

    # After running, you can check the "replays" directory for saved replay files.
    # You can set breakpoints anywhere in this file or inside the Agent class.
