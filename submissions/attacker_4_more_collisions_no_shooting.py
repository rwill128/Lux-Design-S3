import json
from argparse import Namespace

import numpy as np
from scipy.optimize import linear_sum_assignment


class BestAgentAttackerMoreCollisions:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        self.last_team_points = 0
        self.relic_allocation = 20
        self.expected_baseline_gain = 0

        self.relic_tile_data = {}
        self.end_of_match_printed = False
        self.last_unit_positions = []  # store positions of units from previous turn

        self.possible_reward_tiles = set()
        self.last_reward_occupied = set()
        self.unknown_tiles = set()
        self.not_reward_tiles = set()
        self.known_reward_tiles = set()
        self.last_gain = 0
        self.last_unknown_occupied = set()
        self.newly_unoccupied_unknown = set()
        self.newly_unoccupied_known = set()

        # Confidence tracking for reward tiles
        self.tile_confidence = {}  # (x,y) -> confidence score (0-100)
        self.tile_visit_count = {}  # (x,y) -> number of times visited
        # Persistence across games
        self._persistent_tile_confidence = {}  # (x,y) -> confidence score
        self.known_relic_positions = []  # list of (x,y) relic coordinates

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
        if dx > 0:
            return 2  # right
        elif dx < 0:
            return 4  # left
        elif dy > 0:
            return 3  # down
        elif dy < 0:
            return 1  # up
        return 0  # stay

    def get_direction_via_pathfinding(self, from_pos, to_pos, obs):
        if from_pos == to_pos:
            return 0  # Already at target
        path = self.dijkstra_pathfind(self.env_cfg["map_width"], self.env_cfg["map_height"], from_pos, to_pos, obs)
        if path is None or len(path) < 2:
            return self.simple_heuristic_move(from_pos, to_pos)
        next_step = path[1]
        dx = next_step[0] - from_pos[0]
        dy = next_step[1] - from_pos[1]
        action = self.dxdy_to_action(dx, dy)
        assert isinstance(action, int), f"Action is not an integer! Found: {action} (type: {type(action)})"
        return action

    def select_tiles_for_relic(self, relic_pos):
        # ... unchanged ...
        reward_tiles = [t for t, d in self.relic_tile_data[relic_pos].items() if d["reward_tile"]]
        untested_tiles = [t for t, d in self.relic_tile_data[relic_pos].items() if not d["tested"]]
        return reward_tiles, untested_tiles

    def deduce_reward_tiles(self, obs):
        """Deduce reward tiles with confidence tracking.
        
        Improvements:
        - Confidence tracking for each tile (0-100)
        - Better handling of multi-unit movement
        - Persistence between games
        - Enhanced debugging and validation
        """
        # Current points
        current_team_points = obs["team_points"][self.team_id]
        # If current_team_points is a scalar array, convert to python int
        if hasattr(current_team_points, 'item'):
            current_team_points = current_team_points.item()
        gain = current_team_points - self.last_team_points
        
        # Load persistent confidence from previous games
        if not self.tile_confidence and self._persistent_tile_confidence:
            self.tile_confidence = self._persistent_tile_confidence.copy()

        # Occupied unknown tiles this turn
        unit_positions = obs["units"]["position"][self.team_id]
        unit_mask = obs["units_mask"][self.team_id].astype(bool)
        occupied_this_turn = set()
        for uid in np.where(unit_mask)[0]:
            x, y = unit_positions[uid]
            if (x, y) in self.unknown_tiles:
                occupied_this_turn.add((x, y))

        # Compute currently occupied known reward tiles
        currently_reward_occupied = set()
        for uid in np.where(unit_mask)[0]:
            x, y = unit_positions[uid]
            pos = (x, y)
            if pos in self.known_reward_tiles:
                currently_reward_occupied.add(pos)

        self.newly_unoccupied_unknown = self.last_unknown_occupied - occupied_this_turn
        self.newly_unoccupied_known = self.last_reward_occupied - currently_reward_occupied

        newly_unoccupied = self.newly_unoccupied_unknown.union(self.newly_unoccupied_known)

        # Determine if gain rate went down compared to last turn's gain
        last_gain = getattr(self, 'last_gain', 0)  # if not set, assume 0 from previous turn
        gain_rate = gain - last_gain

        # We have the same number of reward squares
        if gain == 0:
            # No point gain means all occupied tiles are not reward tiles
            self.not_reward_tiles.update(occupied_this_turn)
            self.unknown_tiles -= self.not_reward_tiles
            if len(currently_reward_occupied) != 0:
                # We think we're in a reward square but we gained nothing, we've made a mistake
                self.known_reward_tiles = self.known_reward_tiles - currently_reward_occupied
                self.not_reward_tiles.update(currently_reward_occupied)
                # assert False

        if gain_rate == 0:
            newly_occupied = occupied_this_turn - self.last_unknown_occupied

            if len(newly_occupied) == 1 and len(newly_unoccupied) == 0:
                self.not_reward_tiles.update(newly_occupied)

            if len(newly_unoccupied) == 1 and len(newly_occupied) == 0:
                self.not_reward_tiles.update(newly_unoccupied)

            if len(newly_occupied) == 1 and len(self.newly_unoccupied_known) == 1:
                self.known_reward_tiles.update(newly_occupied)
                self.unknown_tiles -= newly_occupied

        if gain_rate > 0:
            # We entered a new reward square
            newly_occupied = occupied_this_turn - self.last_unknown_occupied
            if len(newly_occupied) == 1 and len(self.newly_unoccupied_known) == 0:
                # Exactly one new tile caused the gain
                self.known_reward_tiles.update(newly_occupied)
                self.unknown_tiles -= newly_occupied
            elif len(newly_occupied) > 1:
                # More than one new unknown tile is occupied, can't deduce which is reward
                pass
            else:
                # gain_rate > 0 but no new tiles were occupied?
                # This should not happen if our logic relies on new occupancy for gain
                pass
                # assert False, "Points went up but no new unknown tile was occupied."

        # We have fewer reward squares
        if gain_rate < 0:
            # We had fewer points gained this turn than last turn
            # This suggests we lost a reward tile occupant
            # Tiles that were occupied last turn but not this turn:
            newly_occupied = occupied_this_turn - self.last_unknown_occupied
            if len(newly_occupied) == 1 and len(self.newly_unoccupied_known) == 0:
                # This isn't working correctly
                # For now it's degrading bot performance in subsequent rounds because we're doing a
                # pretty good job of finding reward tiles the first round and then marking them as non-reward incorrectly
                self.not_reward_tiles.update(newly_occupied)

            if len(newly_unoccupied) == 1:
                # Exactly one tile was vacated and gain rate dropped
                # That tile must have been a reward tile that we lost
                self.known_reward_tiles.update(newly_unoccupied)
                self.unknown_tiles -= newly_unoccupied
            elif len(newly_unoccupied) > 1:
                # More than one tile vacated - can't deduce which one caused the drop
                pass
            elif len(newly_unoccupied) == 0:
                pass
                # assert False, "We lost points but we don't have any newly unoccupied tiles?"

        # Update visit counts and confidence scores based on point gains
        for pos in occupied_this_turn:
            self.tile_visit_count[pos] = self.tile_visit_count.get(pos, 0) + 1
            
            # Handle confirmed tiles
            if pos in self.known_reward_tiles:
                # Maximum confidence for confirmed reward tiles
                self.tile_confidence[pos] = 100
            elif pos in self.not_reward_tiles:
                # Zero confidence for confirmed non-reward tiles
                self.tile_confidence[pos] = 0
            else:
                # For unknown tiles, update based on point gains
                current_confidence = self.tile_confidence.get(pos, 50)
                if gain > 0 and len(newly_occupied) == 1 and pos in newly_occupied:
                    # Single tile caused point gain - high confidence
                    self.tile_confidence[pos] = 100
                elif gain == 0 and pos in newly_occupied:
                    # No points gained from new tile - decrease confidence
                    visits = self.tile_visit_count[pos]
                    decay = min(20, 5 * visits)  # Increasing decay with visits, capped at 20
                    self.tile_confidence[pos] = max(0, current_confidence - decay)
            
        # Persist confidence
        self._persistent_tile_confidence = self.tile_confidence.copy()
        
        # Update tracking
        self.last_reward_occupied = currently_reward_occupied
        self.last_unknown_occupied = occupied_this_turn
        self.last_team_points = current_team_points
        self.last_gain = gain  # Store current gain for next turn

        self.last_unit_positions = []
        for uid in np.where(obs["units_mask"][self.team_id])[0]:
            ux, uy = obs["units"]["position"][self.team_id][uid]
            self.last_unit_positions.append((ux, uy))

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

    def dijkstra_pathfind(self, map_width, map_height, start, goal, obs):
        sensor_mask = obs["sensor_mask"]
        tile_type_map = obs["map_features"]["tile_type"]
        tile_energy_map = obs["map_features"]["energy"]

        def is_passable(x, y):
            if x < 0 or x >= map_width or y < 0 or y >= map_height:
                return False
            if sensor_mask[x, y]:
                # Visible tile
                if tile_type_map[x, y] == 2:  # Asteroid
                    return False
                return True
            else:
                # Unknown tile, assume passable
                return True

        import heapq
        dist = {(start): 0}
        came_from = {start: None}
        pq = [(0, start)]  # priority queue with tuples (cost, position)

        while pq:
            current_dist, current = heapq.heappop(pq)
            if current == goal:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            if current_dist > dist[current]:
                continue  # Already found a better path

            (cx, cy) = current
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < map_width and 0 <= ny < map_height and is_passable(nx, ny):
                    # Calculate the step cost using tile energy
                    tile_cost = 10 - tile_energy_map[nx, ny]
                    # Floor at 0
                    if tile_cost < 0:
                        tile_cost = 0

                    new_dist = current_dist + tile_cost
                    if (nx, ny) not in dist or new_dist < dist[(nx, ny)]:
                        dist[(nx, ny)] = new_dist
                        came_from[(nx, ny)] = (cx, cy)
                        heapq.heappush(pq, (new_dist, (nx, ny)))

        return None  # No path found

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        opp_positions = np.array(obs["units"]["position"][self.opp_team_id])

        # 1) Update possible reward tiles
        self.update_possible_reward_tiles(obs)

        # 2) Deduce reward tiles based on occupancy and point gains
        self.deduce_reward_tiles(obs)

        relic_nodes_mask = obs["relic_nodes_mask"]
        self.add_newly_discovered_relics(obs["relic_nodes"][relic_nodes_mask])

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        available_unit_ids = np.where(obs["units_mask"][self.team_id])[0]

        map_width = self.env_cfg["map_width"]
        map_height = self.env_cfg["map_height"]

        opp_visible_mask = (opp_positions[:, 0] != -1) & (opp_positions[:, 1] != -1)
        visible_opp_ids = np.where(opp_visible_mask)[0]

        enemy_positions = {}
        for oid in visible_opp_ids:
            ex, ey = opp_positions[oid]
            if (ex, ey) not in enemy_positions:
                enemy_positions[(ex, ey)] = []
            enemy_positions[(ex, ey)].append(oid)

        sap_done = set()
        # self.do_sapping_logic(actions,
        #                       available_unit_ids,
        #                       enemy_positions,
        #                       self.env_cfg.get("unit_sap_cost", 10),
        #                       sap_done,
        #                       self.env_cfg.get("unit_sap_range", 1),
        #                       np.array(obs["units"]["energy"][self.team_id]),
        #                       unit_positions,
        #                       opp_energy=np.array(obs["units"]["energy"][self.opp_team_id]))

        remaining_units = [u for u in available_unit_ids if u not in sap_done]

        NON_REWARD_PENALTY, REWARD_BONUS, remaining_units = self.send_to_relic_points(actions, map_height, map_width,
                                                                                      obs, remaining_units,
                                                                                      unit_positions)

        if max(obs['team_wins']) <= 0:
            self.send_to_explore_if_not_going_to_relic(NON_REWARD_PENALTY, REWARD_BONUS, actions, map_height, map_width,
                                                       obs, remaining_units, unit_positions)
        else:
            # This is where I'd like to attack instead.
            self.send_to_attack_if_not_going_to_relic(NON_REWARD_PENALTY, REWARD_BONUS, actions, map_height, map_width,
                                                      obs, remaining_units, unit_positions)

        # If the match ended, print known relic positions and reward tiles
        # Do not clear self.known_relic_positions here, so it's usable next game
        if obs["steps"] == 500 and not self.end_of_match_printed:
            all_reward_tiles = []
            for relic_pos, tiles_data in self.relic_tile_data.items():
                for tile_pos, tile_info in tiles_data.items():
                    if tile_info["reward_tile"]:
                        all_reward_tiles.append((relic_pos, tile_pos))

        # Before returning actions:
        a = actions[:, 0]  # action codes
        dx = actions[:, 1]
        dy = actions[:, 2]

        # Actions that are not "sap" (5) should remain in the original range
        non_sap_mask = (a != 5)
        assert np.all(a[non_sap_mask] >= 0), f"Non-sap actions must be >= 0. Got: {a[non_sap_mask]}"
        assert np.all(a[non_sap_mask] <= 4), f"Non-sap actions must be <= 4. Got: {a[non_sap_mask]}"
        assert np.all(dx[non_sap_mask] == 0), f"dx must be 0 for non-sap actions. Got: {dx[non_sap_mask]}"
        assert np.all(dy[non_sap_mask] == 0), f"dy must be 0 for non-sap actions. Got: {dy[non_sap_mask]}"

        # Actions that are "sap" (5) must have dx and dy in [-10, 10]
        sap_mask = (a == 5)
        assert np.all(dx[sap_mask] >= -10), f"Sap dx out of range. Got: {dx[sap_mask]}"
        assert np.all(dx[sap_mask] <= 10), f"Sap dx out of range. Got: {dx[sap_mask]}"
        assert np.all(dy[sap_mask] >= -10), f"Sap dy out of range. Got: {dy[sap_mask]}"
        assert np.all(dy[sap_mask] <= 10), f"Sap dy out of range. Got: {dy[sap_mask]}"

        actions = actions.astype(np.int32)
        return actions

    def send_to_attack_if_not_going_to_relic(self, NON_REWARD_PENALTY, REWARD_BONUS, actions, map_height, map_width,
                                             obs, remaining_units, unit_positions):
        if len(remaining_units) == 0:
            return

        # Determine enemy spawn corner based on our team_id or known game logic.
        # Example: If we are team 0 starting near (0,0), enemy is at (map_width-1, map_height-1).
        # If we are team 1 starting near (map_width-1,map_height-1), enemy is at (0,0).
        if self.team_id == 0:
            enemy_corner_x, enemy_corner_y = map_width - 1, map_height - 1
        else:
            enemy_corner_x, enemy_corner_y = 0, 0

        # We will create a set of target positions near the enemy corner.
        # Similar logic as exploration, but focused on the enemy quadrant.
        still_num_units = len(remaining_units)

        # Let's try a grid of targets around the enemy corner. For example,
        # we can create a smaller grid (like rows x cols) near enemy corner.
        # The size of this grid might depend on how many units we have.
        rows = int(np.floor(np.sqrt(still_num_units)))
        cols = int(np.ceil(still_num_units / rows))
        while rows * cols < still_num_units:
            cols += 1

        # Define how large the "attack staging area" is. For instance, a 1/3 portion of the map
        # closer to the enemy corner could be chosen as the area we distribute targets in.
        # This is arbitrary and can be tuned. For example:
        # If enemy is bottom-right, we'll pick a sub-area in the bottom-right quadrant.
        # If enemy is top-left, do the opposite. We'll just center around enemy_corner_x, enemy_corner_y.
        # Let's say we form a grid around that corner within some offset.
        offset_x = max(map_width // 4, 1)  # a quarter of the map width
        offset_y = max(map_height // 4, 1)  # a quarter of the map height

        # Based on where the enemy corner is, define a bounding rectangle for targets.
        # If enemy is bottom-right corner:
        min_x = max(0, enemy_corner_x - offset_x)
        max_x = min(map_width - 1, enemy_corner_x)
        min_y = max(0, enemy_corner_y - offset_y)
        max_y = min(map_height - 1, enemy_corner_y)

        # If enemy is top-left corner:
        # (If self.team_id == 1, we already set enemy_corner_x,y = 0,0)
        # The above min_x,max_x,min_y,max_y should still work, just reversed.
        # For example, if enemy_corner_x = 0, then max_x might be offset_x and so forth.
        if enemy_corner_x < map_width // 2:
            # Enemy is on the left side, so let's spread on the left.
            max_x = min(map_width - 1, enemy_corner_x + offset_x)
            min_x = max(0, enemy_corner_x)
        if enemy_corner_y < map_height // 2:
            # Enemy is on the top side, so let's spread on the top.
            max_y = min(map_height - 1, enemy_corner_y + offset_y)
            min_y = max(0, enemy_corner_y)

        # Compute cell width/height for the grid in the defined bounding rectangle
        region_width = max_x - min_x + 1
        region_height = max_y - min_y + 1
        cell_width = region_width / cols if cols > 0 else region_width
        cell_height = region_height / rows if rows > 0 else region_height

        assigned_cell_count = min(still_num_units, rows * cols)
        targets = []
        for i in range(assigned_cell_count):
            r = i // cols
            c = i % cols
            cell_center_x = int(min_x + (c + 0.5) * cell_width)
            cell_center_y = int(min_y + (r + 0.5) * cell_height)
            cell_center_x = min(cell_center_x, map_width - 1)
            cell_center_y = min(cell_center_y, map_height - 1)
            targets.append((cell_center_x, cell_center_y))

        num_remaining = still_num_units
        used_cell_count = min(num_remaining, assigned_cell_count)
        if used_cell_count == 0:
            # No targets assigned, stay put
            for u in remaining_units:
                actions[u] = [0, 0, 0]
            return

        # Build a cost matrix for Hungarian assignment
        cost_matrix = np.zeros((num_remaining, used_cell_count), dtype=int)
        for i, unit_id in enumerate(remaining_units):
            ux, uy = unit_positions[unit_id]
            for j in range(used_cell_count):
                tx, ty = targets[j]
                dist = abs(ux - tx) + abs(uy - ty)
                cost = dist
                # If desired, adjust cost based on known tiles:
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

        # Assign moves towards targets
        for unit_id, (tx, ty) in unit_to_target.items():
            ux, uy = unit_positions[unit_id]
            if ux == tx and uy == ty:
                # Already at target, hold position (or consider a "sap" or another action)
                actions[unit_id] = [0, 0, 0]
            else:
                direction = self.get_direction_via_pathfinding((ux, uy), (tx, ty), obs)
                actions[unit_id] = [direction, 0, 0]

        # Any unassigned units can idle
        for unit_id in unassigned_units:
            actions[unit_id] = [0, 0, 0]

    def send_to_explore_if_not_going_to_relic(self, NON_REWARD_PENALTY, REWARD_BONUS, actions, map_height, map_width,
                                              obs, remaining_units, unit_positions):
        if len(remaining_units) > 0:
            still_num_units = len(remaining_units)
            rows = int(np.floor(np.sqrt(still_num_units)))
            cols = int(np.ceil(still_num_units / rows))
            while rows * cols < still_num_units:
                cols += 1

            cell_width = self.env_cfg["map_width"] / cols
            cell_height = self.env_cfg["map_height"] / rows
            assigned_cell_count = min(still_num_units, rows * cols)
            targets = []
            for i in range(assigned_cell_count):
                r = i // cols
                c = i % cols
                cell_center_x = int((c + 0.5) * cell_width)
                cell_center_y = int((r + 0.5) * cell_height)
                cell_center_x = min(cell_center_x, map_width - 1)
                cell_center_y = min(cell_center_y, map_height - 1)
                targets.append((cell_center_x, cell_center_y))

            # NEW LOGIC: Remove occupied positions from general targets as well
            # targets = [t for t in targets if t not in occupied_positions]
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

    def send_to_relic_points(self, actions, map_height, map_width, obs, remaining_units, unit_positions):
        # NEW LOGIC: Identify units already on a reward tile.
        already_on_reward_units = []
        occupied_positions = set()
        for u in remaining_units:
            ux, uy = unit_positions[u]
            if (ux, uy) in self.known_reward_tiles:
                # If the unit is on a reward tile, do not move it.
                actions[u] = [0, 0, 0]
                already_on_reward_units.append(u)
                occupied_positions.add((ux, uy))
        # NEW LOGIC: If a unit is already on a reward tile, let it stay there.
        already_on_reward_units = []
        for u in remaining_units:
            ux, uy = unit_positions[u]
            if (ux, uy) in self.known_reward_tiles:
                # If the unit is on a reward tile, do not move it.
                actions[u] = [0, 0, 0]
                already_on_reward_units.append(u)
        # Remove these units from the remaining pool so they are not reassigned.
        # remaining_units = [u for u in remaining_units if u not in already_on_reward_units]
        relic_targets = list(self.known_relic_positions)
        if len(self.known_relic_positions) > 0:
            block_radius = 2
            for (rx, ry) in self.known_relic_positions:
                for bx in range(rx - block_radius, rx + block_radius + 1):
                    for by in range(ry - block_radius, ry + block_radius + 1):
                        if 0 <= bx < map_width and 0 <= by < map_height:
                            relic_targets.append((bx, by))
            relic_targets = list(set(relic_targets))
        relic_targets = list(set(list(set(relic_targets) - set(self.not_reward_tiles)) + list(self.known_reward_tiles)))
        # NEW LOGIC: Remove occupied positions (units that are staying put) from relic targets
        # relic_targets = [t for t in relic_targets if t not in occupied_positions]
        # Prioritization constants
        REWARD_BONUS = -100
        POTENTIAL_RELIC_POINTS = -10
        NON_REWARD_PENALTY = 0
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
                        # if cost < 0:
                        #     cost = 0
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
        return NON_REWARD_PENALTY, REWARD_BONUS, remaining_units

    def send_toward_relic_areas(self, actions, available_unit_ids, map_height, map_width, obs, sap_done,
                                unit_positions):
        remaining_units = [u for u in available_unit_ids if u not in sap_done]
        # NEW LOGIC: Identify units already on a reward tile.
        already_on_reward_units = []
        occupied_positions = set()
        for u in remaining_units:
            ux, uy = unit_positions[u]
            if (ux, uy) in self.known_reward_tiles:
                # If the unit is on a reward tile, do not move it.
                actions[u] = [0, 0, 0]
                already_on_reward_units.append(u)
                occupied_positions.add((ux, uy))
        #
        # # NEW LOGIC: If a unit is already on a reward tile, let it stay there.
        already_on_reward_units = []
        for u in remaining_units:
            ux, uy = unit_positions[u]
            if (ux, uy) in self.known_reward_tiles:
                # If the unit is on a reward tile, do not move it.
                actions[u] = [0, 0, 0]
                already_on_reward_units.append(u)
        # Remove these units from the remaining pool so they are not reassigned.
        # remaining_units = [u for u in remaining_units if u not in already_on_reward_units]
        relic_targets = list(self.known_relic_positions)
        if len(self.known_relic_positions) > 0:
            block_radius = 2
            for (rx, ry) in self.known_relic_positions:
                for bx in range(rx - block_radius, rx + block_radius + 1):
                    for by in range(ry - block_radius, ry + block_radius + 1):
                        if 0 <= bx < map_width and 0 <= by < map_height:
                            relic_targets.append((bx, by))
            relic_targets = list(set(relic_targets))
        relic_targets = list(set(list(set(relic_targets) - set(self.not_reward_tiles)) + list(self.known_reward_tiles)))
        # NEW LOGIC: Remove occupied positions (units that are staying put) from relic targets
        # relic_targets = [t for t in relic_targets if t not in occupied_positions]
        # Prioritization constants
        REWARD_BONUS = -50
        POTENTIAL_RELIC_POINTS = -10
        NON_REWARD_PENALTY = 0
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
                        # if cost < 0:
                        #     cost = 0
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
        return NON_REWARD_PENALTY, REWARD_BONUS, remaining_units

    def do_sapping_logic(self, actions, available_unit_ids, enemy_positions, sap_cost, sap_done, sap_range,
                         unit_energy, unit_positions, opp_energy):
        targeted_enemies = set()
        for unit_id in available_unit_ids:
            ux, uy = unit_positions[unit_id]
            uenergy = unit_energy[unit_id]

            # Only consider sapping if we have enough energy
            if uenergy > sap_cost:
                # Search for a visible weaker enemy unit in sap range
                found_target = False
                for dx in range(-sap_range, sap_range + 1):
                    if found_target:
                        break
                    for dy in range(-sap_range, sap_range + 1):
                        tx = ux + dx
                        ty = uy + dy
                        # Check if an enemy occupies this tile and not already targeted
                        if (tx, ty) in enemy_positions and (tx, ty) not in targeted_enemies:
                            # Check enemy units in that tile
                            enemy_ids = enemy_positions[(tx, ty)]
                            # Find any enemy weaker than us
                            for eid in enemy_ids:
                                enemy_energy = opp_energy[eid]
                                if enemy_energy < uenergy:
                                    # We found a weaker enemy unit to sap
                                    # action code 5 = sap, dx and dy are relative moves
                                    actions[unit_id] = [5, dx, dy]
                                    sap_done.add(unit_id)
                                    targeted_enemies.add((tx, ty))
                                    found_target = True
                                    break
                        if found_target:
                            break

    def add_newly_discovered_relics(self, relic_nodes_positions):
        # Add newly discovered relics to known_relic_positions
        for (rx, ry) in relic_nodes_positions:
            if (rx, ry) not in self.relic_tile_data:
                self.relic_tile_data[(rx, ry)] = {}
                for bx in range(rx - 2, rx + 3):
                    for by in range(ry - 2, ry + 3):
                        if 0 <= bx < self.env_cfg["map_width"] and 0 <= by < self.env_cfg["map_height"]:
                            self.relic_tile_data[(rx, ry)][(bx, by)] = {"tested": False, "reward_tile": False}
            # If it's a new relic not seen in previous games, store it
            if (rx, ry) not in self.known_relic_positions:
                self.known_relic_positions.append((rx, ry))

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
        agent_dict[player] = BestAgentAttackerMoreCollisions(player, configurations["env_cfg"])
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
