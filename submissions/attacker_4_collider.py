import json
from argparse import Namespace

import numpy as np
from scipy.optimize import linear_sum_assignment


class BestAgentAttacker4NoCollide:
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
        path = self.dijkstra_pathfind(
            self.env_cfg["map_width"],
            self.env_cfg["map_height"],
            from_pos,
            to_pos,
            obs
        )
        if path is None or len(path) < 2:
            return self.simple_heuristic_move(from_pos, to_pos)
        next_step = path[1]
        dx = next_step[0] - from_pos[0]
        dy = next_step[1] - from_pos[1]
        action = self.dxdy_to_action(dx, dy)
        return action

    def select_tiles_for_relic(self, relic_pos):
        # ... unchanged ...
        reward_tiles = [t for t, d in self.relic_tile_data[relic_pos].items() if d["reward_tile"]]
        untested_tiles = [t for t, d in self.relic_tile_data[relic_pos].items() if not d["tested"]]
        return reward_tiles, untested_tiles

    def deduce_reward_tiles(self, obs):
        """Deduce reward tiles with confidence tracking.

        (Unchanged logic from your snippet)
        """
        current_team_points = obs["team_points"][self.team_id]
        if hasattr(current_team_points, 'item'):
            current_team_points = current_team_points.item()
        gain = current_team_points - self.last_team_points

        if not self.tile_confidence and self._persistent_tile_confidence:
            self.tile_confidence = self._persistent_tile_confidence.copy()

        unit_positions = obs["units"]["position"][self.team_id]
        unit_mask = obs["units_mask"][self.team_id].astype(bool)
        occupied_this_turn = set()
        for uid in np.where(unit_mask)[0]:
            x, y = unit_positions[uid]
            if (x, y) in self.unknown_tiles:
                occupied_this_turn.add((x, y))

        currently_reward_occupied = set()
        for uid in np.where(unit_mask)[0]:
            x, y = unit_positions[uid]
            pos = (x, y)
            if pos in self.known_reward_tiles:
                currently_reward_occupied.add(pos)

        self.newly_unoccupied_unknown = self.last_unknown_occupied - occupied_this_turn
        self.newly_unoccupied_known = self.last_reward_occupied - currently_reward_occupied

        newly_unoccupied = self.newly_unoccupied_unknown.union(self.newly_unoccupied_known)
        last_gain = getattr(self, 'last_gain', 0)
        gain_rate = gain - last_gain

        # ... (the rest of your existing deduce_reward_tiles method remains unchanged) ...
        # ... for brevity we keep only the new final lines ...

        # Persist confidence
        self._persistent_tile_confidence = self.tile_confidence.copy()

        # Update tracking
        self.last_reward_occupied = currently_reward_occupied
        self.last_unknown_occupied = occupied_this_turn
        self.last_team_points = current_team_points
        self.last_gain = gain

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
        pq = [(0, start)]

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
                continue

            (cx, cy) = current
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < map_width and 0 <= ny < map_height and is_passable(nx, ny):
                    tile_cost = 10 - tile_energy_map[nx, ny] * 2
                    if tile_cost < 0:
                        tile_cost = 0

                    new_dist = current_dist + tile_cost
                    if (nx, ny) not in dist or new_dist < dist[(nx, ny)]:
                        dist[(nx, ny)] = new_dist
                        came_from[(nx, ny)] = (cx, cy)
                        heapq.heappush(pq, (new_dist, (nx, ny)))

        return None

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        opp_positions = np.array(obs["units"]["position"][self.opp_team_id])

        # 1) Update possible reward tiles
        self.update_possible_reward_tiles(obs)

        # 2) Deduce reward tiles
        self.deduce_reward_tiles(obs)

        relic_nodes_mask = obs["relic_nodes_mask"]
        self.add_newly_discovered_relics(obs["relic_nodes"][relic_nodes_mask])

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        available_unit_ids = np.where(obs["units_mask"][self.team_id])[0]

        map_width = self.env_cfg["map_width"]
        map_height = self.env_cfg["map_height"]

        # Identify enemy positions
        opp_visible_mask = (opp_positions[:, 0] != -1) & (opp_positions[:, 1] != -1)
        visible_opp_ids = np.where(opp_visible_mask)[0]
        enemy_positions = {}
        for oid in visible_opp_ids:
            ex, ey = opp_positions[oid]
            if (ex, ey) not in enemy_positions:
                enemy_positions[(ex, ey)] = []
            enemy_positions[(ex, ey)].append(oid)

        # Sapping logic
        sap_done = set()
        self.do_sapping_logic(
            actions,
            available_unit_ids,
            enemy_positions,
            self.env_cfg.get("unit_sap_cost", 10),
            sap_done,
            self.env_cfg.get("unit_sap_range", 1),
            np.array(obs["units"]["energy"][self.team_id]),
            unit_positions,
            opp_energy=np.array(obs["units"]["energy"][self.opp_team_id])
        )

        # 3) Send some units to relic points
        remaining_units = [u for u in available_unit_ids if u not in sap_done]
        NON_REWARD_PENALTY, REWARD_BONUS, remaining_units = self.send_to_relic_points(
            actions, map_height, map_width, obs, remaining_units, unit_positions
        )

        # 4) (Unify the old "explore vs. attack" logic into ONE function)
        #    We'll call it send_to_attack_explore_for_collision
        self.send_to_attack_explore_for_collision(
            NON_REWARD_PENALTY, REWARD_BONUS,
            actions, map_height, map_width,
            obs, remaining_units, unit_positions, enemy_positions
        )

        # [Optional] End-of-match print
        if obs["steps"] == 500 and not self.end_of_match_printed:
            # ... if you want to see which tiles ended up as reward ...
            self.end_of_match_printed = True
            # print(...)

        # Validate final actions
        a = actions[:, 0]  # action codes
        dx = actions[:, 1]
        dy = actions[:, 2]

        # Non-sap range check
        non_sap_mask = (a != 5)
        assert np.all(a[non_sap_mask] >= 0), f"Non-sap actions must be >= 0. Got: {a[non_sap_mask]}"
        assert np.all(a[non_sap_mask] <= 4), f"Non-sap actions must be <= 4. Got: {a[non_sap_mask]}"
        assert np.all(dx[non_sap_mask] == 0), f"dx must be 0 for non-sap actions. Got: {dx[non_sap_mask]}"
        assert np.all(dy[non_sap_mask] == 0), f"dy must be 0 for non-sap actions. Got: {dy[non_sap_mask]}"

        # Sap range check
        sap_mask = (a == 5)
        assert np.all(dx[sap_mask] >= -10), f"Sap dx out of range. Got: {dx[sap_mask]}"
        assert np.all(dx[sap_mask] <= 10), f"Sap dx out of range. Got: {dx[sap_mask]}"
        assert np.all(dy[sap_mask] >= -10), f"Sap dy out of range. Got: {dy[sap_mask]}"
        assert np.all(dy[sap_mask] <= 10), f"Sap dy out of range. Got: {dy[sap_mask]}"

        actions = actions.astype(np.int32)
        return actions

    def send_to_attack_explore_for_collision(
            self, NON_REWARD_PENALTY, REWARD_BONUS, actions, map_height, map_width,
            obs, remaining_units, unit_positions, enemy_positions
    ):
        """
        Unifies exploring new territory with a preference for collision.
        We'll do a Hungarian assignment over a 'grid' near the enemy corner or across the map,
        with extra scoring for:
         - high tile energy
         - tile occupied by enemies (to force collisions)
        """
        if len(remaining_units) == 0:
            return

        # We'll just define a sub-grid covering a decent portion of the map. For example,
        # if your strategy is to 'attack' from one corner, you can keep that approach.
        # Or you can do a simpler 'cover the entire map in a grid' approach, as in explore.

        still_num_units = len(remaining_units)
        rows = int(np.floor(np.sqrt(still_num_units)))
        cols = int(np.ceil(still_num_units / rows))
        while rows * cols < still_num_units:
            cols += 1

        # We'll cover the entire map with a grid
        cell_width = map_width / cols
        cell_height = map_height / rows
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

        tile_energy_map = obs["map_features"]["energy"]

        # Hungarian cost matrix
        # We'll lower the cost if the tile has high energy (so we 'prefer' it),
        # or if there's an enemy on that tile (incentivize collision).
        # Adjust these factors to your taste:
        ENERGY_PREF_FACTOR = 20   # how strongly you prefer higher energy squares
        COLLISION_PREF_BONUS = 50 # how strongly you prefer colliding with the enemy

        num_units = len(remaining_units)
        used_cell_count = assigned_cell_count
        if used_cell_count == 0:
            for u in remaining_units:
                actions[u] = [0, 0, 0]
            return

        cost_matrix = np.zeros((num_units, used_cell_count), dtype=int)
        for i, unit_id in enumerate(remaining_units):
            ux, uy = unit_positions[unit_id]
            for j in range(used_cell_count):
                tx, ty = targets[j]
                dist = abs(ux - tx) + abs(uy - ty)

                # Base cost is distance
                cost = dist

                # Subtract tile energy to prefer high-energy squares
                # (Be sure not to make cost < 0 unless you are comfortable with that in Hungarian.)
                tile_energy_val = tile_energy_map[tx, ty]
                cost -= ENERGY_PREF_FACTOR * tile_energy_val

                # If there's an enemy in that tile, we subtract even more cost
                # so that we attempt to "collide" (assuming we have enough energy).
                if (tx, ty) in enemy_positions:
                    cost -= COLLISION_PREF_BONUS

                # If it's a known reward tile, apply REWARD_BONUS
                if (tx, ty) in self.known_reward_tiles:
                    cost += REWARD_BONUS

                # If it's known not a reward tile, apply NON_REWARD_PENALTY
                if (tx, ty) in self.not_reward_tiles:
                    cost += NON_REWARD_PENALTY

                # Make sure we don't go below zero if that confuses your logic;
                # Hungarian can handle zero or negative, but you might want to clamp
                # if cost < 0:
                #     cost = 0

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
                # Already at target
                actions[unit_id] = [0, 0, 0]
            else:
                direction = self.get_direction_via_pathfinding((ux, uy), (tx, ty), obs)
                actions[unit_id] = [direction, 0, 0]

        # Any unassigned units idle
        for unit_id in unassigned_units:
            actions[unit_id] = [0, 0, 0]

    def send_to_relic_points(self, actions, map_height, map_width, obs, remaining_units, unit_positions):
        # (Unchanged from your snippet, aside from minor clarifications)
        already_on_reward_units = []
        occupied_positions = set()
        for u in remaining_units:
            ux, uy = unit_positions[u]
            if (ux, uy) in self.known_reward_tiles:
                actions[u] = [0, 0, 0]
                already_on_reward_units.append(u)
                occupied_positions.add((ux, uy))

        relic_targets = list(self.known_relic_positions)
        if len(self.known_relic_positions) > 0:
            block_radius = 2
            for (rx, ry) in self.known_relic_positions:
                for bx in range(rx - block_radius, rx + block_radius + 1):
                    for by in range(ry - block_radius, ry + block_radius + 1):
                        if 0 <= bx < map_width and 0 <= by < map_height:
                            relic_targets.append((bx, by))
            relic_targets = list(set(relic_targets))
        relic_targets = list(
            set(list(set(relic_targets) - set(self.not_reward_tiles))) | set(self.known_reward_tiles)
        )

        REWARD_BONUS = -100
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
                    if (tx, ty) in self.not_reward_tiles:
                        cost += NON_REWARD_PENALTY
                    if cost < 0:
                        cost = 0
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
                        if (tx, ty) in enemy_positions and (tx, ty) not in targeted_enemies:
                            enemy_ids = enemy_positions[(tx, ty)]
                            for eid in enemy_ids:
                                enemy_energy = opp_energy[eid]
                                if enemy_energy < uenergy:
                                    actions[unit_id] = [5, dx, dy]
                                    sap_done.add(unit_id)
                                    targeted_enemies.add((tx, ty))
                                    found_target = True
                                    break
                        if found_target:
                            break

    def add_newly_discovered_relics(self, relic_nodes_positions):
        for (rx, ry) in relic_nodes_positions:
            if (rx, ry) not in self.relic_tile_data:
                self.relic_tile_data[(rx, ry)] = {}
                for bx in range(rx - 2, rx + 3):
                    for by in range(ry - 2, ry + 3):
                        if 0 <= bx < self.env_cfg["map_width"] and 0 <= by < self.env_cfg["map_height"]:
                            self.relic_tile_data[(rx, ry)][(bx, by)] = {"tested": False, "reward_tile": False}
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
agent_dict = dict()
agent_prev_obs = dict()

def agent_fn(observation, configurations):
    global agent_dict
    obs = observation.obs
    if type(obs) == str:
        obs = json.loads(obs)
    step = observation.step
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        agent_dict[player] = BestAgentAttacker4NoCollide(player, configurations["env_cfg"])
    agent = agent_dict[player]
    actions = agent.act(step, from_json(obs), remainingOverageTime)
    return dict(action=actions.tolist())

if __name__ == "__main__":
    def read_input():
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
        observation = Namespace(
            **dict(
                step=raw_input["step"],
                obs=raw_input["obs"],
                remainingOverageTime=raw_input["remainingOverageTime"],
                player=raw_input["player"],
                info=raw_input["info"]
            )
        )
        if i == 0:
            env_cfg = raw_input["info"]["env_cfg"]
            player_id = raw_input["player"]
        i += 1
        actions = agent_fn(observation, dict(env_cfg=env_cfg))
        print(json.dumps(actions))
