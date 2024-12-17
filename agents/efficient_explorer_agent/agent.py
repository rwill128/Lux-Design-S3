from lux.utils import direction_to
import sys
import numpy as np

class Role:
    """Base class for a role, defining a default action method."""
    def __init__(self, agent, unit_id, unit_pos, step):
        self.agent = agent
        self.unit_id = unit_id
        self.unit_pos = unit_pos
        self.step = step

    def decide_action(self):
        """Override this method in specific roles."""
        raise NotImplementedError

class EfficientExplorer(Role):
    """Exploration role focusing on energy, vision, and avoiding revisited tiles."""
    def decide_action(self):
        # Initialize exploration map if not done yet
        if "exploration_map" not in self.agent.__dict__:
            self.agent.exploration_map = np.zeros((self.agent.env_cfg["map_height"], self.agent.env_cfg["map_width"]))

        # Update exploration map based on current position
        self.agent.exploration_map[self.unit_pos[1], self.unit_pos[0]] += 1  # Mark tile as visited

        # Find the best neighboring tile to move to
        best_move = None
        best_score = -np.inf

        # Define possible moves (directions)
        directions = [
            (0, 0),  # Do nothing
            (0, -1),  # Up
            (1, 0),   # Right
            (0, 1),   # Down
            (-1, 0),  # Left
        ]

        for i, (dx, dy) in enumerate(directions):
            new_x = self.unit_pos[0] + dx
            new_y = self.unit_pos[1] + dy

            # Ensure move is within the map boundaries
            if 0 <= new_x < self.agent.env_cfg["map_width"] and 0 <= new_y < self.agent.env_cfg["map_height"]:
                # Tile scores based on energy, novelty, and visibility
                energy = self.agent.obs['map_features']['energy'][new_y][new_x]
                visits = self.agent.exploration_map[new_y, new_x]
                score = energy - 2 * visits  # Higher energy, fewer visits = better

                if score > best_score:
                    best_score = score
                    best_move = i

        return [best_move, 0, 0]  # Return the best move decision

class RelicPointGainer(Role):
    """Role to hover around relic nodes to gain points."""
    def decide_action(self):
        nearest_relic_node = self.agent.relic_node_positions[0]
        manhattan_distance = abs(self.unit_pos[0] - nearest_relic_node[0]) + abs(self.unit_pos[1] - nearest_relic_node[1])
        if manhattan_distance <= 4:
            return [np.random.randint(0, 5), 0, 0]  # Random move near relic
        return [direction_to(self.unit_pos, nearest_relic_node), 0, 0]  # Move towards relic

class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.team_id = 0 if player == "player_0" else 1
        self.env_cfg = env_cfg
        np.random.seed(0)

        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        observed_relic_nodes = np.array(obs["relic_nodes"])
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])

        available_units = np.where(unit_mask)[0]
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # Save newly discovered relic nodes
        for node_id in visible_relic_node_ids:
            if node_id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(node_id)
                self.relic_node_positions.append(observed_relic_nodes[node_id])

        # Assign roles to units
        for unit_id in available_units:
            unit_pos = unit_positions[unit_id]
            role = (
                RelicPointGainer(self, unit_id, unit_pos, step)
                if self.relic_node_positions
                else EfficientExplorer(self, unit_id, unit_pos, step)
            )
            actions[unit_id] = role.decide_action()


        return actions
