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

class RandomMover(Role):
    """Random exploration role for units."""
    def decide_action(self):
        if self.step % 20 == 0 or self.unit_id not in self.agent.unit_explore_locations:
            rand_loc = (
                np.random.randint(0, self.agent.env_cfg["map_width"]),
                np.random.randint(0, self.agent.env_cfg["map_height"]),
            )
            self.agent.unit_explore_locations[self.unit_id] = rand_loc
        return [direction_to(self.unit_pos, self.agent.unit_explore_locations[self.unit_id]), 0, 0]

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
        """
        :param relic_hunters: Number of bots to become RelicPointGainers when relics are found.
        """
        self.player = player
        self.team_id = 0 if player == "player_0" else 1
        self.env_cfg = env_cfg
        self.relic_hunters = 10  # Parameterized number of relic hunters
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
        relic_hunters_assigned = 0
        for unit_id in available_units:
            unit_pos = unit_positions[unit_id]
            if self.relic_node_positions and relic_hunters_assigned < self.relic_hunters:
                role = RelicPointGainer(self, unit_id, unit_pos, step)
                relic_hunters_assigned += 1
            else:
                role = RandomMover(self, unit_id, unit_pos, step)
            actions[unit_id] = role.decide_action()

        return actions
