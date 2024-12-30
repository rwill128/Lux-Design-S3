from typing import Dict, List, Optional

import numpy as np

from .game_map import Position
from .game_constants import GAME_CONSTANTS


class Player:
    def __init__(self, team):
        self.team = team
        self.relic_points = 0
        self.units: List[Unit] = []

    def get_relic_points(self) -> int:
        return self.relic_points

    @property
    def total_energy(self) -> int:
        return sum([un.energy for un in self.units])

    def get_unit_by_id(self, unit_id: str) -> Optional['Unit']:
        for unit in self.units:
            if unit_id == unit.id:
                return unit
        return None


class RelicNode:
    def __init__(self, x, y):
        self.pos = Position(x, y)

    def __str__(self):
        return self.pos

    def __repr__(self):
        return f"Relic node: {str(self)}"

class Unit:
    def __init__(self, teamid, unitid, x, y, energy):
        self.pos = Position(x, y)
        self.team = teamid
        self.id = unitid
        self.energy = energy

    def can_sap(self) -> bool:
        return self.energy > GAME_CONSTANTS.UNIT_SAP_COST

    def get_energy(self) -> bool:
        return self.energy

    def move(self, dir) -> np.array:
        """
        return the command to move unit in the given direction
        """
        return [int(dir), 0, 0]

    # TODO: Maybe this is where I return an array with coordinates for the closest enemy?
    #   at least a first draft
    def sap(self, target_unit) -> np.array:
        """
        return the command to pillage whatever is underneath the worker
        """
        return [5, 2, 2]


    def __str__(self):
        return f"Unit: team_{self.team}/energy_{self.energy}/pos_{self.pos}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Unit) and self.id == other.id
