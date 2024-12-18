
# You can install scipy if not already available:
# !pip install scipy
from scipy.optimize import linear_sum_assignment

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