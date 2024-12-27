from lux_ai.lux.constants import Constants

ACTION_MEANINGS = {
    "unit": [
        "NO-OP",
    ],
}

# Extract a tuple of possible directions (like N, E, S, W) from Constants.
DIRECTIONS = Constants.DIRECTIONS.astuple(include_center=False)

# Add move directions (e.g., MOVE_N, MOVE_S, etc.) and transfer actions (TRANSFER_resource_direction)
# to both "worker" and "cart" sets of actions.
for u in ["unit"]:
    for d in DIRECTIONS:
        ACTION_MEANINGS[u].append(f"MOVE_{d}")


ACTION_MEANINGS["unit"].extend(["SAP"])