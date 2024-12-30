#!/usr/bin/env python3
"""
plot_board.py
Usage:
    python plot_board.py game_X.pkl

Description:
    Loads a saved board dictionary (with top-level keys 'player_0' and
    'player_1') and visualizes 2D slices or channels from their observation
    data as heatmaps in matplotlib, with special logic for 'units'
    (positions + energies) that get plotted on a 24x24 grid. We also
    handle 'relic_nodes' in a similar fashion, marking them on a 24x24 grid.
"""

import pickle
import numpy as np
import sys

import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you have PyQt5 installed
import matplotlib.pyplot as plt


# Sub-keys that each player's dict might have.
PLAYER_OBS_KEYS = [
    "units",           # dictionary with 'position' & 'energy'
    "units_mask",
    "sensor_mask",
    "map_features",
    "relic_nodes",     # array of shape (N,2)
    "relic_nodes_mask",
    "team_points",
    "team_wins",
    "steps",
    "match_steps"
]


def transform_units_to_channels(
        units: dict,
        team_idx: int,
        grid_size: int = 24,
        max_units: float = 16.0,
        max_energy: float = 500.0
) -> dict:
    """
    Transform the 'units' observation into 24x24 image-like arrays.
    ...
    (Same as before.)
    """

    friendly_positions = units["position"][team_idx]
    friendly_energies  = units["energy"][team_idx]
    enemy_positions    = units["position"][1 - team_idx]
    enemy_energies     = units["energy"][1 - team_idx]

    friend_units       = np.zeros((grid_size, grid_size), dtype=np.float32)
    friendly_energy    = np.zeros((grid_size, grid_size), dtype=np.float32)
    friendly_num_units = np.zeros((grid_size, grid_size), dtype=np.float32)

    enemy_units        = np.zeros((grid_size, grid_size), dtype=np.float32)
    enemy_energy       = np.zeros((grid_size, grid_size), dtype=np.float32)
    enemy_num_units    = np.zeros((grid_size, grid_size), dtype=np.float32)

    def accumulate_units(positions, energies, grid_presence, grid_energy, grid_count):
        for i in range(len(positions)):
            x, y = positions[i]
            e = energies[i]
            if x < 0 or y < 0 or x >= grid_size or y >= grid_size:
                continue
            if e < 0:
                continue
            grid_presence[y, x] = 1
            grid_energy[y, x]   += e
            grid_count[y, x]    += 1

    # Fill in the grids for friendly units
    accumulate_units(
        friendly_positions,
        friendly_energies,
        friend_units,
        friendly_energy,
        friendly_num_units
    )

    # Fill in the grids for enemy units
    accumulate_units(
        enemy_positions,
        enemy_energies,
        enemy_units,
        enemy_energy,
        enemy_num_units
    )

    # Normalize energies and counts
    friendly_energy    = np.clip(friendly_energy / max_energy, 0.0, 1.0)
    friendly_num_units = np.clip(friendly_num_units / max_units, 0.0, 1.0)

    enemy_energy       = np.clip(enemy_energy / max_energy, 0.0, 1.0)
    enemy_num_units    = np.clip(enemy_num_units / max_units, 0.0, 1.0)

    return {
        "enemy_units":        enemy_units,
        "enemy_energy":       enemy_energy,
        "enemy_num_units":    enemy_num_units,
        "friendly_units":     friend_units,
        "friendly_energy":    friendly_energy,
        "friendly_num_units": friendly_num_units
    }


def transform_relic_nodes_to_grid(relic_nodes: np.ndarray, grid_size: int = 24) -> np.ndarray:
    """
    Convert an (N,2) array of [x,y] relic positions into a 24x24 grid of 0 or 1.
    If a position is [-1, -1], it's invisible or doesn't exist, so skip it.
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for (x, y) in relic_nodes:
        if x < 0 or y < 0 or x >= grid_size or y >= grid_size:
            # Means we can't see it or it's out of bounds
            continue
        grid[y, x] = 1.0
    return grid


def _plot_array(arr, title_prefix):
    """
    Fallback array-plotting logic for normal multi-dimensional arrays
    (2D up to 5D). This is used for keys other than 'units' or specifically
    processed keys like 'relic_nodes'.
    """
    ndim = arr.ndim


    if ndim == 2:
        data_2d = arr
        plt.figure()
        plt.imshow(data_2d, cmap="viridis", aspect="auto", origin="upper")
        plt.colorbar()
        plt.title(f"{title_prefix} (2D)")

    elif ndim == 3:
        # For generic 3D, let's just pick arr[0] to visualize
        data_2d = arr[0]
        plt.figure()
        plt.imshow(data_2d, cmap="viridis", aspect="auto", origin="upper")
        plt.colorbar()
        plt.title(f"{title_prefix} (3D, showing arr[0])")

    elif ndim == 4:
        # e.g. (B, C, H, W) – pick arr[0,0]
        data_2d = arr[0, 0]
        plt.figure()
        plt.imshow(data_2d, cmap="viridis", aspect="auto", origin="upper")
        plt.colorbar()
        plt.title(f"{title_prefix} (4D, showing arr[0,0])")

    elif ndim == 5:
        # e.g. (B, C, H, W, Channels=?)
        B, C, H, W, Channels = arr.shape
        for ch in range(Channels):
            data_2d = arr[0, 0, :, :, ch]
            plt.figure()
            plt.imshow(data_2d, cmap="viridis", aspect="auto", origin="upper")
            plt.colorbar()
            plt.title(f"{title_prefix} (5D, channel={ch})")

    else:
        print(f"Skipping {title_prefix} with shape {arr.shape}, "
              "as no plotting logic is defined for >5D or <2D.")


def _walk_player_obs(obs_dict, player_name, player_index):
    """
    Given a dictionary of observations for a single player (e.g. board_dict['player_0']),
    walk through known keys and plot any multi-dimensional NumPy arrays,
    with special logic for 'units' (position + energy) and 'relic_nodes'.
    """
    is_player_zero = player_index == 0
    for key in PLAYER_OBS_KEYS:
        if key not in obs_dict:
            continue  # skip if it doesn't exist

        value = obs_dict[key]
        full_path = f"{player_name}.{key}"

        # 1) Special case: "units" => transform to channels and plot
        if key == "units" and isinstance(value, dict):
            position = value.get("position")
            energy   = value.get("energy")
            if (isinstance(position, np.ndarray) and position.ndim == 3 and
                    isinstance(energy, np.ndarray) and energy.ndim == 2):
                print(f"Plotting special 'units' at {full_path}")
                unit_values_dict = transform_units_to_channels(value, player_index)

                # Plot each channel
                _plot_array(unit_values_dict["friendly_units"],
                            title_prefix=f"{full_path}.friendly_units")
                _plot_array(unit_values_dict["friendly_energy"],
                            title_prefix=f"{full_path}.friendly_energy")
                _plot_array(unit_values_dict["friendly_num_units"],
                            title_prefix=f"{full_path}.friendly_num_units")
                _plot_array(unit_values_dict["enemy_units"],
                            title_prefix=f"{full_path}.enemy_units")
                _plot_array(unit_values_dict["enemy_energy"],
                            title_prefix=f"{full_path}.enemy_energy")
                _plot_array(unit_values_dict["enemy_num_units"],
                            title_prefix=f"{full_path}.enemy_num_units")
            else:
                # Fallback: if we don't have the expected shapes
                for subkey, subval in value.items():
                    subpath = f"{full_path}.{subkey}"
                    if isinstance(subval, np.ndarray) and subval.ndim >= 2:
                        print(f"Plotting array at {subpath} with shape {subval.shape}")
                        _plot_array(subval, title_prefix=subpath)

        # 2) Special case: "relic_nodes" => transform into 24x24 grid
        elif key == "relic_nodes" and isinstance(value, np.ndarray):
            # We expect shape (N,2)
            if value.ndim == 2 and value.shape[1] == 2:
                print(f"Plotting 'relic_nodes' at {full_path}")
                relic_grid = transform_relic_nodes_to_grid(value, grid_size=24)
                # Then we can simply use _plot_array on that 2D grid
                _plot_array(relic_grid, title_prefix=f"{full_path}.relic_nodes_grid")
            else:
                print(f"'relic_nodes' has unexpected shape {value.shape}, skipping.")

        elif key == "sensor_mask" and isinstance(value, np.ndarray):
            # We expect shape (N,2)
            print(f"Plotting array at {full_path} with shape {value.shape}")
            _plot_array(value.transpose(), title_prefix=full_path)

        elif key == "map_features":
            for subkey, subval in value.items():
                subpath = f"{full_path}.{subkey}"
                if isinstance(subval, np.ndarray) and subval.ndim >= 2:
                    print(f"Plotting array at {subpath} with shape {subval.shape}")
                    _plot_array(subval.transpose(), title_prefix=subpath)

        # 4) Generic multi-dimensional NumPy arrays
        elif key == "units_mask" and isinstance(value, np.ndarray) and value.ndim >= 2:
            print(f"Plotting array at {full_path} with shape {value.shape}")
            _plot_array(value.transpose(), title_prefix=full_path)


        # 3) If the value is a dict, potentially recurse deeper
        elif isinstance(value, dict):
            for subkey, subval in value.items():
                subpath = f"{full_path}.{subkey}"
                if isinstance(subval, np.ndarray) and subval.ndim >= 2:
                    print(f"Plotting array at {subpath} with shape {subval.shape}")
                    _plot_array(subval, title_prefix=subpath)

        # 4) Generic multi-dimensional NumPy arrays
        elif isinstance(value, np.ndarray) and value.ndim >= 2:
            print(f"Plotting array at {full_path} with shape {value.shape}")
            _plot_array(value, title_prefix=full_path)

        else:
            # If it's not a multi-dim array, or not a dict, skip
            pass


def visualize_board_dict(board_dict):
    """
    Specifically handle the top-level keys 'player_0' and 'player_1'.
    """
    for player_name in ["player_0", "player_1"]:
        if player_name not in board_dict:
            print(f"Warning: {player_name} not found in the board dict.")
            continue
        print(f"Visualizing data for {player_name}")
        player_index = 0 if player_name == "player_0" else 1
        _walk_player_obs(board_dict[player_name], player_name, player_index)

    # Finally, show all plots
    plt.show()


def load_and_plot(filename):
    """
    Load board dictionary from 'filename' and call visualize_board_dict.
    """
    with open(filename, "rb") as f:
        board_dict = pickle.load(f)

    print(f"Loaded board dict from {filename}")
    visualize_board_dict(board_dict)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_board.py <path_to_board_file.pkl>")
        sys.exit(1)
    board_file = sys.argv[1]
    load_and_plot(board_file)
