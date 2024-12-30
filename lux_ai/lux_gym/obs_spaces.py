import functools
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from lux_ai.lux.game import Game
from lux_ai.lux.game_constants import GAME_CONSTANTS
from lux_ai.utility_constants import MAX_BOARD_SIZE
import gym

class BaseObsSpace(ABC):
    # NB: Avoid using Discrete() space, as it returns a shape of ()
    # NB: "_COUNT" keys indicate that the value is used to scale the embedding of another value
    @abstractmethod
    def get_obs_spec(
            self,
            board_dims: Tuple[int, int] = MAX_BOARD_SIZE
    ) -> gym.spaces.Dict:
        """
        Get the observation space specification.

        Args:
            board_dims: Tuple of (width, height) for the game board
                      Defaults to MAX_BOARD_SIZE for maximum compatibility

        Returns:
            A gym.spaces.Dict specifying the structure of observations
        """
        pass

    @abstractmethod
    def wrap_env(self, env) -> gym.Wrapper:
        """
        Create a wrapper that converts game states to observation tensors.

        Args:
            env: The Lux environment to wrap

        Returns:
            A gym.Wrapper that implements the observation conversion
        """
        pass



class FixedShapeObs(BaseObsSpace, ABC):
    """
    Abstract base class for observation spaces with fixed tensor shapes.

    This class serves as a marker for observation spaces that maintain consistent
    tensor dimensions regardless of the game state. This is important for:

    1. Neural Network Compatibility:
       - Ensures inputs have consistent shapes for network layers
       - Allows batch processing of observations

    2. Performance:
       - Avoids dynamic tensor allocation
       - Enables efficient GPU utilization

    All fixed shape implementations should:
    - Pre-allocate tensors of maximum size
    - Use padding/masking for variable content
    - Maintain consistent dimensions across episodes
    """
    pass

MAX_ENERGY = 500
P = 2

class FixedShapeContinuousObsV2(FixedShapeObs):
    """
    Enhanced version of FixedShapeContinuousObs with additional spatial features.

    Key Improvements over V1:
    1. Distance Features:
       - Adds distance-from-center metrics for both X and Y axes
       - Helps model understand spatial relationships and board positioning

    2. Board Size Information:
       - Explicitly encodes the current map dimensions
       - Enables better generalization across different board sizes

    3. Normalized Features:
       - All continuous values scaled to [0,1] range
       - Consistent with V1 but adds new normalized spatial features

    The observation space maintains backward compatibility while adding
    new features that help the model understand board geometry better.
    """
    def get_obs_spec(
            self,
            board_dims: Tuple[int, int] = MAX_BOARD_SIZE
    ) -> gym.spaces.Dict:
        x = board_dims[0]
        y = board_dims[1]
        return gym.spaces.Dict({
            # Player specific observations
            # none, worker
            "friendly_units": gym.spaces.MultiBinary((1, P, x, y)),
            # Number of units in the square (only relevant on city tiles)
            "friendly_energy": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # Normalized from 0-500
            "friendly_num_units": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # none, worker
            "enemy_units": gym.spaces.MultiBinary((1, P, x, y)),
            # Number of units in the square (only relevant on city tiles)
            "enemy_energy": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # Normalized from 0-500
            "enemy_num_units": gym.spaces.Box(0., 1., shape=(1, P, x, y)),

            "sensor_mask": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "energy_grid": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "nebula_grid": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "asteroid_grid": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "relic_grid": gym.spaces.Box(0., 1., shape=(1, P, x, y)),

            "dist_from_center_x": gym.spaces.Box(0., 1., shape=(1, 1, x, y)),
            "dist_from_center_y": gym.spaces.Box(0., 1., shape=(1, 1, x, y)),

            # Non-spatial observations
            "team_points": gym.spaces.Box(0., 1., shape=(1, P)),
            "team_wins": gym.spaces.Box(0., 1., shape=(1, P)),
            "team_energy": gym.spaces.Box(0., 1., shape=(1, P)),
            "match": gym.spaces.MultiDiscrete(
                np.zeros((1, 1)) + 500 / 100
            ),
            # The turn number, normalized from 0-360
            "turn": gym.spaces.Box(0., 1., shape=(1, 1)),
        })

    def wrap_env(self, env) -> gym.Wrapper:
        return _FixedShapeContinuousObsWrapperV2(env)

class _FixedShapeContinuousObsWrapperV2(gym.Wrapper):
    def __init__(self, env: gym.Env):
        """
        Initialize V2 wrapper with empty observation tensors.

        Creates pre-allocated tensors for all observation types including:
        - Unit positions and properties
        - Resource locations and quantities
        - City tile states
        - Spatial distance features
        - Global game state information

        Args:
            env: The Lux environment to wrap
        """
        super(_FixedShapeContinuousObsWrapperV2, self).__init__(env)
        self._empty_obs = {}
        for key, spec in FixedShapeContinuousObsV2().get_obs_spec().spaces.items():
            if isinstance(spec, gym.spaces.MultiBinary) or isinstance(spec, gym.spaces.MultiDiscrete):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.int64)
            elif isinstance(spec, gym.spaces.Box):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.float32) + spec.low
            else:
                raise NotImplementedError(f"{type(spec)} is not an accepted observation space.")

    def reset(self, **kwargs):
        observation, reward, done, info = self.env.reset(**kwargs)
        return self.observation(observation), reward, done, info

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation: Game) -> Dict[str, np.ndarray]:
        w_capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
        ca_capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
        w_cooldown = GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["WORKER"] * 2. - 1.
        ca_cooldown = GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["CART"] * 2. - 1.
        ci_light = GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CITY"]
        ci_cooldown = GAME_CONSTANTS["PARAMETERS"]["CITY_ACTION_COOLDOWN"]
        max_road = GAME_CONSTANTS["PARAMETERS"]["MAX_ROAD"]
        max_research = max(GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"].values())

        obs = {
            key: val.copy() if val.ndim == 2 else val[:, :, :observation.map_width, :observation.map_height].copy()
            for key, val in self._empty_obs.items()
        }

        for player in observation.players:
            p_id = player.team
            for unit in player.units:
                x, y = unit.pos.x, unit.pos.y
                if unit.is_worker():
                    obs["worker"][0, p_id, x, y] = 1
                    obs["worker_COUNT"][0, p_id, x, y] += 1
                    obs["worker_cooldown"][0, p_id, x, y] = unit.cooldown / w_cooldown
                    obs["worker_cargo_full"][0, p_id, x, y] = unit.get_cargo_space_left() == 0

                    obs[f"worker_cargo_{WOOD}"][0, p_id, x, y] = unit.cargo.wood / w_capacity
                    obs[f"worker_cargo_{COAL}"][0, p_id, x, y] = unit.cargo.coal / w_capacity
                    obs[f"worker_cargo_{URANIUM}"][0, p_id, x, y] = unit.cargo.uranium / w_capacity

                elif unit.is_cart():
                    obs["cart"][0, p_id, x, y] = 1
                    obs["cart_COUNT"][0, p_id, x, y] += 1
                    obs["cart_cooldown"][0, p_id, x, y] = unit.cooldown / ca_cooldown
                    obs["cart_cargo_full"][0, p_id, x, y] = unit.get_cargo_space_left() == 0

                    obs[f"cart_cargo_{WOOD}"][0, p_id, x, y] = unit.cargo.wood / ca_capacity
                    obs[f"cart_cargo_{COAL}"][0, p_id, x, y] = unit.cargo.coal / ca_capacity
                    obs[f"cart_cargo_{URANIUM}"][0, p_id, x, y] = unit.cargo.uranium / ca_capacity
                else:
                    raise NotImplementedError(f'New unit type: {unit}')

            for city in player.cities.values():
                city_fuel_normalized = city.fuel / MAX_FUEL / len(city.citytiles)
                city_light_normalized = city.light_upkeep / ci_light / len(city.citytiles)
                for city_tile in city.citytiles:
                    x, y = city_tile.pos.x, city_tile.pos.y
                    obs["city_tile"][0, p_id, x, y] = 1
                    obs["city_tile_fuel"][0, p_id, x, y] = city_fuel_normalized
                    # NB: This doesn't technically register the light upkeep of a given city tile, but instead
                    # the average light cost of every tile in the given city
                    obs["city_tile_cost"][0, p_id, x, y] = city_light_normalized
                    obs["city_tile_cooldown"][0, p_id, x, y] = city_tile.cooldown / ci_cooldown

            for cell in itertools.chain(*observation.map.map):
                x, y = cell.pos.x, cell.pos.y
                obs["road_level"][0, 0, x, y] = cell.road / max_road
                if cell.has_resource():
                    obs[f"{cell.resource.type}"][0, 0, x, y] = cell.resource.amount / MAX_RESOURCE[cell.resource.type]

            obs["research_points"][0, p_id] = min(player.research_points / max_research, 1.)
            obs["researched_coal"][0, p_id] = player.researched_coal()
            obs["researched_uranium"][0, p_id] = player.researched_uranium()
        obs["dist_from_center_x"][:] = self.get_dist_from_center_x(observation.map_width, observation.map_height)
        obs["dist_from_center_y"][:] = self.get_dist_from_center_y(observation.map_width, observation.map_height)
        obs["night"][0, 0] = observation.is_night
        obs["day_night_cycle"][0, 0] = observation.turn % DN_CYCLE_LEN
        obs["phase"][0, 0] = min(
            observation.turn // DN_CYCLE_LEN,
            GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"] / DN_CYCLE_LEN - 1
        )
        obs["turn"][0, 0] = observation.turn / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
        obs["board_size"][0, 0] = MAP_SIZES.index((observation.map_width, observation.map_height))

        # def save_board(board_dict, filename="board.pkl"):
        #     """
        #     Saves the observation dictionary (with NumPy arrays) to disk using pickle.
        #     """
        #     with open(filename, "wb") as f:
        #         pickle.dump(board_dict, f)
        #     print(f"Saved board dict to {filename}.")
        #
        # save_board(obs, filename="board-" + str(observation.turn) + ".pkl")

        return obs

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_dist_from_center_x(map_height: int, map_width: int) -> np.ndarray:
        """
        Calculate normalized X-axis distances from board center for each cell.

        Uses LRU cache to avoid recomputing for same board dimensions.
        Values are normalized to [0,1] where 0 is center and 1 is edge.

        Args:
            map_height: Height of the game board
            map_width: Width of the game board

        Returns:
            Array of shape [1,1,height,width] with normalized X-distances
        """
        pos = np.linspace(0, 2, map_width, dtype=np.float32)[None, :].repeat(map_height, axis=0)
        return np.abs(1 - pos)[None, None, :, :]

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_dist_from_center_y(map_height: int, map_width: int) -> np.ndarray:
        """
        Calculate normalized Y-axis distances from board center for each cell.

        Uses LRU cache to avoid recomputing for same board dimensions.
        Values are normalized to [0,1] where 0 is center and 1 is edge.

        Args:
            map_height: Height of the game board
            map_width: Width of the game board

        Returns:
            Array of shape [1,1,height,width] with normalized Y-distances
        """
        pos = np.linspace(0, 2, map_height)[:, None].repeat(map_width, axis=1)
        return np.abs(1 - pos)[None, None, :, :]
