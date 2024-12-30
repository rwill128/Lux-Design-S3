import copy
import gym
import itertools
import json
import numpy as np
from kaggle_environments import make
import math
from pathlib import Path
from queue import Queue, Empty
import random
from subprocess import Popen, PIPE
import sys
from threading import Thread
from typing import Any, Dict, List, NoReturn, Optional, Tuple

from src.luxai_s3.wrappers import LuxAIS3GymEnv
from ..lux.game import Game
from ..lux.game_objects import Unit
from ..lux_gym.act_spaces import BaseActSpace, ACTION_MEANINGS
from ..lux_gym.obs_spaces import BaseObsSpace
from ..lux_gym.reward_spaces import GameResultReward
from ..utility_constants import MAX_BOARD_SIZE

# In case dir_path is removed in production environment
try:
    from kaggle_environments.envs.lux_ai_2021.lux_ai_2021 import dir_path as DIR_PATH
except Exception:
    DIR_PATH = None


"""
def _cleanup_dimensions_factory(dimension_process: Popen) -> NoReturn:
    def cleanup_dimensions():
        if dimension_process is not None:
            dimension_process.kill()
    return cleanup_dimensions
"""


def _enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()


def _generate_pos_to_unit_dict(game_state: Game) -> Dict[Tuple, Optional[Unit]]:
    pos_to_unit_dict = {(cell.pos.x, cell.pos.y): None for cell in itertools.chain(*game_state.map.map)}
    for player in game_state.players:
        for unit in reversed(player.units):
            pos_to_unit_dict[(unit.pos.x, unit.pos.y)] = unit

    return pos_to_unit_dict


# noinspection PyProtectedMember
class Lux3Env(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
            self,
            act_space: BaseActSpace,
            obs_space: BaseObsSpace,
            configuration: Optional[Dict[str, Any]] = None,
            seed: Optional[int] = None,
            run_game_automatically: bool = True,
            restart_subproc_after_n_resets: int = 100
    ):
        super(Lux3Env, self).__init__()
        self.obs_space = obs_space
        self.action_space = act_space
        self.default_reward_space = GameResultReward()
        self.observation_space = self.obs_space.get_obs_spec()
        self.board_dims = MAX_BOARD_SIZE
        self.run_game_automatically = run_game_automatically
        self.restart_subproc_after_n_resets = restart_subproc_after_n_resets

        self.game_state = Game()
        if configuration is not None:
            self.configuration = configuration
        else:
            self.configuration = make("lux_ai_s3").configuration
            # 2: warnings, 1: errors, 0: none
            self.configuration["loglevel"] = 0
        if seed is not None:
            self.seed(seed)
        elif "seed" not in self.configuration:
            self.seed()
        self.done = False
        self.info = {}
        self.pos_to_unit_dict = dict()
        self.reset_count = 0

        self._real_game_env = None
        self._q = None
        self._t = None
        self._real_game_env = LuxAIS3GymEnv(numpy_output=True)

        obs, info = self._real_game_env.reset(seed=self.get_seed())

        self.env_config = info["params"]



    def reset(self, observation_updates: Optional[List[str]] = None) -> Tuple[Game, Tuple[float, float], bool, Dict]:

        self.configuration["seed"] = self.get_seed() + 1
        self._real_game_env = LuxAIS3GymEnv(numpy_output=True)
        obs, info = self._real_game_env.reset(seed=self.get_seed())
        self.env_config = info["params"]

        self.game_state._initialize(obs, self.env_config)
        self.game_state._update(obs)


        self.done = False
        self.board_dims = (24, 24)
        self.observation_space = self.obs_space.get_obs_spec(self.board_dims)
        self.info = {
            "actions_taken": {
                key: np.zeros(space.shape + (len(ACTION_MEANINGS[key]),), dtype=bool)
                for key, space in self.action_space.get_action_space(self.board_dims).spaces.items()
            },
            "available_actions_mask": {
                key: np.ones(space.shape + (len(ACTION_MEANINGS[key]),), dtype=bool)
                for key, space in self.action_space.get_action_space(self.board_dims).spaces.items()
            }
        }
        self._update_internal_state()

        return self.get_obs_reward_done_info()

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Game, Tuple[float, float], bool, Dict]:
        if self.run_game_automatically:
            actions_processed, actions_taken = self.process_actions(action)
            self._step(actions_processed)
            self.info["actions_taken"] = actions_taken
        self._update_internal_state()

        return self.get_obs_reward_done_info()

    def manual_step(self, observation_updates: List[str]) -> NoReturn:
        assert not self.run_game_automatically
        self.game_state._update(observation_updates)

    def get_obs_reward_done_info(self) -> Tuple[Game, Tuple[float, float], bool, Dict]:
        rewards = self.default_reward_space.compute_rewards(game_state=self.game_state, done=self.done)
        return self.game_state, rewards, self.done, copy.copy(self.info)

    def process_actions(self, action: Dict[str, np.ndarray]) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
        return self.action_space.process_actions(
            action,
            self.game_state,
            self.board_dims,
            self.pos_to_unit_dict
        )

    def _step(self, action_player_one: List[List[str]], action_player_two) -> NoReturn:
        # 2.: Pass in actions (json representation along with id of who made that action),
        #       and agent information (id, status) to dimensions via stdin
        actions = {
            "player_0": action_player_one,
            "player_1": action_player_two,
        }

        obs, reward, terminated, truncated, info = self._real_game_env.step(actions)
        # 3.1 : Receive and parse the observations returned by dimensions via stdout
        self.game_state._update(obs)

        # Check if done
        self.done = terminated or truncated

    def _update_internal_state(self) -> NoReturn:
        self.pos_to_unit_dict = _generate_pos_to_unit_dict(self.game_state)
        self.info["available_actions_mask"] = self.action_space.get_available_actions_mask(
            self.game_state,
            self.board_dims,
            self.pos_to_unit_dict,
            self.pos_to_city_tile_dict
        )

    def seed(self, seed: Optional[int] = None) -> NoReturn:
        if seed is not None:
            # Seed is incremented on reset()
            self.configuration["seed"] = seed + 1
        else:
            self.configuration["seed"] = math.floor(random.random() * 1e9)

    def get_seed(self) -> int:
        return self.configuration["seed"]

    def render(self, mode='human'):
        raise NotImplementedError('LuxEnv rendering is not implemented. Use the Lux visualizer instead.')
