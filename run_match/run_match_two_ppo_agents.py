import ray
from luxai_s3.wrappers import LuxAIS3GymEnv

from ray import tune
from ray.rllib import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig

ray.init()


import json
import os
from typing import Any, SupportsFloat
import dataclasses

import flax
import flax.serialization
import gymnasium as gym
import jax
import numpy as np

from gymnasium import spaces
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams, env_params_ranges
from luxai_s3.state import serialize_env_actions, serialize_env_states
from luxai_s3.utils import to_numpy

# ----------------------------------------------------------------------
#  FLATTENING HELPER: EXACTLY LIKE YOUR "flatten_observation" EXAMPLE
# ----------------------------------------------------------------------
def flatten_player_obs(obs: dict, team_id: int, obs_size=100, cfg_array=None):
    """
    Flatten one player's portion of the structured observation into a shape (110,).
    This is the same logic you had in your DQN code, but now we break it out so
    the gym Env can call it.
    """
    # The user-provided doc says "obs" might look like:
    #   {
    #     "units": {
    #       "position": Array(T, N, 2),
    #       "energy": Array(T, N, 1)
    #     },
    #     "units_mask": Array(T, N),
    #     "sensor_mask": Array(W, H),
    #     "map_features": {...},
    #     "relic_nodes_mask": Array(R),
    #     "relic_nodes": Array(R, 2),
    #     "team_points": Array(T),
    #     ...
    #   }
    # We'll pick out obs["units"]["position"][team_id], etc. for flattening.

    # NOTE: If "team_id" dimension in the raw arrays is the first dimension,
    # we do obs["units"]["position"][team_id]. If it's the second dimension,
    # adjust accordingly.

    # For safety, we do a lot of .get checks, but presumably all these exist:
    player_id = "player_0" if team_id is 0 else "player_1"
    id_ = obs[player_id]
    position = np.array(id_["units"]["position"][team_id]).flatten()
    energy   = np.array(id_["units"]["energy"][team_id]).flatten()
    sensor   = np.array(id_["sensor_mask"]).flatten()               # shape W*H
    tile_e   = np.array(id_["map_features"]["energy"]).flatten()    # shape W*H
    tile_t   = np.array(id_["map_features"]["tile_type"]).flatten() # shape W*H
    relic_m  = np.array(id_["relic_nodes_mask"]).flatten()          # shape R
    relic_p  = np.array(id_["relic_nodes"]).flatten()               # shape (R,2) => 2R
    tpoints  = np.array(id_["team_points"]).flatten()               # shape T => e.g. 2

    features = np.concatenate([
        position,
        energy,
        sensor,
        tile_e,
        tile_t,
        relic_m,
        relic_p,
        tpoints,
    ])

    if len(features) >= obs_size:
        features = features[:obs_size]
    else:
        features = np.pad(features, (0, obs_size - len(features)))

    if cfg_array is None:
        # If you have no env_cfg to append, just return the features
        return features.astype(np.float32)
    else:
        # shape = obs_size + len(cfg_array)
        return np.concatenate([features, cfg_array]).astype(np.float32)


def flatten_config(params_dict_kept: dict):
    """
    Turn the env_cfg dict into a fixed-size vector of length 10 (like your original example).
    """
    return np.array([
        params_dict_kept["max_units"],
        params_dict_kept["match_count_per_episode"],
        params_dict_kept["max_steps_in_match"],
        params_dict_kept["map_height"],
        params_dict_kept["map_width"],
        params_dict_kept["num_teams"],
        params_dict_kept["unit_move_cost"],
        params_dict_kept["unit_sap_cost"],
        params_dict_kept["unit_sap_range"],
        params_dict_kept["unit_sensor_range"],
    ], dtype=np.float32)


# ----------------------------------------------------------------------
#  OUR ENV CLASS
# ----------------------------------------------------------------------
class LuxAIS3GymEnvWrap(MultiAgentEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(self, numpy_output: bool = False):
        super().__init__()
        self.numpy_output = numpy_output
        self.rng_key = jax.random.PRNGKey(0)
        self.jax_env = LuxAIS3Env(auto_reset=False)
        self.env_params: EnvParams = EnvParams()

        # We'll define these to help with flattening
        # You mentioned you typically do obs_size=100, plus 10 from config => 110
        self.obs_size = 100
        self.cfg_size = 10  # if you keep 10 param values
        self.single_obs_shape = (self.obs_size + self.cfg_size,)  # => (110,)

        # We want a multi-agent observation space (dict with 'player_0' and 'player_1')
        # Each is a Box of shape (110,).
        single_player_obs_space = spaces.Box(
            low=-1e9, high=-1e9, shape=self.single_obs_shape, dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "player_0": single_player_obs_space,
                "player_1": single_player_obs_space,
            }
        )

        # For the action space, you already have a Dict with keys "player_0" and "player_1".
        # Example from your code:
        low = np.zeros((self.env_params.max_units, 3))
        low[:, 1:] = -self.env_params.unit_sap_range
        high = np.ones((self.env_params.max_units, 3)) * 6
        high[:, 1:] = self.env_params.unit_sap_range
        self.action_space = spaces.Dict(
            dict(
                player_0=spaces.Box(low=low, high=high, dtype=np.int16),
                player_1=spaces.Box(low=low, high=high, dtype=np.int16),
            )
        )

    def render(self):
        # Just pass to underlying JAX env’s render
        self.jax_env.render(self.state, self.env_params)

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ):
        if seed is not None:
            self.rng_key = jax.random.PRNGKey(seed)
        self.rng_key, reset_key = jax.random.split(self.rng_key)

        # Randomize environment parameters
        randomized_game_params = {}
        for k, v in env_params_ranges.items():
            self.rng_key, subkey = jax.random.split(self.rng_key)
            randomized_game_params[k] = jax.random.choice(subkey, jax.numpy.array(v)).item()

        params = EnvParams(**randomized_game_params)
        if options is not None and "params" in options:
            params = options["params"]
        self.env_params = params

        # Reset the JAX env
        obs, self.state = self.jax_env.reset(reset_key, params=params)

        if self.numpy_output:
            obs = to_numpy(flax.serialization.to_state_dict(obs))

        # Keep only relevant params
        params_dict = dataclasses.asdict(params)
        params_dict_kept = {
            k: params_dict[k]
            for k in [
                "max_units",
                "match_count_per_episode",
                "max_steps_in_match",
                "map_height",
                "map_width",
                "num_teams",
                "unit_move_cost",
                "unit_sap_cost",
                "unit_sap_range",
                "unit_sensor_range",
            ]
        }
        cfg_array = flatten_config(params_dict_kept)  # shape=(10,)

        # "obs" here is the dictionary from JAX with structure
        #   obs["units"]["position"][2D array], obs["sensor_mask"], etc.
        # We must produce two flattened obs: for player_0 and player_1.
        # Let's assume T=2 for 2 teams, so:
        #   flatten_player_obs(obs, team_id=0, obs_size=100, cfg_array=cfg_array)
        #   flatten_player_obs(obs, team_id=1, obs_size=100, cfg_array=cfg_array)
        # If your environment doesn't strictly do [team_id=0, team_id=1], adjust as needed.

        obs_player_0 = flatten_player_obs(
            obs, team_id=0, obs_size=self.obs_size, cfg_array=cfg_array
        )
        obs_player_1 = flatten_player_obs(
            obs, team_id=1, obs_size=self.obs_size, cfg_array=cfg_array
        )

        final_obs = {
            "player_0": obs_player_0,
            "player_1": obs_player_1,
        }

        info = {
            "params": params_dict_kept,
            "full_params": params_dict,
            "state": self.state,
        }
        return final_obs, info

    def step(
            self, action: Any
    ) -> tuple[dict, SupportsFloat, dict, dict, dict]:
        """
        Must return: (obs, reward, terminated, truncated, info)
         - obs: dict of obs for each agent
         - reward: dict of float for each agent
         - terminated: dict of bool for each agent
         - truncated: dict of bool for each agent
         - info: dict of extra info
        """
        self.rng_key, step_key = jax.random.split(self.rng_key)
        obs, self.state, reward, terminated, truncated, info = self.jax_env.step(
            step_key, self.state, action, self.env_params
        )

        if self.numpy_output:
            obs = to_numpy(flax.serialization.to_state_dict(obs))
            reward = to_numpy(reward)
            terminated = to_numpy(terminated)
            truncated = to_numpy(truncated)
            # info = to_numpy(flax.serialization.to_state_dict(info)) # optional

        # Flatten new observation for each agent
        # We also must keep track of the same env_cfg array if needed. For simplicity,
        # let's just store it in self.cfg_array from the last reset or from info, etc.
        # Alternatively, we can do the same logic in reset (assuming env_params won't change).
        params_dict_kept = info.get("params", None)
        if params_dict_kept:
            cfg_array = flatten_config(params_dict_kept)
        else:
            # fallback, or store it from last time
            cfg_array = np.zeros(self.cfg_size, dtype=np.float32)

        obs_player_0 = flatten_player_obs(
            obs, team_id=0, obs_size=self.obs_size, cfg_array=cfg_array
        )
        obs_player_1 = flatten_player_obs(
            obs, team_id=1, obs_size=self.obs_size, cfg_array=cfg_array
        )
        final_obs = {
            "player_0": obs_player_0,
            "player_1": obs_player_1,
        }

        # Because this is multi-agent, we want to return dicts for reward/terminated/truncated.
        # The JAX env might produce them as arrays or whatever shape. Let's convert:
        # reward might be shape (2,) => we map it to { "player_0": reward[0], "player_1": reward[1] }
        # same with terminated, truncated.
        # If your underlying env is guaranteed 2 players, index them. If variable, adapt accordingly.
        rew_dict = {
            "player_0": float(reward[0]),
            "player_1": float(reward[1]),
        }
        term_dict = {
            "player_0": bool(terminated[0]),
            "player_1": bool(terminated[1]),
        }
        trunc_dict = {
            "player_0": bool(truncated[0]),
            "player_1": bool(truncated[1]),
        }

        return final_obs, rew_dict, term_dict, trunc_dict, info



gym_env = LuxAIS3GymEnvWrap(numpy_output=True)


def my_policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    return f"policy_{agent_id[-1]}"

# Suppose your environment returns a dictionary of {agent_id: obs, ...} etc.
# and can handle actions in a dict as well.
config = (
    PPOConfig()
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .environment(env=LuxAIS3GymEnvWrap, env_config={"param_1": 123, "param_2": 456})
    .framework("torch")
    .env_runners(num_env_runners=1)
    .training(model={"fcnet_hiddens": [128, 128]})
    .multi_agent(
        policies={
            "policy_0": (None, gym_env.observation_space, gym_env.action_space, {}),
            "policy_1": (None, gym_env.observation_space, gym_env.action_space, {}),
        },
        policy_mapping_fn=my_policy_mapping_fn,
    )
    .resources(num_gpus=0)
)

tune.run(
    "PPO",
    config=config.to_dict(),
    stop={"timesteps_total": 1_000_000},
    storage_path="/home/rick/IdeaProjects/Lux-Design-S3/run_match/rllib_logs"
)
