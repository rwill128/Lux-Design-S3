import os
import json
import numpy as np
import jax
import flax
import dataclasses
import gymnasium as gym
from gymnasium import spaces
from typing import Any, SupportsFloat

import ray
from ray import tune
from ray.rllib import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig

# Suppose these come from your own code:
# - LuxAIS3Env: The JAX environment
# - flatten_player_obs / flatten_config: flatten obs & config
# - RecordEpisode: the wrapper that records episodes to JSON
from luxai_s3.env import LuxAIS3Env
from luxai_s3.utils import to_numpy
from luxai_s3.params import EnvParams, env_params_ranges
from luxai_s3.wrappers import RecordEpisode  # your existing code
# ^^^ Replace "your_flatten_helpers" with the actual file where you define them


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
#  1) Base multi-agent environment that returns dict obs, etc.
# ----------------------------------------------------------------------
class LuxAIS3GymEnvWrap(MultiAgentEnv):
    """
    A multi-agent environment that calls a JAX-based LuxAIS3Env,
    returning multi-agent style (obs_dict, rew_dict, terminated_dict, truncated_dict, info).
    """

    def __init__(self, numpy_output: bool = False):
        super().__init__()
        self.numpy_output = numpy_output
        self.rng_key = jax.random.PRNGKey(0)
        self.jax_env = LuxAIS3Env(auto_reset=False)
        self.env_params: EnvParams = EnvParams()

        # We typically flatten obs to shape (110,):
        self.obs_size = 100
        self.cfg_size = 10
        self.single_obs_shape = (self.obs_size + self.cfg_size,)  # => (110,)

        # One Box(...) for a single agent's observation:
        self.observation_space = spaces.Box(
            low=-1e9, high=1e9,
            shape=self.single_obs_shape,  # (110,)
            dtype=np.float32
        )

        # One Box(...) for a single agent's action. Suppose 16 units, each with 3 dims => shape=(48,).
        self.action_space = spaces.Box(
            low=-10, high=10,
            shape=(16 * 3,),
            dtype=np.int16,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
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

        obs, self.state = self.jax_env.reset(reset_key, params=params)

        if self.numpy_output:
            # Convert JAX arrays to numpy
            obs = to_numpy(flax.serialization.to_state_dict(obs))

        # Flatten config
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

        # Flatten obs for each agent
        obs_player_0 = flatten_player_obs(obs, team_id=0, obs_size=self.obs_size, cfg_array=cfg_array)
        obs_player_1 = flatten_player_obs(obs, team_id=1, obs_size=self.obs_size, cfg_array=cfg_array)

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
            self, action: dict
    ) -> tuple[dict, dict, dict, dict, dict]:
        """
        Must return: (obs_dict, rew_dict, terminated_dict, truncated_dict, info_dict)
        each keyed by agent plus '__all__' for done flags.
        """
        self.rng_key, step_key = jax.random.split(self.rng_key)
        # action is e.g. {"player_0": np.array(48-dim), "player_1": np.array(48-dim)}
        obs, self.state, reward, terminated, truncated, info = self.jax_env.step(
            step_key, self.state, action, self.env_params
        )

        if self.numpy_output:
            obs = to_numpy(flax.serialization.to_state_dict(obs))
            reward = to_numpy(reward)
            terminated = to_numpy(terminated)
            truncated = to_numpy(truncated)

        # Possibly re-flatten config
        params_dict_kept = info.get("params", None)
        if params_dict_kept:
            cfg_array = flatten_config(params_dict_kept)
        else:
            cfg_array = np.zeros(self.cfg_size, dtype=np.float32)

        # Flatten obs for each agent
        obs_player_0 = flatten_player_obs(obs, team_id=0, obs_size=self.obs_size, cfg_array=cfg_array)
        obs_player_1 = flatten_player_obs(obs, team_id=1, obs_size=self.obs_size, cfg_array=cfg_array)
        final_obs = {
            "player_0": obs_player_0,
            "player_1": obs_player_1,
        }

        # Convert reward, done, truncated to dict
        rew_dict = {
            "player_0": float(reward["player_0"]),
            "player_1": float(reward["player_1"]),
        }
        term_dict = {
            "player_0": bool(terminated["player_0"]),
            "player_1": bool(terminated["player_1"]),
            "__all__": bool(terminated["player_0"]) or bool(terminated["player_1"]),
        }
        trunc_dict = {
            "player_0": bool(truncated["player_0"]),
            "player_1": bool(truncated["player_1"]),
            "__all__": bool(truncated["player_0"]) or bool(truncated["player_1"]),
        }

        # Remove any extra keys from info that RLlib doesn't like:
        common_dict = {}
        for k in ["discount", "final_observation", "final_state"]:
            if k in info:
                common_dict[k] = info.pop(k)
        if common_dict:
            info["__common__"] = common_dict

        return final_obs, rew_dict, term_dict, trunc_dict, info

    def render(self):
        # Optional
        self.jax_env.render(self.state, self.env_params)

    def close(self):
        pass


# ----------------------------------------------------------------------
#  2) "RecordingLuxEnv" class that wraps the above in RecordEpisode
# ----------------------------------------------------------------------
class RecordingLuxEnv(MultiAgentEnv):
    """
    This environment *wraps* LuxAIS3GymEnvWrap with RecordEpisode,
    so that every episode is saved to JSON in `save_dir`.
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}
        save_dir = config.get("save_dir", "replays_rllib")

        # 1) Create the base multi-agent env
        base_env = LuxAIS3GymEnvWrap(numpy_output=True)

        # 2) Wrap it to record episodes
        self.env = RecordEpisode(
            env=base_env,
            save_dir=save_dir,
            save_on_close=True,
            save_on_reset=True,
        )

        # Expose observation_space/action_space as single-agent definitions
        # (Same for "player_0" and "player_1" in multi-agent).
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        return obs, rew, term, trunc, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

# ----------------------------------------------------------------------
#  3) RLlib Training Config
# ----------------------------------------------------------------------

def my_policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    # For 2 teams => "player_0" => policy_0, "player_1" => policy_1
    return f"policy_{agent_id[-1]}"

# We'll define each policy's obs/action space as 110-dim obs, 48-dim actions
obs_space = spaces.Box(low=-1e9, high=1e9, shape=(110,), dtype=np.float32)
act_space = spaces.Box(low=-10, high=10, shape=(16*3,), dtype=np.int16)

config = (
    PPOConfig()
    # Use the old API stack
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .environment(
        env=RecordingLuxEnv,  # the class
        env_config={"save_dir": "replays_rllib"}  # pass the directory
    )
    .framework("torch")
    .env_runners(num_env_runners=1)  # number of rollout workers
    .training(
        model={"fcnet_hiddens": [128, 128]}  # old API stack setting
    )
    .multi_agent(
        policies={
            "policy_0": (None, obs_space, act_space, {}),
            "policy_1": (None, obs_space, act_space, {}),
        },
        policy_mapping_fn=my_policy_mapping_fn,
    )
    .resources(num_gpus=0)
)

if __name__ == "__main__":
    ray.init()
    tune.run(
        "PPO",
        config=config.to_dict(),
        stop={"timesteps_total": 1_000_000},
        storage_path="/home/rick/IdeaProjects/Lux-Design-S3/run_match/rllib_logs"
    )
