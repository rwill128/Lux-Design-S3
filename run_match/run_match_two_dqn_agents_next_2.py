import os
import numpy as np
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

LR = 1e-3
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1e-4
REPLAY_SIZE = 10_000
BATCH_SIZE = 32

global_qnet_1 = None
global_optim_1 = None
global_replay_buffer_1 = None
global_epsilon_1 = EPS_START

global_qnet_2 = None
global_optim_2 = None
global_replay_buffer_2 = None
global_epsilon_2 = EPS_START

def flatten_config(env_cfg):
    """
    Turn the env_cfg dict into a fixed-size vector of 10 elements.
    """
    return np.array([
        env_cfg["max_units"],
        env_cfg["match_count_per_episode"],
        env_cfg["max_steps_in_match"],
        env_cfg["map_height"],
        env_cfg["map_width"],
        env_cfg["num_teams"],
        env_cfg["unit_move_cost"],
        env_cfg["unit_sap_cost"],
        env_cfg["unit_sap_range"],
        env_cfg["unit_sensor_range"],
    ], dtype=np.float32)

def maybe_init_globals(agent_id: int, state_size: int, action_size: int):
    global global_qnet_1, global_optim_1, global_replay_buffer_1, global_epsilon_1
    global global_qnet_2, global_optim_2, global_replay_buffer_2, global_epsilon_2

    class QNet(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(QNet, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )

        def forward(self, x):
            return self.net(x)

    if agent_id == 1:
        if global_qnet_1 is None:
            global_qnet_1 = QNet(state_size, action_size)
            global_optim_1 = optim.Adam(global_qnet_1.parameters(), lr=LR)
            global_replay_buffer_1 = deque(maxlen=REPLAY_SIZE)
            global_epsilon_1 = EPS_START
    else:
        if global_qnet_2 is None:
            global_qnet_2 = QNet(state_size, action_size)
            global_optim_2 = optim.Adam(global_qnet_2.parameters(), lr=LR)
            global_replay_buffer_2 = deque(maxlen=REPLAY_SIZE)
            global_epsilon_2 = EPS_START

def get_globals(agent_id: int):
    if agent_id == 1:
        return (
            global_qnet_1, global_optim_1, global_replay_buffer_1,
            lambda: None,  # placeholder
            lambda: None,  # placeholder
            lambda: None,  # placeholder
        )
    else:
        return (
            global_qnet_2, global_optim_2, global_replay_buffer_2,
            lambda: None,
            lambda: None,
            lambda: None,
        )

def get_epsilon(agent_id: int):
    global global_epsilon_1, global_epsilon_2
    return global_epsilon_1 if agent_id == 1 else global_epsilon_2

def set_epsilon(agent_id: int, val: float):
    global global_epsilon_1, global_epsilon_2
    if agent_id == 1:
        global_epsilon_1 = val
    else:
        global_epsilon_2 = val

def store_transition(agent_id: int, transition):
    _, _, replay_buffer, _, _, _ = get_globals(agent_id)
    replay_buffer.append(transition)

def sample_transitions(agent_id: int):
    _, _, replay_buffer, _, _, _ = get_globals(agent_id)
    batch = random.sample(replay_buffer, BATCH_SIZE)
    return batch

def compute_q_loss(agent_id: int):
    qnet, optim_, replay_buffer, _, _, _ = get_globals(agent_id)

    if len(replay_buffer) < BATCH_SIZE:
        return  # not enough data to train

    batch = sample_transitions(agent_id)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    q_values = qnet(states)  # (batch_size, action_dim)
    q_val_a = q_values.gather(1, actions.view(-1, 1)).squeeze(-1)

    with torch.no_grad():
        q_next = qnet(next_states)
        q_next_max = q_next.max(dim=1)[0]

    target = rewards + GAMMA * q_next_max * (1 - dones)

    loss = nn.MSELoss()(q_val_a, target)

    optim_.zero_grad()
    loss.backward()
    optim_.step()

def select_action_dqn(agent_id: int, state: np.ndarray, action_size: int):
    eps = get_epsilon(agent_id)
    new_eps = max(EPS_END, eps - EPS_DECAY)
    set_epsilon(agent_id, new_eps)

    qnet, _, _, _, _, _ = get_globals(agent_id)
    if random.random() < eps:
        return np.random.randint(0, action_size)
    else:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = qnet(state_t)
            return int(q_values.argmax(dim=1).item())

###########################################
#  DQNAgent1 and DQNAgent2 with env_cfg   #
###########################################

class DQNAgent1():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.team_id = 0 if self.player == "player_0" else 1

        # Flatten the config so we can incorporate it in our net inputs.
        self.cfg_array = flatten_config(env_cfg)

        self.env_cfg = env_cfg
        self.max_units = env_cfg["max_units"]
        self.action_size = 10

        # OLD: self.obs_size = 100
        # We now do:
        self.obs_size = 100
        self.cfg_size = len(self.cfg_array)  # = 10
        self.input_size = self.obs_size + self.cfg_size

        # Initialize the global DQN model with the new input size
        maybe_init_globals(1, self.input_size, self.action_size)

        self.last_obs = None
        self.last_action = None

    def flatten_observation(self, obs):
        unit_positions = np.array(obs["units"]["position"][self.team_id]).flatten()
        unit_energies = np.array(obs["units"]["energy"][self.team_id]).flatten()
        sensor_mask = np.array(obs["sensor_mask"]).flatten()
        tile_energy = np.array(obs["map_features"]["energy"]).flatten()
        tile_type = np.array(obs["map_features"]["tile_type"]).flatten()
        relic_mask = np.array(obs["relic_nodes_mask"]).flatten()
        relic_positions = np.array(obs["relic_nodes"]).flatten()
        team_points = np.array(obs["team_points"]).flatten()

        features = np.concatenate(
            (
                unit_positions,
                unit_energies,
                sensor_mask,
                tile_energy,
                tile_type,
                relic_mask,
                relic_positions,
                team_points,
            )
        )
        if len(features) >= self.obs_size:
            features = features[: self.obs_size]
        else:
            features = np.pad(features, (0, self.obs_size - len(features)))

        # Concatenate the env_cfg array to form final state
        # ( shape = 100 + 10 = 110 )
        state_with_cfg = np.concatenate([features, self.cfg_array])
        return state_with_cfg

    def act(self, step: int, obs, reward, done, remainingOverageTime: int = 60):
        current_state = self.flatten_observation(obs)

        if self.last_obs is not None and self.last_action is not None:
            store_transition(
                1,
                (self.last_obs, self.last_action, reward, current_state, float(done))
            )
            compute_q_loss(1)

        action_idx = select_action_dqn(1, current_state, self.action_size)

        actions = np.zeros((self.max_units, 3), dtype=int)
        direction = action_idx % 3
        magnitude = (action_idx // 3)
        actions[0, 0] = direction
        actions[0, 1] = magnitude
        actions[0, 2] = 0

        self.last_obs = current_state
        self.last_action = action_idx

        return actions


class DQNAgent2():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.team_id = 0 if self.player == "player_0" else 1

        # Flatten config
        self.cfg_array = flatten_config(env_cfg)
        self.env_cfg = env_cfg

        self.max_units = env_cfg["max_units"]
        self.action_size = 10

        # Same input dimension as agent1
        self.obs_size = 100
        self.cfg_size = len(self.cfg_array)
        self.input_size = self.obs_size + self.cfg_size

        maybe_init_globals(2, self.input_size, self.action_size)

        self.last_obs = None
        self.last_action = None

    def flatten_observation(self, obs):
        unit_positions = np.array(obs["units"]["position"][self.team_id]).flatten()
        unit_energies = np.array(obs["units"]["energy"][self.team_id]).flatten()
        sensor_mask = np.array(obs["sensor_mask"]).flatten()
        tile_energy = np.array(obs["map_features"]["energy"]).flatten()
        tile_type = np.array(obs["map_features"]["tile_type"]).flatten()
        relic_mask = np.array(obs["relic_nodes_mask"]).flatten()
        relic_positions = np.array(obs["relic_nodes"]).flatten()
        team_points = np.array(obs["team_points"]).flatten()

        features = np.concatenate(
            (
                unit_positions,
                unit_energies,
                sensor_mask,
                tile_energy,
                tile_type,
                relic_mask,
                relic_positions,
                team_points,
            )
        )
        if len(features) >= self.obs_size:
            features = features[: self.obs_size]
        else:
            features = np.pad(features, (0, self.obs_size - len(features)))

        # Merge obs with config
        state_with_cfg = np.concatenate([features, self.cfg_array])
        return state_with_cfg

    def act(self, step: int, obs, reward, done, remainingOverageTime: int = 60):
        current_state = self.flatten_observation(obs)

        if self.last_obs is not None and self.last_action is not None:
            store_transition(
                2,
                (self.last_obs, self.last_action, reward, current_state, float(done))
            )
            compute_q_loss(2)

        action_idx = select_action_dqn(2, current_state, self.action_size)

        actions = np.zeros((self.max_units, 3), dtype=int)
        direction = action_idx % 3
        magnitude = (action_idx // 3)
        actions[0, 0] = direction
        actions[0, 1] = magnitude
        actions[0, 2] = 0

        self.last_obs = current_state
        self.last_action = action_idx

        return actions


def evaluate_agents(agent_1_cls, agent_2_cls, seed=45, games_to_play=3, replay_save_dir="replays"):
    # Ensure the replay directory exists
    os.makedirs(replay_save_dir, exist_ok=True)

    # Create an environment wrapped to record episodes
    gym_env = LuxAIS3GymEnv(numpy_output=True)
    gym_env.render_mode = "human"
    env = RecordEpisode(
        gym_env, save_on_close=True, save_on_reset=True, save_dir=replay_save_dir
    )

    for i in range(games_to_play):
        # Reset the environment
        obs, info = env.reset(seed=seed + i)  # changing seed each game
        env_cfg = info["params"]

        # Instantiate the RL agents. They will share the global Q-network.
        player_0 = agent_1_cls("player_0", env_cfg)
        player_1 = agent_2_cls("player_1", env_cfg)

        game_done = False
        step = 0
        print(f"Running game {i + 1}/{games_to_play}")

        # We'll track each player's last reward for training usage
        last_reward_0 = 0.0
        last_reward_1 = 0.0

        while not game_done:
            # Agents produce actions
            act_0 = player_0.act(
                step=step,
                obs=obs["player_0"],
                reward=last_reward_0,
                done=game_done
            )
            act_1 = player_1.act(
                step=step,
                obs=obs["player_1"],
                reward=last_reward_1,
                done=game_done
            )

            actions = {
                "player_0": act_0,
                "player_1": act_1,
            }

            obs, reward, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] or truncated[k] for k in terminated}
            if dones["player_0"] or dones["player_1"]:
                game_done = True

            # Update last rewards
            last_reward_0 = reward["player_0"]
            last_reward_1 = reward["player_1"]

            step += 1

        player_0.act(
            step=step,
            obs=obs["player_0"],
            reward=last_reward_0,
            done=game_done
        )
        player_1.act(
            step=step,
            obs=obs["player_1"],
            reward=last_reward_1,
            done=game_done
        )

    env.close()
    print(f"Finished {games_to_play} games. Replays (if any) saved to {replay_save_dir}")


if __name__ == "__main__":
    evaluate_agents(
        DQNAgent1,
        DQNAgent2,
        games_to_play=20,
        seed=2,
        replay_save_dir="replays/" + DQNAgent1.__name__ + "_" + DQNAgent2.__name__
    )
    print("Done.")
