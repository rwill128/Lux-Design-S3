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

def maybe_init_globals(agent_id: int, state_dim: int, max_units: int, dx_range=5, dy_range=5):
    global global_qnet_1, global_optim_1, global_replay_buffer_1, global_epsilon_1
    global global_qnet_2, global_optim_2, global_replay_buffer_2, global_epsilon_2

    class MultiUnitQNet(nn.Module):
        """
        Multi-head DQN-style network for controlling up to `max_units` units.
        Produces discrete distributions for:
          - action_type in [0..5]
          - dx in some discrete range (e.g. 5 steps => [-2, -1, 0, +1, +2])
          - dy in that same discrete range
        """
        def __init__(self, state_dim, max_units, dx_range=5, dy_range=5):
            super(MultiUnitQNet, self).__init__()
            self.max_units = max_units
            self.dx_range = dx_range
            self.dy_range = dy_range

            # A shared feature extractor for the global state
            self.shared_net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )

            # For each of the max_units, we have:
            #   - 1 head for action_type logits (size 6)
            #   - 1 head for dx logits (size dx_range)
            #   - 1 head for dy logits (size dy_range)
            self.action_type_heads = nn.ModuleList([
                nn.Linear(128, 6) for _ in range(max_units)
            ])
            self.dx_heads = nn.ModuleList([
                nn.Linear(128, dx_range) for _ in range(max_units)
            ])
            self.dy_heads = nn.ModuleList([
                nn.Linear(128, dy_range) for _ in range(max_units)
            ])

        def forward(self, state):
            """
            Args:
                state: (batch_size, state_dim) or (state_dim,) for a single sample
            Returns:
                tuple of (action_type_logits, dx_logits, dy_logits):
                  Each is a tensor of shape (batch_size, max_units, <num_classes>)
            """
            # If a single sample comes in, reshape to (1, state_dim)
            if state.dim() == 1:
                state = state.unsqueeze(0)  # (1, state_dim)

            base_out = self.shared_net(state)  # (batch_size, 128)

            # For each unit, produce a distribution over action_type, dx, dy
            # We'll collect them in Python lists, then stack.
            at_logits_list = []
            dx_logits_list = []
            dy_logits_list = []

            for i in range(self.max_units):
                at_logits_list.append(self.action_type_heads[i](base_out))
                dx_logits_list.append(self.dx_heads[i](base_out))
                dy_logits_list.append(self.dy_heads[i](base_out))

            # Each entry is (batch_size, something), stack them => (batch_size, max_units, something)
            action_type_logits = torch.stack(at_logits_list, dim=1)
            dx_logits = torch.stack(dx_logits_list, dim=1)
            dy_logits = torch.stack(dy_logits_list, dim=1)

            return action_type_logits, dx_logits, dy_logits

    if agent_id == 1:
        if global_qnet_1 is None:
            global_qnet_1 = MultiUnitQNet(state_dim, max_units)
            global_optim_1 = optim.Adam(global_qnet_1.parameters(), lr=LR)
            global_replay_buffer_1 = deque(maxlen=REPLAY_SIZE)
            global_epsilon_1 = EPS_START
    else:
        if global_qnet_2 is None:
            global_qnet_2 = MultiUnitQNet(state_dim, max_units)
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


def select_actions_from_qnet(qnet, state, max_units, dx_range=5, dy_range=5):
    """
    Args:
        qnet: instance of MultiUnitQNet
        state: np.ndarray or torch.Tensor of shape (state_dim,)
        max_units: number of units to control
        dx_range, dy_range: int, how many discrete values for x and y
    Returns:
        actions: np.ndarray of shape (max_units, 3), each row = [action_type, dx, dy]
    """
    # Convert state to torch and forward pass
    state_t = torch.FloatTensor(state)
    with torch.no_grad():
        at_logits, dx_logits, dy_logits = qnet(state_t)
        # Each has shape (1, max_units, classes)
        # So we remove the batch dim => shape (max_units, classes)
        at_logits = at_logits.squeeze(0)  # (max_units, 6)
        dx_logits = dx_logits.squeeze(0)  # (max_units, dx_range)
        dy_logits = dy_logits.squeeze(0)  # (max_units, dy_range)

        # For each unit i in [0..max_units-1], pick argmax
        action_types = at_logits.argmax(dim=1)  # (max_units,)
        dx_idx = dx_logits.argmax(dim=1)       # (max_units,)
        dy_idx = dy_logits.argmax(dim=1)       # (max_units,)

    # Convert to actual integers
    # Suppose we define a mapping for dx_idx => [-2..2], same for dy
    # This is just an example: if dx_range=5, indices=0..4 => offsets [-2..2].
    offset_min = -2
    offset_max = 2
    dx_values = dx_idx + offset_min  # shift from 0..4 -> -2..2
    dy_values = dy_idx + offset_min  # same shift

    # Build final actions array
    actions = []
    for i in range(max_units):
        a_type = int(action_types[i].item())  # 0..5
        dx = int(dx_values[i].item())
        dy = int(dy_values[i].item())
        actions.append([a_type, dx, dy])

    actions = np.array(actions, dtype=int)  # shape (max_units, 3)
    return actions


def select_action_dqn(agent_id: int, state: np.ndarray, max_units: int):
    eps = get_epsilon(agent_id)
    new_eps = max(EPS_END, eps - EPS_DECAY)
    set_epsilon(agent_id, new_eps)

    qnet, _, _, _, _, _ = get_globals(agent_id)
    if random.random() < eps:
        # Generate a random array with shape (max_units, 3).
        # First column in range [0, 5], next two columns in range [-10, 10].
        random_col0 = np.random.randint(0, 6, size=(max_units, 1))     # shape (max_units, 1)
        random_col12 = np.random.randint(-10, 11, size=(max_units, 2)) # shape (max_units, 2)

        random_actions = np.concatenate([random_col0, random_col12], axis=1)  # shape (max_units, 3)
        return random_actions
    else:
        with torch.no_grad():
            actions = select_actions_from_qnet(qnet, state, max_units)
            return actions


class DQNAgent1():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.team_id = 0 if self.player == "player_0" else 1

        # Flatten the config so we can incorporate it in our net inputs.
        self.cfg_array = flatten_config(env_cfg)

        self.env_cfg = env_cfg
        self.max_units = env_cfg["max_units"]
        self.action_size = 3

        # OLD: self.obs_size = 100
        # We now do:
        self.obs_size = 100
        self.cfg_size = len(self.cfg_array)  # = 10
        self.input_size = self.obs_size + self.cfg_size

        # Initialize the global DQN model with the new input size
        maybe_init_globals(1, self.input_size, self.action_size, self.max_units)

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
                (self.last_obs, self.last_action, float(reward), current_state, float(done))
            )
            compute_q_loss(1)

        actions = select_action_dqn(1, current_state, self.max_units)

        self.last_obs = current_state
        self.last_action = actions

        return actions


class DQNAgent2():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.team_id = 0 if self.player == "player_0" else 1

        # Flatten config
        self.cfg_array = flatten_config(env_cfg)
        self.env_cfg = env_cfg

        self.max_units = env_cfg["max_units"]
        self.action_size = 3

        # Same input dimension as agent1
        self.obs_size = 100
        self.cfg_size = len(self.cfg_array)
        self.input_size = self.obs_size + self.cfg_size

        maybe_init_globals(2, self.input_size, self.action_size, self.max_units)

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
                (self.last_obs, self.last_action, float(reward), current_state, float(done))
            )
            compute_q_loss(2)

        actions = select_action_dqn(1, current_state, self.max_units)

        self.last_obs = current_state
        self.last_action = actions

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
        last_reward_0 = np.array(0)
        last_reward_1 = np.array(0)

        while not game_done:
            # Agents produce actions
            act_0 = player_0.act(
                step=step,
                obs=obs["player_0"],
                reward=last_reward_0.item(),
                done=game_done
            )
            act_1 = player_1.act(
                step=step,
                obs=obs["player_1"],
                reward=last_reward_1.item(),
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
            reward=last_reward_0.item(),
            done=game_done
        )
        player_1.act(
            step=step,
            obs=obs["player_1"],
            reward=last_reward_1.item(),
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
