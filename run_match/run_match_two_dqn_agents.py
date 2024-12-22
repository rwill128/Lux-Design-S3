import os
import numpy as np
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

#########################
#  BEGIN DQN UTILITIES  #
#########################

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Hyperparameters (tweak as needed)
LR = 1e-3
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1e-4
REPLAY_SIZE = 10_000
BATCH_SIZE = 32

# Because the environment re-instantiates each agent on every new game,
# we store all DQN "globals" here so they persist across games.
# Each agent has its own network, memory, epsilon, etc.

global_qnet_1 = None
global_optim_1 = None
global_replay_buffer_1 = None
global_epsilon_1 = EPS_START

global_qnet_2 = None
global_optim_2 = None
global_replay_buffer_2 = None
global_epsilon_2 = EPS_START

def maybe_init_globals(agent_id: int, state_size: int, action_size: int):
    """
    Checks if the global Q network and replay buffers have been initialized
    for the given agent. If not, initialize them.
    """
    global global_qnet_1, global_optim_1, global_replay_buffer_1, global_epsilon_1
    global global_qnet_2, global_optim_2, global_replay_buffer_2, global_epsilon_2

    # Simple fully connected MLP for demonstration.
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
    """
    Helper to retrieve the QNet, optimizer, replay buffer, etc. for each agent.
    """
    if agent_id == 1:
        return (
            global_qnet_1, global_optim_1, global_replay_buffer_1,
            lambda: None,  # No target network in this minimal example
            lambda: None,  # No target update in this minimal example
            lambda: None,  # No scheduled separate target update
        )
    else:
        return (
            global_qnet_2, global_optim_2, global_replay_buffer_2,
            lambda: None,
            lambda: None,
            lambda: None,
        )

def get_epsilon(agent_id: int):
    """Retrieve global epsilon."""
    global global_epsilon_1, global_epsilon_2
    return global_epsilon_1 if agent_id == 1 else global_epsilon_2

def set_epsilon(agent_id: int, val: float):
    """Set global epsilon."""
    global global_epsilon_1, global_epsilon_2
    if agent_id == 1:
        global_epsilon_1 = val
    else:
        global_epsilon_2 = val

def store_transition(agent_id: int, transition):
    """Push a transition (s, a, r, s', done) into the replay buffer."""
    _, _, replay_buffer, _, _, _ = get_globals(agent_id)
    replay_buffer.append(transition)

def sample_transitions(agent_id: int):
    """Sample a mini-batch from the replay buffer."""
    _, _, replay_buffer, _, _, _ = get_globals(agent_id)
    batch = random.sample(replay_buffer, BATCH_SIZE)
    return batch

def compute_q_loss(agent_id: int):
    """
    Perform one gradient update step using a sample from the replay buffer.
    This is the simplest DQN style update:
      Q(s,a) = r + gamma * max_a'[Q(s', a')]
    """
    qnet, optim_, replay_buffer, _, _, _ = get_globals(agent_id)

    if len(replay_buffer) < BATCH_SIZE:
        return  # not enough data to train

    batch = sample_transitions(agent_id)

    # Each transition is (s, a, r, s_next, done)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    # Current Q
    q_values = qnet(states)               # (batch_size, action_dim)
    q_val_a = q_values.gather(1, actions.view(-1, 1)).squeeze(-1)  # Q(s,a)

    # Next Q
    with torch.no_grad():
        q_next = qnet(next_states)  # no separate target network, purely illustrative
        q_next_max = q_next.max(dim=1)[0]

    # Target
    target = rewards + GAMMA * q_next_max * (1 - dones)

    loss = nn.MSELoss()(q_val_a, target)

    optim_.zero_grad()
    loss.backward()
    optim_.step()

def select_action_dqn(agent_id: int, state: np.ndarray, action_size: int):
    """
    Epsilon-greedy action selection from Q network.
    We treat the entire 'action_size' as a single integer for demonstration.
    In practice, you might have to produce multi-dimensional actions
    (like shape = (max_units, 3)).
    """
    eps = get_epsilon(agent_id)

    # Epsilon-decay
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

#########################
#   END DQN UTILITIES   #
#########################

# Below is your original skeleton code for the environment, lightly adapted
# to incorporate the DQN logic.

class DQNAgent1():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.team_id = 0 if self.player == "player_0" else 1

        self.env_cfg = env_cfg
        self.max_units = env_cfg["max_units"]
        # Suppose each "unit" has 3 possible integer-coded actions => total 3 * max_units
        # But for simplicity, let's define a smaller discrete action space
        # (like 10 possible combos, purely for demonstration).
        # A real solution would handle (max_units x 3) properly.
        self.action_size = 10

        # For demonstration, we define a naive flattened observation size.
        # (Again, real usage will vary a lot.)
        self.obs_size = 100  # Arbitrary dimension for the demonstration

        # Make sure the global model is initialized
        maybe_init_globals(1, self.obs_size, self.action_size)

        # placeholders for storing the last obs/action so we can store transitions
        self.last_obs = None
        self.last_action = None

    def flatten_observation(self, obs):
        """
        Flatten the entire observation into a fixed-size vector for the DQN.
        This is VERY naive. In practice, you'd design a better feature extractor.
        """
        # Example: we just flatten everything we can find into one vector (capped or padded).
        # shape (T,N,2), (T,N,1), etc. We only take our team's data.
        # This is purely for demonstration!
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
        # Just pad or truncate to self.obs_size
        if len(features) >= self.obs_size:
            return features[: self.obs_size]
        else:
            return np.pad(features, (0, self.obs_size - len(features)))

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        In a standard RL loop, you might want to handle reward shaping, etc.
        For demonstration, let's assume a naive reward = team_points[self.team_id] delta.
        We'll do a single-step Q-learning style update each time `act` is called.
        """
        # "obs" is the raw data from the environment.
        # Flatten to create our DQN state:
        current_state = self.flatten_observation(obs)

        # Very naive reward shaping: difference in team_points from last to current
        team_points = np.array(obs["team_points"])[self.team_id]
        # Suppose we define reward to be just the current score (extremely naive).
        reward = float(team_points)

        # If we had a previous observation, we store the transition
        if self.last_obs is not None and self.last_action is not None:
            done = False  # If your environment says it's done, you'd supply that here
            store_transition(
                1,
                (
                    self.last_obs,
                    self.last_action,
                    reward,
                    current_state,
                    float(done),
                ),
            )
            # do a training step with the replay buffer
            compute_q_loss(1)

        # Choose an action
        action_idx = select_action_dqn(1, current_state, self.action_size)

        # The environment expects shape (max_units, 3). We don't have a real policy
        # that picks an action for each unit. Instead, we'll decode action_idx
        # arbitrarily for demonstration.
        # We'll just produce zeros for all units, and attempt to inject something
        # that depends on action_idx in the first unit slot as an example:
        actions = np.zeros((self.max_units, 3), dtype=int)

        # For example, let "action_idx % 3" be the direction, and "action_idx // 3" be the magnitude
        # This is arbitrary for demonstration.
        direction = action_idx % 3  # 0 to 2
        magnitude = (action_idx // 3)
        actions[0, 0] = direction
        actions[0, 1] = magnitude
        actions[0, 2] = 0  # always zero, just a placeholder

        # store current observation and action for next step
        self.last_obs = current_state
        self.last_action = action_idx

        return actions


class DQNAgent2():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.team_id = 0 if self.player == "player_0" else 1

        self.env_cfg = env_cfg
        self.max_units = env_cfg["max_units"]
        self.action_size = 10  # same as agent1
        self.obs_size = 100    # same as agent1

        # Make sure the global model is initialized for agent2
        maybe_init_globals(2, self.obs_size, self.action_size)

        self.last_obs = None
        self.last_action = None

    def flatten_observation(self, obs):
        # Same naive flattening approach
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
            return features[: self.obs_size]
        else:
            return np.pad(features, (0, self.obs_size - len(features)))

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # Flatten obs
        current_state = self.flatten_observation(obs)

        # define naive reward
        team_points = np.array(obs["team_points"])[self.team_id]
        reward = float(team_points)

        if self.last_obs is not None and self.last_action is not None:
            done = False
            store_transition(
                2,
                (
                    self.last_obs,
                    self.last_action,
                    reward,
                    current_state,
                    float(done),
                ),
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
        # Reset the environment for each game
        obs, info = env.reset(seed=seed + i)  # changing seed each game
        env_cfg = info["params"]  # game parameters that agents can see

        player_0 = agent_1_cls("player_0", env_cfg)
        player_1 = agent_2_cls("player_1", env_cfg)

        game_done = False
        step = 0
        print(f"Running game {i + 1}/{games_to_play}")
        while not game_done:
            actions = {
                "player_0": player_0.act(step=step, obs=obs["player_0"]),
                "player_1": player_1.act(step=step, obs=obs["player_1"]),
            }

            obs, reward, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] or truncated[k] for k in terminated}
            if dones["player_0"] or dones["player_1"]:
                game_done = True
            step += 1

    env.close()  # saves the replay of the last game and frees resources
    print(f"Finished {games_to_play} games. Replays saved to {replay_save_dir}")


if __name__ == "__main__":
    # Run evaluation with the DQN Agents against each other
    evaluate_agents(
        DQNAgent1,
        DQNAgent2,
        games_to_play=20,
        seed=2,
        replay_save_dir="replays/" + DQNAgent1.__name__ + "_" + DQNAgent2.__name__
    )
    print("Done.")
