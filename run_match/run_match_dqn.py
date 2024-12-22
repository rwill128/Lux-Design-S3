import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

##########################################
# GLOBALS for storing networks and replay buffer
##########################################
# Because the agents will be re-instantiated each game, we keep these in the global scope.

# Hyperparameters (toy values; adjust as needed)
LR = 1e-3
GAMMA = 0.99
EPS_START = 0.95   # initial exploration probability
EPS_END = 0.01     # final exploration probability
EPS_DECAY = 0.9995 # decay per step
BATCH_SIZE = 32
MAX_MEMORY_SIZE = 20000

# We'll define a small feedforward network for Q-values:
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        # Very simple two-layer MLP for demonstration
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# We define a global Q-network (for local updates) and optionally a target network
# For simplicity, we'll just use one Q-network in this minimal example
if not hasattr(globals(), "global_q_network"):
    # The input dimension is environment-specific. We'll guess something small for demonstration.
    # The output dimension is now 6 to represent [0..5].
    input_dim = 20  # Example dimension; adapt to the actual shape of your state
    output_dim = 6  # 6 possible discrete actions: 0..5

    global_q_network = QNetwork(input_dim, output_dim)
    global_optimizer = optim.Adam(global_q_network.parameters(), lr=LR)
    global_replay_buffer = deque(maxlen=MAX_MEMORY_SIZE)

# We also keep track of a global step count to decay epsilon
if not hasattr(globals(), "global_step"):
    global_step = 0
else:
    global_step = globals()["global_step"]

if not hasattr(globals(), "epsilon"):
    epsilon = EPS_START
else:
    epsilon = globals()["epsilon"]

##########################################
# DQN AGENT
##########################################
class DQNAgent:
    def __init__(self, player: str, env_cfg: dict):
        """
        player: "player_0" or "player_1"
        env_cfg: dictionary with environment config (including "max_units")
        """
        self.player = player
        self.env_cfg = env_cfg

        # These references point to the same objects in global scope
        self.q_network = global_q_network
        self.optimizer = global_optimizer
        self.replay_buffer = global_replay_buffer

        # Epsilon for exploration
        self.epsilon = epsilon

        # We assume there's a constant number of units = env_cfg["max_units"] for each player
        self.num_units = self.env_cfg["max_units"]

    def _flatten_observation(self, obs):
        """
        Flatten the observation into a 1D numpy array or Torch tensor.
        This is a placeholder. You should adapt it to your real obs structure.

        Example:
          obs["units"]["position"], obs["units"]["energy"], obs["sensor_mask"], ...
        For demonstration, we'll just produce random floats from obs to emulate flattening.
        """
        flattened_size = 20  # must match QNetwork's input_dim
        return np.random.random(flattened_size).astype(np.float32)

    def _choose_unit_action(self, q_values):
        """
        Given the Q-values for (num_units) * 6 actions, choose discrete action for each unit.
        q_values: shape (num_units, 6)
        returns a np array shape (num_units,) of discrete actions in [0..5]
        """
        actions = []
        for unit_q in q_values:
            if random.random() < self.epsilon:
                # Explore
                action = random.randint(0, 5)
            else:
                # Exploit
                action = int(torch.argmax(unit_q).item())
            actions.append(action)
        return np.array(actions)

    def act(self, step, obs, reward=0.0, done=False):
        """
        Returns actions in shape: (self.env_cfg["max_units"], 3) dtype=int

        - step: current timestep
        - obs: your observation structure
        - reward: reward at this step (for online learning)
        - done: whether the episode is done
        """
        # Flatten the obs
        state_vec = self._flatten_observation(obs)

        # Convert to torch
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)  # shape (1, input_dim)

        # We'll do one forward pass per unit for simplicity (though you can do more advanced approaches).
        q_values_units = []
        for _ in range(self.num_units):
            with torch.no_grad():
                q_values_unit = self.q_network(state_tensor)  # shape (1, 6)
            q_values_units.append(q_values_unit[0])
        q_values_units = torch.stack(q_values_units, dim=0)  # shape (num_units, 6)

        # Choose discrete actions in [0..5] for each unit
        chosen_actions = self._choose_unit_action(q_values_units)

        # For training, store transitions if we have a previous state, action
        # (not shown fully here). Then do a train step.
        self._train_step()

        # Build the final action shape: (max_units, 3)
        # Action layout: [x, y, z]
        #   - x in [0..5]
        #   - y, z = 0 unless x == 5, in which case y,z might be used to indicate the sap direction
        final_actions = np.zeros((self.num_units, 3), dtype=int)

        for i, a in enumerate(chosen_actions):
            # set x
            final_actions[i][0] = a
            if a == 5:
                # For the sap action, pick a delta x,y from e.g. [-1,0,1] except (0,0).
                # If you'd prefer always [5,0,0], just comment out the random part.
                possible_deltas = [(dx, dy) for dx in [-1,0,1] for dy in [-1,0,1] if not (dx==0 and dy==0)]
                # pick random delta
                dx, dy = random.choice(possible_deltas)
                final_actions[i][1] = dx
                final_actions[i][2] = dy
            else:
                # For all other actions, [0..4], we keep y=0, z=0
                final_actions[i][1] = 0
                final_actions[i][2] = 0

        return final_actions

    def _train_step(self):
        global global_step
        global epsilon

        # We can do an epsilon decay here
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        self.epsilon = epsilon
        global_step += 1

        # If there's not enough data in the replay buffer, skip training
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        # Example of how you'd sample a batch for training:
        batch = random.sample(self.replay_buffer, BATCH_SIZE)

        # Usually you'd parse the batch into (states, actions, rewards, next_states, dones)
        # Then something like:
        #   q_values = self.q_network(states)
        #   q_value = q_values.gather(1, actions)
        #   next_q_values = self.q_network(next_states).max(1)[0]
        #   target = rewards + GAMMA * next_q_values * (1 - dones)
        #   loss = MSE(q_value, target)
        #   backprop
        #
        # We'll skip a real training routine here for brevity.

##########################################
# EVALUATION LOOP
##########################################
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

    env.close()
    print(f"Finished {games_to_play} games. Replays (if any) saved to {replay_save_dir}")


if __name__ == "__main__":
    # Run evaluation with the same RL agent against itself
    evaluate_agents(DQNAgent, DQNAgent, games_to_play=5000, seed=2,
                    replay_save_dir="replays/DQN_selfplay")

