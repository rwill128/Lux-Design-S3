import jax
import jax.numpy as jnp
import numpy as np

from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams, env_params_ranges
from luxai_s3.utils import to_numpy

# -------------------------------------------------------
# Step 1: Define the environment params and environment
# -------------------------------------------------------
def sample_env_params(rng_key):
    randomized_game_params = {}
    for k, v in env_params_ranges.items():
        rng_key, subkey = jax.random.split(rng_key)
        randomized_game_params[k] = jax.random.choice(subkey, jnp.array(v)).item()
    return EnvParams(**randomized_game_params), rng_key

env = LuxAIS3Env(auto_reset=False)

# -------------------------------------------------------
# Step 2: Define a baseline policy (pure function)
# -------------------------------------------------------
def baseline_policy(obs, rng):
    max_units = obs["player_0"].units.position.shape[1]

    rng, subkey = jax.random.split(rng)
    actions_p0 = jax.random.randint(subkey, (max_units,), minval=0, maxval=5)
    rng, subkey = jax.random.split(rng)
    actions_p1 = jax.random.randint(subkey, (max_units,), minval=0, maxval=5)

    # Define a lookup table for actions:
    # action_idx: 0=stay,1=up,2=right,3=down,4=left
    # Corresponding (action_code, dx, dy) arrays:
    action_lut = jnp.array([
        [0, 0, 0],   # stay
        [1, 0, -1],  # up
        [2, 1, 0],   # right
        [3, 0, 1],   # down
        [4, -1, 0]   # left
    ], dtype=jnp.int32)

    # Use vmap to index into action_lut:
    actions_p0_lux = action_lut[actions_p0]
    actions_p1_lux = action_lut[actions_p1]

    combined_action = {
        "player_0": actions_p0_lux,
        "player_1": actions_p1_lux
    }
    return combined_action, rng

# -------------------------------------------------------
# Step 3: Data collection loop
# -------------------------------------------------------
def collect_data(num_episodes=5000):
    rng = jax.random.PRNGKey(0)
    dataset = []

    for ep in range(num_episodes):
        # Sample environment params
        env_params, rng = sample_env_params(rng)
        rng, reset_key = jax.random.split(rng)
        obs, state = env.reset(reset_key, params=env_params)

        done = False
        episode_data = []
        terminated = False
        truncated = False
        while not (terminated or truncated):
            rng, subkey = jax.random.split(rng)
            action, rng = baseline_policy(obs, rng)
            rng, step_key = jax.random.split(rng)
            next_obs, next_state, reward, terminated, truncated, info = env.step(
                step_key, state, action, env_params
            )

            # Convert jax arrays to numpy for storage if desired
            # They might be dictionaries with arrays, just store as-is or convert
            step_data = {
                "obs": obs,
                "action": action["player_0"][0],
                "reward": reward,
                "next_obs": next_obs,
                "done": terminated["player_0"] or truncated["player_0"]
            }
            episode_data.append(step_data)

            obs = next_obs
            state = next_state

            if step_data["done"]:
                break
        dataset.append(episode_data)
    return dataset

dataset = collect_data(num_episodes=5000)

# -------------------------------------------------------
# Step 4: Offline Training (Example: Behavior Cloning)
# -------------------------------------------------------
# Suppose we want to do a simple behavior cloning step:
# Extract (obs, action) pairs from dataset and train a policy network to predict action from obs.

def prepare_bc_data(dataset):
    # Flatten all episodes into a big list of transitions
    all_transitions = [t for ep in dataset for t in ep]
    # Extract player_0 obs and actions
    # Suppose obs["player_0"] contains a suitable representation
    # You might want to process obs into features: obs_features
    obs_list = []
    act_list = []
    for transition in all_transitions:
        # Extract a meaningful feature vector from transition["obs"]["player_0"]
        # This depends heavily on the structure of the obs.
        obs_vec = process_obs(transition["obs"]["player_0"])
        # Extract action for player_0:
        # The action is an array (max_units, 3)
        action_vec = transition["action"]["player_0"]
        obs_list.append(obs_vec)
        act_list.append(action_vec)
    obs_array = np.array(obs_list)
    act_array = np.array(act_list)
    return obs_array, act_array

def process_obs(obs):
    # Transform obs dict into a vector or something suitable for a model
    # Just a placeholder:
    return obs.units.position[0] # or something like that

obs_array, act_array = prepare_bc_data(dataset)

# Now we have obs_array and act_array. We can use a supervised training approach:
# Train a neural network f_theta(obs) ~ action using standard supervised learning.

# Pseudocode for supervised training with JAX/Optax:
import optax
from flax import linen as nn

class PolicyNetwork(nn.Module):
    # define your network
    @nn.compact
    def __call__(self, x):
        # x: [batch, feature_dim]
        # Output: [batch, max_units, action_dim (if discrete, a logits vector)]
        # Just a placeholder linear:
        dense_layer = nn.Dense(5)
        # Apply it per unit using vmap over the second dimension (units):
        # in_axes=1 means we map over the 'max_units' dimension
        return jax.vmap(dense_layer, in_axes=1, out_axes=1)(x)

# Initialize model
feature_dim = obs_array.shape[-1] # depends on your obs processing
model = PolicyNetwork()
rng = jax.random.PRNGKey(42)
params = model.init(rng, obs_array[0]) # init with a single example

# Define loss and update:
def loss_fn(params, batch_obs, batch_act):
    # batch_obs: (batch, max_units, feature_dim_per_unit)
    # batch_act: (batch, max_units)
    logits = model.apply(params, batch_obs)  # (batch, max_units, 5)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot_actions = jax.nn.one_hot(batch_act, 5)  # (batch, max_units, 5)
    loss = -jnp.mean(jnp.sum(one_hot_actions * log_probs, axis=-1))
    # sum over action dim -> (batch, max_units)
    # mean over both batch and units implicitly by jnp.mean
    return loss

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# @jax.jit
def update(params, opt_state, batch_obs, batch_act):
    grads = jax.grad(loss_fn)(params, batch_obs, batch_act)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Now do a simple training loop:
for epoch in range(10):
    # Shuffle and batch dataset:
    indices = np.arange(len(obs_array))
    np.random.shuffle(indices)
    batch_size = 64
    for start in range(0, len(indices), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        batch_obs = obs_array[batch_idx]
        batch_act = act_array[batch_idx]
        params, opt_state = update(params, opt_state, batch_obs, batch_act)

# After training, params represent a policy that approximates the baseline player's behavior.
# You can now run a new round of data collection using this improved policy or refine it further.

