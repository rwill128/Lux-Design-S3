import json
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Union, Any

# Allow printing for debugging
import os, sys
os.environ['PYTHONWARNINGS'] = 'ignore'

def direction_to(src, target):
    """Convert to JAX arrays and compute direction using pure JAX operations"""
    src = jnp.asarray(src, dtype=jnp.int16)
    target = jnp.asarray(target, dtype=jnp.int16)
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    
    # Use JAX's where for conditional logic
    direction = jnp.where(
        (dx == 0) & (dy == 0), 0,
        jnp.where(
            jnp.abs(dx) > jnp.abs(dy),
            jnp.where(dx > 0, 2, 4),
            jnp.where(dy > 0, 3, 1)
        )
    )
    
    return direction

class BaselineAgent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.key = jax.random.PRNGKey(0)
        self.env_cfg = env_cfg

        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        try:
            # Initialize actions array with zeros for movement and sap deltas
            actions = np.zeros((self.env_cfg["max_units"], 3), dtype=np.int16)
            
            # Get units for this player
            if "units" not in obs or self.player not in obs["units"]:
                return actions
                
            units = obs["units"][self.player]
            if not units:
                return actions
                
            # Get relic positions
            relic_positions = []
            if "relics" in obs:
                for relic in obs["relics"]:
                    if "pos" in relic:
                        relic_positions.append([relic["pos"]["x"], relic["pos"]["y"]])
            
            # Convert relic positions to numpy array
            if relic_positions:
                relic_positions = np.array(relic_positions, dtype=np.int16)
            else:
                relic_positions = np.zeros((0, 2), dtype=np.int16)

            # Process each unit
            for unit_id, unit in units.items():
                unit_pos = jnp.array([unit["pos"]["x"], unit["pos"]["y"]], dtype=jnp.int16)
                unit_idx = int(unit_id)  # Convert unit_id to integer index
                
                if len(relic_positions) > 0:
                    # Find nearest relic
                    nearest_relic_pos = relic_positions[0]  # Just use first relic for now
                    manhattan_distance = np.sum(np.abs(unit_pos - nearest_relic_pos))
                    
                    # If close to relic, stay and sap
                    if manhattan_distance <= 4: 
                        sap_dx = np.random.randint(-2, 3, dtype=np.int16)
                        sap_dy = np.random.randint(-2, 3, dtype=np.int16)
                        action = np.array([0, sap_dx, sap_dy], dtype=np.int16)
                    else:
                        # Move towards relic
                        direction = direction_to(unit_pos, nearest_relic_pos)
                        action = np.array([direction, 0, 0], dtype=np.int16)
                else:
                    # Random exploration
                    direction = np.random.randint(1, 5, dtype=np.int16)  # Random direction 1-4
                    action = np.array([direction, 0, 0], dtype=np.int16)
                
                actions[unit_idx] = action

            # Ensure final actions array is a JAX array
            return jnp.asarray(actions, dtype=jnp.int16)

        except Exception as e:
            # Return empty action on error but don't print
            return jnp.zeros((self.env_cfg["max_units"], 3), dtype=jnp.int16)

def from_json(state):
    if isinstance(state, (list, np.ndarray)):
        # Convert to numpy first to handle nested lists properly
        arr = np.asarray(state)
        # Determine appropriate dtype based on content
        if np.issubdtype(arr.dtype, np.integer):
            return jnp.asarray(arr, dtype=jnp.int16)
        elif np.issubdtype(arr.dtype, np.floating):
            return jnp.asarray(arr, dtype=jnp.float32)
        elif np.issubdtype(arr.dtype, np.bool_):
            return jnp.asarray(arr, dtype=jnp.bool_)
        else:
            # For other types, try to convert to int16 if possible
            try:
                return jnp.asarray(arr, dtype=jnp.int16)
            except:
                return jnp.asarray(arr)
    elif isinstance(state, dict):
        out = {}
        for k in state:
            out[k] = from_json(state[k])
        return out
    elif isinstance(state, (int, np.integer)):
        return jnp.asarray(state, dtype=jnp.int16)
    elif isinstance(state, (float, np.floating)):
        return jnp.asarray(state, dtype=jnp.float32)
    elif isinstance(state, (bool, np.bool_)):
        return jnp.asarray(state, dtype=jnp.bool_)
    else:
        try:
            # Try to convert unknown types to int16 if possible
            return jnp.asarray(state, dtype=jnp.int16)
        except:
            return state

### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = dict() # store potentially multiple dictionaries as kaggle imports code directly
agent_prev_obs = dict()
# Initialize global state
agent_dict = {}
agent_prev_obs = {}

def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    try:
        global agent_dict
        global agent_prev_obs
        
        # Handle observation format
        if hasattr(observation, 'obs'):
            # Kaggle format
            obs = observation.obs
            if isinstance(obs, str):
                obs = json.loads(obs)
            step = observation.step
            player = observation.player
            remainingOverageTime = observation.remainingOverageTime
        else:
            # Local testing format
            obs = observation
            step = 0  # Default to 0 if not provided
            player = "player_0"  # Default to player_0 if not provided
            remainingOverageTime = 60  # Default to 60 if not provided
            
        # Set default configurations if None
        if configurations is None or "env_cfg" not in configurations:
            configurations = {
                "env_cfg": {
                    "max_units": 100,  # Default max units
                    "map_size": 64,    # Default map size
                    "max_episode_length": 1000  # Default episode length
                }
            }
            
        # Initialize agent if needed
        if step == 0 or player not in agent_dict:
            agent_dict[player] = BaselineAgent(player, configurations["env_cfg"])
            
        agent = agent_dict[player]
        
        # Convert observation to the format expected by the agent
        processed_obs = from_json(obs)
        
        # Get actions from agent
        actions = agent.act(step, processed_obs, remainingOverageTime)
        
        # Ensure actions is a numpy array with correct dtype
        max_units = configurations["env_cfg"]["max_units"]
        if actions is None:
            actions = np.zeros((max_units, 3), dtype=np.int16)
        else:
            actions = np.asarray(actions, dtype=np.int16)
            
        empty_actions = np.zeros((max_units, 3), dtype=np.int16)
        
        # Store observation for next step
        agent_prev_obs[player] = obs
        
        # Return dictionary with numpy arrays
        return {
            "player_0": actions if player == "player_0" else empty_actions,
            "player_1": actions if player == "player_1" else empty_actions
        }
    except Exception as e:
        # Return empty actions for both players on error
        max_units = 100  # Default if configurations is not available
        if configurations and "env_cfg" in configurations:
            max_units = configurations["env_cfg"]["max_units"]
            
        empty_actions = np.zeros((max_units, 3), dtype=np.int16)
        return {
            "player_0": empty_actions,
            "player_1": empty_actions
        }
