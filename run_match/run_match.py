import os
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

from agents.baseline_agent.baselineagent import BaselineAgent

from scipy.optimize import linear_sum_assignment
import numpy as np
from collections import deque

class RelicHuntingShootingAgent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        # Points and relic data
        self.last_team_points = 0
        self.relic_allocation = 20
        self.relic_tile_data = {}

        # Memory
        self.last_unit_positions = []  # store positions of units from previous turn
        self.end_of_match_printed = False

        # If we want to keep it very simple, no tester tile logic:
        # Just rely on global inference of newly discovered reward tiles
        # self.current_tester_tile = None
        # self.current_tester_tile_relic = None
        # Instead, we just pick a new untested tile each turn if available.

    def update_tile_results(self, current_points, obs):
        # Compute how many known reward tiles were occupied last turn
        baseline = 0
        for (x, y) in self.last_unit_positions:
            for relic_pos, tiles_data in self.relic_tile_data.items():
                if (x, y) in tiles_data and tiles_data[(x, y)]["reward_tile"]:
                    baseline += 1

        gain = current_points - self.last_team_points
        extra = gain - baseline
        if extra <= 0:
            # No new information
            return

        # Identify unknown tiles occupied last turn
        unknown_tiles_occupied = []
        for (x, y) in self.last_unit_positions:
            for relic_pos, tiles_data in self.relic_tile_data.items():
                if (x, y) in tiles_data:
                    tile_info = tiles_data[(x, y)]
                    if not tile_info["tested"] and not tile_info["reward_tile"]:
                        unknown_tiles_occupied.append((relic_pos, (x, y)))

        # Assign reward_tile = True to some unknown tiles
        if len(unknown_tiles_occupied) > 0:
            selected = unknown_tiles_occupied[:extra]
            for relic_pos, tile_pos in selected:
                self.relic_tile_data[relic_pos][tile_pos]["reward_tile"] = True
                self.relic_tile_data[relic_pos][tile_pos]["tested"] = True
            # Extra unknown tiles remain untested

    def select_tiles_for_relic(self, relic_pos):
        reward_tiles = [t for t, d in self.relic_tile_data[relic_pos].items() if d["reward_tile"]]
        untested_tiles = [t for t, d in self.relic_tile_data[relic_pos].items() if not d["tested"]]
        return reward_tiles, untested_tiles

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        # unit_energy = np.array(obs["units"]["energy"][self.team_id])

        available_unit_ids = np.where(unit_mask)[0]
        num_units = len(available_unit_ids)

        current_team_points = obs["team_points"][self.team_id]

        # Update global inference of reward tiles
        self.update_tile_results(current_team_points, obs)

        # Discover visible relic nodes and init tile data
        relic_nodes_mask = obs["relic_nodes_mask"]
        relic_nodes_positions = obs["relic_nodes"][relic_nodes_mask]
        for (rx, ry) in relic_nodes_positions:
            if (rx, ry) not in self.relic_tile_data:
                self.relic_tile_data[(rx, ry)] = {}
                for bx in range(rx-2, rx+3):
                    for by in range(ry-2, ry+3):
                        if 0 <= bx < self.env_cfg["map_width"] and 0 <= by < self.env_cfg["map_height"]:
                            self.relic_tile_data[(rx, ry)][(bx, by)] = {"tested": False, "reward_tile": False}

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # Assign units to relic tasks
        all_known_relic_positions = list(self.relic_tile_data.keys())

        for (rx, ry) in all_known_relic_positions:
            reward_tiles, untested_tiles = self.select_tiles_for_relic((rx, ry))

            # Place units on known reward tiles first
            for tile in reward_tiles:
                if len(available_unit_ids) == 0:
                    break
                u = available_unit_ids[0]
                available_unit_ids = available_unit_ids[1:]
                ux, uy = obs["units"]["position"][self.team_id][u]
                tx, ty = tile
                # Move simple: if not on tile, step closer horizontally or vertically
                actions[u] = self.simple_move((ux, uy), (tx, ty))

            # Test an untested tile if available and units remain
            if untested_tiles and len(available_unit_ids) > 0:
                test_tile = untested_tiles[0]
                u = available_unit_ids[0]
                available_unit_ids = available_unit_ids[1:]
                ux, uy = obs["units"]["position"][self.team_id][u]
                tx, ty = test_tile
                actions[u] = self.simple_move((ux, uy), (tx, ty))
                # Not tracking current_tester_tile or expected_baseline_gain now for simplicity

        # Any remaining units not assigned do nothing
        for u in available_unit_ids:
            actions[u] = [0,0,0]

        # Update last_team_points for next turn
        self.last_team_points = current_team_points

        # Record current positions as last_unit_positions
        self.last_unit_positions = []
        for uid in np.where(obs["units_mask"][self.team_id])[0]:
            ux, uy = obs["units"]["position"][self.team_id][uid]
            self.last_unit_positions.append((ux, uy))

        # Print known reward tiles at end of match
        if obs["match_steps"] == 100 and not self.end_of_match_printed:
            all_reward_tiles = []
            for relic_pos, tiles_data in self.relic_tile_data.items():
                for tile_pos, tile_info in tiles_data.items():
                    if tile_info["reward_tile"]:
                        all_reward_tiles.append((relic_pos, tile_pos))
            print("Known reward tiles at end of match:")
            for relic_pos, tile_pos in all_reward_tiles:
                print(f"Relic: {relic_pos}, Reward Tile: {tile_pos}")
            self.end_of_match_printed = True

        return actions

    def simple_move(self, from_pos, to_pos):
        # Very simple movement: pick one axis to move along if needed.
        fx, fy = from_pos
        tx, ty = to_pos
        dx = tx - fx
        dy = ty - fy
        # Example action encoding: 1=UP,2=RIGHT,3=DOWN,4=LEFT,0=STAY
        if abs(dx) > abs(dy):
            return np.array([2 if dx > 0 else 4, 0, 0])
        elif dy != 0:
            return np.array([3 if dy > 0 else 1, 0, 0])
        else:
            return np.array([0,0,0])

def evaluate_agents(agent_1_cls, agent_2_cls, seed=42, games_to_play=3, replay_save_dir="replays"):
    # Ensure the replay directory exists
    os.makedirs(replay_save_dir, exist_ok=True)

    # Create an environment wrapped to record episodes
    env = RecordEpisode(
        LuxAIS3GymEnv(numpy_output=True), save_on_close=True, save_on_reset=True, save_dir=replay_save_dir
    )

    for i in range(games_to_play):
        # Reset the environment for each game
        obs, info = env.reset(seed=seed+i)  # changing seed each game
        env_cfg = info["params"]  # game parameters that agents can see

        player_0 = agent_1_cls("player_0", env_cfg)
        player_1 = agent_2_cls("player_1", env_cfg)

        game_done = False
        step = 0
        print(f"Running game {i+1}/{games_to_play}")
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
    # Run evaluation with the dummy Agent against itself
    evaluate_agents(BaselineAgent, RelicHuntingShootingAgent, games_to_play=3,
                    replay_save_dir="replays/" + BaselineAgent.__name__ + "_" + RelicHuntingShootingAgent.__name__)

    # After running, you can check the "replays" directory for saved replay files.
    # You can set breakpoints anywhere in this file or inside the Agent class.
