import logging
from abc import ABC, abstractmethod
from typing import NamedTuple, Tuple, Dict

import numpy as np
from scipy.stats import rankdata

from lux_ai.lux.game import Game
from lux_ai.lux.game_objects import Player


class RewardSpec(NamedTuple):
    """
    Specification for a reward space defining its key characteristics.

    Attributes:
        reward_min: Minimum possible reward value
        reward_max: Maximum possible reward value
        zero_sum: Whether rewards sum to zero across players
        only_once: Whether reward is given once (True) or can repeat (False)

    This specification helps ensure reward spaces are properly bounded and
    behave consistently with respect to the learning algorithm's expectations.
    """
    reward_min: float  # Minimum possible reward value
    reward_max: float  # Maximum possible reward value
    zero_sum: bool    # Whether rewards sum to zero across players
    only_once: bool   # Whether reward is given once or can repeat


class BaseRewardSpace(ABC):
    """
    Abstract base class for defining reward spaces in the Lux environment.

    This class provides the interface for:
    1. Reward Calculation:
       - Computing rewards for each player
       - Determining episode termination
       - Handling both full game and subtask rewards

    2. Reward Specification:
       - Defining reward bounds
       - Specifying zero-sum properties
       - Indicating one-time vs repeating rewards

    3. State Information:
       - Accessing game state
       - Tracking progress
       - Providing debugging info

    All reward spaces must implement:
    - get_reward_spec(): Define reward properties
    - compute_rewards_and_done(): Calculate rewards and termination

    This abstraction allows for:
    - Consistent reward space interface
    - Flexible reward definitions
    - Clear separation of concerns
    """
    def __init__(self, **kwargs):
        if kwargs:
            logging.warning(f"RewardSpace received unexpected kwargs: {kwargs}")

    @staticmethod
    @abstractmethod
    def get_reward_spec() -> RewardSpec:
        pass

    @abstractmethod
    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        pass

    def get_info(self) -> Dict[str, np.ndarray]:
        return {}

class FullGameRewardSpace(BaseRewardSpace):
    """
    Base class for reward spaces that span the entire game duration.

    This class provides a framework for:
    1. Game-Level Rewards:
       - Victory/defeat conditions
       - Resource accumulation
       - Territory control
       - Research progress

    2. Continuous Feedback:
       - Per-step rewards
       - Progress indicators
       - Strategic incentives

    3. Terminal Rewards:
       - Final game outcome
       - Achievement bonuses
       - Performance metrics

    The distinction between FullGameRewardSpace and Subtask is that
    full game rewards provide continuous feedback throughout the entire
    game, while subtasks focus on specific objectives that can be
    completed before the game ends.
    """
    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        return self.compute_rewards(game_state, done), done

    @abstractmethod
    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        pass


def count_points(game_state: Game) -> np.ndarray:
    """
    Count the number of city tiles owned by each player.

    Args:
        game_state: Current game state containing player information

    Returns:
        numpy array of shape (2,) containing city tile counts for each player
    """
    return np.array([player.relic_points for player in game_state.players])


class RelicPointsReward(FullGameRewardSpace):
    """
    Reward space based on city tile control.

    This reward space provides:
    1. Continuous Feedback:
       - Rewards proportional to team points
       - Updated every step
       - Normalized to [0, 1] range

    2. Strategic Incentives:
       - Encourages gathering points from relics

    3. Implementation Details:
       - Non-zero sum between players

    This reward helps agents learn the importance of being close to relics.
    """
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=0.,
            reward_max=1.,
            zero_sum=False,
            only_once=False
        )

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        return tuple(count_points(game_state) / 1024.)

class GameResultReward(FullGameRewardSpace):
    """
    Reward space that focuses on the final game outcome.

    This reward space:
    1. Terminal Rewards:
       - +1 for winner, -1 for loser
       - 0 reward during gameplay
       - Uses city tiles as primary victory metric
       - Uses unit count as tiebreaker

    2. Early Stopping:
       - Optional early game termination
       - Triggers on clear victory conditions
       - Prevents unnecessarily long games

    3. Implementation Details:
       - Normalizes rewards to [-1, 1] range
       - Zero-sum between players
       - Only awarded at game end

    This reward space encourages agents to focus on
    winning the game rather than intermediate objectives.
    """
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=True,
            only_once=True
        )

    def __init__(self, **kwargs):
        super(GameResultReward, self).__init__(**kwargs)

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        return self.compute_rewards(game_state, done), done

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        if not done:
            return 0., 0.

        # reward here is defined as the sum of number of city tiles with unit count as a tie-breaking mechanism
        rewards = [int(GameResultReward.compute_player_reward(p)) for p in game_state.players]
        rewards = (rankdata(rewards) - 1.) * 2. - 1.
        return tuple(rewards)

    @staticmethod
    def compute_player_reward(player: Player):
        relic_point = player.relic_points
        # max board size is 32 x 32 => 1024 max city tiles and units,
        # so this should keep it strictly so we break by city tiles then unit count
        return relic_point
