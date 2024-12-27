from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from lux_ai.utility_constants import MAX_BOARD_SIZE
import gym

class BaseObsSpace(ABC):
    # NB: Avoid using Discrete() space, as it returns a shape of ()
    # NB: "_COUNT" keys indicate that the value is used to scale the embedding of another value
    @abstractmethod
    def get_obs_spec(
            self,
            board_dims: Tuple[int, int] = MAX_BOARD_SIZE
    ) -> gym.spaces.Dict:
        pass

    @abstractmethod
    def wrap_env(self, env) -> gym.Wrapper:
        pass