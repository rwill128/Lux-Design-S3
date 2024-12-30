import logging

from .constants import Constants
from .game_map import GameMap
from .game_objects import Player, Unit

INPUT_CONSTANTS = Constants.INPUT_CONSTANTS
GAME_LENGTH = 100


class Game:
    def _initialize(self, obs, env_cfg):
        """
        initialize state
        """
        self.turn = -1
        # get some other necessary initial input
        self.map_width = env_cfg["map_width"]
        self.map_height = env_cfg["map_height"]
        self.map = GameMap(self.map_width, self.map_height)
        self.players = [Player(0), Player(1)]


    def _reset_player_states(self):
        self.players[0].units = []
        self.players[0].relic_points = 0
        self.players[1].units = []
        self.players[1].relic_points = 0

    # noinspection PyProtectedMember
    def _update(self, obs):
        """
        Process turn update messages to refresh the game state.

        Args:
            messages (list[str]): List of update messages from the game engine describing
                                the current state of resources, units, cities, etc.

        Updates:
        - Players' relic points
        - Unit locations
        - Unit / player energy levels
        - Nebula locations
        - Asteroid locations
        - Energy grid

        The update process continues until a "D_DONE" message is received, indicating
        all state updates for the current turn have been processed.
        """

        self.map = GameMap(self.map_width, self.map_height)
        self.turn += 1
        self._reset_player_states()

        # TODO: This is what will take new obs and update the game state.
        #  We can keep the last X number of these if we want as some kind of memory,
        #  but better to have a NN structure that has recurrent or LSTM capabilities

        # TODO: This should take an obs I guess? And update all our known info. We'll also have to deal with keeping
        #  a queue of the last X game states so that our NN can have some kind of memory. As a rudimentary approach

        # Take observations and parse them into a Game, Game Map, Units, and Cells?
        #  I'm not sure this is needed actually.
        # If it is done, we have to have separate games and game maps for each player because of the hidden information