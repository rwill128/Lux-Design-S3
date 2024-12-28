class Constants:

    # noinspection PyPep8Naming
    class INPUT_CONSTANTS:
        TEAM_POINTS = "tp"
        TEAM_ENERGY = "te"
        RELIC_LOCATIONS = "rl"
        UNITS = "u"
        UNIT_ENERGY = "ue"
        ENERGY_MAP = "e"
        NEBULA_MAP = "nm"
        ASTEROID_MAP = "am"
        ROUND_NUM = "rn"
        DONE = "D_DONE"

    class DIRECTIONS:
        NORTH = 1
        WEST = 4
        SOUTH = 3
        EAST = 2
        CENTER = 0

        @staticmethod
        def astuple(include_center: bool):
            move_directions = (
                Constants.DIRECTIONS.NORTH,
                Constants.DIRECTIONS.EAST,
                Constants.DIRECTIONS.SOUTH,
                Constants.DIRECTIONS.WEST
            )
            if include_center:
                return move_directions + (Constants.DIRECTIONS.CENTER,)
            else:
                return move_directions

    # noinspection PyPep8Naming
    class TILE_TYPES:
        EMPTY = 0
        NEBULA = 1
        ASTEROID = 2

        @staticmethod
        def astuple():
            return Constants.TILE_TYPES.EMPTY, Constants.TILE_TYPES.NEBULA, Constants.TILE_TYPES.ASTEROID
