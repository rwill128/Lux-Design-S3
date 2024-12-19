import unittest



class TestBFSPathfinding(unittest.TestCase):
    def test_simple_path(self):
        map_width, map_height = 5,5
        # Everything passable
        def is_passable(x,y): return True
        path = run_bfs((0,0),(4,4),is_passable,map_width,map_height)
        self.assertIsNotNone(path)
        self.assertIn((4,4), path)

    def test_blocked_path(self):
        map_width, map_height = 5,5
        blocked = {(2,2)}
        def is_passable(x,y): return (x,y) not in blocked
        path = run_bfs((0,0),(4,4),is_passable,map_width,map_height)
        self.assertIsNotNone(path) # should still find a path around (2,2)

    def test_no_path(self):
        map_width, map_height = 3,3
        # block a row so no path exists
        blocked = {(1,0),(1,1),(1,2)}
        def is_passable(x,y): return (x,y) not in blocked
        path = run_bfs((0,0),(2,2),is_passable,map_width,map_height)
        self.assertIsNone(path)
