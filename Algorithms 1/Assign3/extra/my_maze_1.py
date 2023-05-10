from typing import List, Tuple

# Using constants might make this more readable.
START = 'S'
EXIT = 'X'
VISITED = '.'
OBSTACLE = '#'
PATH = ' '


class MyMaze:
    """Maze object, used for demonstrating recursive algorithms."""

    def __init__(self, maze_str: str = ''):
        """Initialize Maze.

        Args:
            maze_str (str): Maze represented by a string,
            where rows are separated by newlines (\n).

        Raises:
            ValueError, if maze_str is empty.
        """
        if len(maze_str) == 0:
            raise ValueError
        else:
            # We internally treat this as a List[List[str]], as it makes indexing easier.
            self._maze = list(list(row) for row in maze_str.splitlines())
            self._row_range = len(self._maze)
            self._col_range = len(self._maze[0])
            self._exits: List[Tuple[int, int]] = []
            self._max_recursion_depth = 0

    def find_exits(self, start_row: int, start_col: int, depth: int = 0) -> bool:
        # check if we have exceeded the maximum recursion depth
        if depth > self._max_depth:
            return False

        # check if the current cell is an exit
        if self.exits(start_row, start_col):
            return True

        # mark the current cell as visited
        self._visited[start_row][start_col] = True

        # check all neighboring cells
        for dr, dc in self._directions:
            new_row = start_row + dr
            new_col = start_col + dc

            # check if the neighboring cell is in bounds and not a wall or visited
            if 0 <= new_row < self._row_range and 0 <= new_col < self._col_range and not self._maze[new_row][new_col] and not self._visited[new_row][new_col]:
                # recursively check the neighboring cell
                if self.find_exits(new_row, new_col, depth + 1):
                    return True

        # if no exits were found, unmark the current cell as visited and return False
        self._visited[start_row][start_col] = False
        return False


    @property
    def exits(self) -> List[Tuple[int, int]]:
        return self._exits

    @property
    def max_recursion_depth(self) -> int:
        return self._max_recursion_depth

    def __str__(self) -> str:
        return '\n'.join(''.join(row) for row in self._maze)

    __repr__ = __str__
