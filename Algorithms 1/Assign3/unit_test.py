"""
Created on May 05, 2020
Updated on April 10, 2022 by Florian Beck
Updated on April 06, 2023 by Florian Beck

@author: martin
"""
import unittest
from datetime import date

from my_maze import MyMaze


class MazeSolveStruct:
    def __init__(self):
        self.maze = None
        self.maze_solved = None
        self.list_of_exits = []
        self.range_row = 0
        self.range_col = 0
        self.start_row = 0
        self.start_col = 0
        self.max_rec_depth = 0


maxPoints = 10.0  # defines the maximum achievable points for the example tested here
# two more points for the creation of own mazes (in my_maze.py)
points = maxPoints  # stores the actually achieved points based on failed unit tests
summary = ""


def deduct_pts(value):
    global points
    points = points - value
    if points < 0:
        points = 0


def resolve_amount_of_pts_to_deduct(argument):
    pool = {
        "test_one_exit": 0.5,
        "test_two_exits": 0.5,
        "test_min_maze_with_exit": 0.5,
        "test_split_maze_with_two_exits": 0.5,
        "test_split_maze_without_exit": 0.5,
        "test_big_maze_one_exit": 0.5,
        "test_min_maze_with_four_exits": 0.5,
        "test_min_maze_without_exit": 0.5,
        "test_max_rec_depth1": 0.5,
        "test_max_rec_depth2": 1,
        "test_max_rec_depth3": 1,
        "test_diagonal_movement_with_exit": 0.5,
        "test_maze_map_one_exit": 1,
        "test_maze_map_multiple_exits": 1,
        "test_maze_map_diagonal": 1,
    }

    # resolve the pts to deduct from pool
    return pool.get(argument, 0)


def create_maze(maze_type):
    myMazeStruct = MazeSolveStruct()
    if maze_type == 0:
        myMazeStruct.range_row = 6
        myMazeStruct.range_col = 6
        myMazeStruct.maze = "######\n"
        myMazeStruct.maze += "#    #\n"
        myMazeStruct.maze += "## # #\n"
        myMazeStruct.maze += "#  # #\n"
        myMazeStruct.maze += "# ## #\n"
        myMazeStruct.maze += "#### #\n"

        myMazeStruct.list_of_exits.append([5, 4])
        myMazeStruct.start_row = 1
        myMazeStruct.start_col = 1

        myMazeStruct.maze_solved = "######\n"
        myMazeStruct.maze_solved += "#S...#\n"
        myMazeStruct.maze_solved += "##.#.#\n"
        myMazeStruct.maze_solved += "#..#.#\n"
        myMazeStruct.maze_solved += "#.##.#\n"
        myMazeStruct.maze_solved += "####X#\n"

    elif maze_type == 1:
        myMazeStruct.range_row = 3
        myMazeStruct.range_col = 3
        myMazeStruct.maze = "###\n"
        myMazeStruct.maze += "  #\n"
        myMazeStruct.maze += "###\n"

        myMazeStruct.list_of_exits.append([1, 0])
        myMazeStruct.start_row = 1
        myMazeStruct.start_col = 1

        myMazeStruct.maze_solved = "###\n"
        myMazeStruct.maze_solved += ".S#\n"
        myMazeStruct.maze_solved += "###\n"

    elif maze_type == 2:
        myMazeStruct.range_row = 3
        myMazeStruct.range_col = 3
        myMazeStruct.maze = "###\n"
        myMazeStruct.maze += "# #\n"
        myMazeStruct.maze += "###\n"

        myMazeStruct.start_row = 1
        myMazeStruct.start_col = 1
        myMazeStruct.max_rec_depth = 1

        myMazeStruct.maze_solved = "###\n"
        myMazeStruct.maze_solved += "#S#\n"
        myMazeStruct.maze_solved += "###\n"

    elif maze_type == 3:
        myMazeStruct.range_row = 6
        myMazeStruct.range_col = 6
        myMazeStruct.maze = "######\n"
        myMazeStruct.maze += "#    #\n"
        myMazeStruct.maze += "## # #\n"
        myMazeStruct.maze += "#  #  \n"
        myMazeStruct.maze += "# ## #\n"
        myMazeStruct.maze += "#### #\n"

        myMazeStruct.list_of_exits.append([5, 4])
        myMazeStruct.list_of_exits.append([3, 5])
        myMazeStruct.start_row = 1
        myMazeStruct.start_col = 1

        myMazeStruct.maze_solved = "######\n"
        myMazeStruct.maze_solved += "#S...#\n"
        myMazeStruct.maze_solved += "##.#.#\n"
        myMazeStruct.maze_solved += "#..#.X\n"
        myMazeStruct.maze_solved += "#.##.#\n"
        myMazeStruct.maze_solved += "####X#\n"

    elif maze_type == 4:
        myMazeStruct.range_row = 3
        myMazeStruct.range_col = 3
        myMazeStruct.maze = "# #\n"
        myMazeStruct.maze += "   \n"
        myMazeStruct.maze += "# #\n"

        myMazeStruct.list_of_exits.append([0, 1])
        myMazeStruct.list_of_exits.append([1, 0])
        myMazeStruct.list_of_exits.append([2, 1])
        myMazeStruct.list_of_exits.append([1, 2])
        myMazeStruct.start_row = 1
        myMazeStruct.start_col = 1

        myMazeStruct.maze_solved = "#X#\n"
        myMazeStruct.maze_solved += "XSX\n"
        myMazeStruct.maze_solved += "#X#\n"

    elif maze_type == 5:
        myMazeStruct.range_row = 6
        myMazeStruct.range_col = 6
        myMazeStruct.maze = "######\n"
        myMazeStruct.maze += "#  # #\n"
        myMazeStruct.maze += "## # #\n"
        myMazeStruct.maze += "#  #  \n"
        myMazeStruct.maze += "# ## #\n"
        myMazeStruct.maze += "#### #\n"

        myMazeStruct.list_of_exits.append([5, 4])
        myMazeStruct.list_of_exits.append([3, 5])
        myMazeStruct.start_row = 4
        myMazeStruct.start_col = 4

        myMazeStruct.maze_solved = "######\n"
        myMazeStruct.maze_solved += "#  #.#\n"
        myMazeStruct.maze_solved += "## #.#\n"
        myMazeStruct.maze_solved += "#  #.X\n"
        myMazeStruct.maze_solved += "# ##S#\n"
        myMazeStruct.maze_solved += "####X#\n"

    elif maze_type == 6:
        myMazeStruct.range_row = 6
        myMazeStruct.range_col = 6
        myMazeStruct.maze = "######\n"
        myMazeStruct.maze += "#  # #\n"
        myMazeStruct.maze += "## # #\n"
        myMazeStruct.maze += "#  #  \n"
        myMazeStruct.maze += "# ## #\n"
        myMazeStruct.maze += "#### #\n"

        myMazeStruct.start_row = 3
        myMazeStruct.start_col = 2

        myMazeStruct.maze_solved = "######\n"
        myMazeStruct.maze_solved += "#..# #\n"
        myMazeStruct.maze_solved += "##.# #\n"
        myMazeStruct.maze_solved += "#.S#  \n"
        myMazeStruct.maze_solved += "#.## #\n"
        myMazeStruct.maze_solved += "#### #\n"

    elif maze_type == 7:
        myMazeStruct.range_row = 15
        myMazeStruct.range_col = 15
        myMazeStruct.maze = "###############\n"
        myMazeStruct.maze += "#      #      #\n"  # 12
        myMazeStruct.maze += "### # ######  #\n"  # 4
        myMazeStruct.maze += "#   # ####### #\n"  # 5
        myMazeStruct.maze += "# ###         #\n"  # 10
        myMazeStruct.maze += "#   #######   #\n"  # 6
        myMazeStruct.maze += "### #         #\n"  # 10
        myMazeStruct.maze += "    # ######  #\n"  # 6
        myMazeStruct.maze += "#####      ## #\n"  # 7
        myMazeStruct.maze += "##### ######  #\n"  # 3
        myMazeStruct.maze += "#         ##  #\n"  # 11
        myMazeStruct.maze += "# ########## ##\n"  # 2
        myMazeStruct.maze += "#          # ##\n"  # 11
        myMazeStruct.maze += "#   ##       ##\n"  # 10
        myMazeStruct.maze += "###############\n"  # --

        myMazeStruct.list_of_exits.append([7, 0])

        myMazeStruct.start_row = 13
        myMazeStruct.start_col = 9
        myMazeStruct.max_rec_depth = 97

        myMazeStruct.maze_solved = "###############\n"
        myMazeStruct.maze_solved += "#......#......#\n"  # 12
        myMazeStruct.maze_solved += "###.#.######..#\n"  # 4
        myMazeStruct.maze_solved += "#...#.#######.#\n"  # 5
        myMazeStruct.maze_solved += "#.###.........#\n"  # 10
        myMazeStruct.maze_solved += "#...#######...#\n"  # 6
        myMazeStruct.maze_solved += "###.#.........#\n"  # 10
        myMazeStruct.maze_solved += "X...#.######..#\n"  # 6
        myMazeStruct.maze_solved += "#####......##.#\n"  # 7
        myMazeStruct.maze_solved += "#####.######..#\n"  # 3
        myMazeStruct.maze_solved += "#.........##..#\n"  # 11
        myMazeStruct.maze_solved += "#.##########.##\n"  # 2
        myMazeStruct.maze_solved += "#..........#.##\n"  # 11
        myMazeStruct.maze_solved += "#...##...S...##\n"  # 10
        myMazeStruct.maze_solved += "###############\n"

    elif maze_type == 8:
        myMazeStruct.range_row = 6
        myMazeStruct.range_col = 6
        # with diagonal movement
        myMazeStruct.maze = "######\n"
        myMazeStruct.maze += "#  # #\n"
        myMazeStruct.maze += "## # #\n"
        myMazeStruct.maze += "#  #  \n"
        myMazeStruct.maze += "# #  #\n"
        myMazeStruct.maze += "#### #\n"

        myMazeStruct.list_of_exits.append([5, 4])
        myMazeStruct.list_of_exits.append([3, 5])
        myMazeStruct.start_row = 1
        myMazeStruct.start_col = 1

        myMazeStruct.maze_solved = "######\n"
        myMazeStruct.maze_solved += "#S.#.#\n"
        myMazeStruct.maze_solved += "##.#.#\n"
        myMazeStruct.maze_solved += "#..#.X\n"
        myMazeStruct.maze_solved += "#.#..#\n"
        myMazeStruct.maze_solved += "####X#\n"

    elif maze_type == 9:
        myMazeStruct.range_row = 15
        myMazeStruct.range_col = 15
        myMazeStruct.maze = "###############\n"
        myMazeStruct.maze += "#             #\n"  # 13
        myMazeStruct.maze += "############# #\n"  # 1
        myMazeStruct.maze += "#             #\n"  # 13
        myMazeStruct.maze += "# #############\n"  # 1
        myMazeStruct.maze += "#             #\n"  # 13
        myMazeStruct.maze += "############# #\n"  # 1
        myMazeStruct.maze += "#             #\n"  # 13
        myMazeStruct.maze += "# #############\n"  # 1
        myMazeStruct.maze += "#             #\n"  # 13
        myMazeStruct.maze += "############# #\n"  # 1
        myMazeStruct.maze += "############# #\n"  # 1
        myMazeStruct.maze += "############# #\n"  # 1
        myMazeStruct.maze += "#             #\n"  # 13
        myMazeStruct.maze += "###############\n"  # --

        myMazeStruct.start_row = 1
        myMazeStruct.start_col = 1
        myMazeStruct.max_rec_depth = 85

        myMazeStruct.maze_solved = "###############\n"
        myMazeStruct.maze_solved += "#S............#\n"  # 13
        myMazeStruct.maze_solved += "#############.#\n"  # 1
        myMazeStruct.maze_solved += "#.............#\n"  # 13
        myMazeStruct.maze_solved += "#.#############\n"  # 1
        myMazeStruct.maze_solved += "#.............#\n"  # 13
        myMazeStruct.maze_solved += "#############.#\n"  # 1
        myMazeStruct.maze_solved += "#.............#\n"  # 13
        myMazeStruct.maze_solved += "#.#############\n"  # 1
        myMazeStruct.maze_solved += "#.............#\n"  # 13
        myMazeStruct.maze_solved += "#############.#\n"  # 1
        myMazeStruct.maze_solved += "#############.#\n"  # 1
        myMazeStruct.maze_solved += "#############.#\n"  # 1
        myMazeStruct.maze_solved += "#.............#\n"  # 13
        myMazeStruct.maze_solved += "###############\n"  # --

    elif maze_type == 10:
        myMazeStruct.range_row = 6
        myMazeStruct.range_col = 6
        myMazeStruct.maze = "## ###\n"
        myMazeStruct.maze += "#    #\n"
        myMazeStruct.maze += "## # #\n"
        myMazeStruct.maze += "#  #  \n"
        myMazeStruct.maze += "  ## #\n"
        myMazeStruct.maze += "#### #\n"

        myMazeStruct.list_of_exits.append([5, 4])
        myMazeStruct.list_of_exits.append([3, 5])
        myMazeStruct.list_of_exits.append([0, 2])
        myMazeStruct.list_of_exits.append([4, 0])
        myMazeStruct.start_row = 1
        myMazeStruct.start_col = 1

        myMazeStruct.maze_solved = "##X###\n"
        myMazeStruct.maze_solved += "#S...#\n"
        myMazeStruct.maze_solved += "##.#.#\n"
        myMazeStruct.maze_solved += "#..#.X\n"
        myMazeStruct.maze_solved += "X.##.#\n"
        myMazeStruct.maze_solved += "####X#\n"
    return myMazeStruct


def maze2list(maze_str):
    return list(list(row) for row in maze_str.splitlines())


def print_maze(maze):
    s = [''.join(x) for x in maze]
    return '\n'.join(s)


def print_list_of_exits(exit_list):
    ret_str = ""
    for i in range(len(exit_list)):
        ret_str += "[x=" + str(exit_list[i][0]) + ",y=" + str(exit_list[i][1]) + "] "
    return ret_str


def analyze_maze(maze_number):
    myMazeStruct = create_maze(maze_number)
    maze = MyMaze(myMazeStruct.maze)
    # analyze maze
    maze.find_exits(myMazeStruct.start_row, myMazeStruct.start_col, 0)
    return maze, myMazeStruct


class UnitTestTemplate(unittest.TestCase):
    def setUp(self):
        pass

    ####################################################
    # Definition of test cases
    ####################################################

    def test_one_exit(self):
        maze, myMazeStruct = analyze_maze(0)
        self.check_exits("ERROR Test with 1 exit failed: ", myMazeStruct.list_of_exits, maze.exits, maze._maze)

    def test_two_exits(self):
        maze, myMazeStruct = analyze_maze(3)
        self.check_exits("ERROR Test with 2 exits failed: ", myMazeStruct.list_of_exits, maze.exits, maze._maze)

    def test_min_maze_with_exit(self):
        maze, myMazeStruct = analyze_maze(1)
        self.check_exits("ERROR Test minimal maze with one exit failed: ", myMazeStruct.list_of_exits, maze.exits,
                         maze._maze)

    def test_split_maze_with_two_exits(self):
        maze, myMazeStruct = analyze_maze(5)
        self.check_exits("ERROR Test split maze failed: ", myMazeStruct.list_of_exits, maze.exits, maze._maze)

    def test_split_maze_without_exit(self):
        maze, myMazeStruct = analyze_maze(6)
        self.check_exits("ERROR Test split maze failed: ", myMazeStruct.list_of_exits, maze.exits, maze._maze)

    def test_big_maze_one_exit(self):
        maze, myMazeStruct = analyze_maze(7)
        self.check_exits("ERROR Test big maze with one exit failed: ", myMazeStruct.list_of_exits, maze.exits,
                         maze._maze)

    def test_min_maze_with_four_exits(self):
        maze, myMazeStruct = analyze_maze(4)
        self.check_exits("ERROR Test minimal maze with four exits failed: ", myMazeStruct.list_of_exits, maze.exits,
                         maze._maze)

    def test_min_maze_without_exit(self):
        maze, myMazeStruct = analyze_maze(2)
        self.check_exits("ERROR Test split maze failed: ", myMazeStruct.list_of_exits, maze.exits, maze._maze)

    def test_max_rec_depth1(self):
        maze_number = 2
        maze, myMazeStruct = analyze_maze(maze_number)
        self.assertTrue(maze.max_recursion_depth <= myMazeStruct.max_rec_depth,
                        "ERROR in 'max_recursion_depth()': max depth failed. Your depth (" + str(
                            maze.max_recursion_depth)
                        + ") is greater than max. depth (" + str(myMazeStruct.max_rec_depth) + ") for the maze:\n"
                        + print_maze(maze._maze) + "\n -->")
        self.assertTrue(maze.max_recursion_depth > myMazeStruct.max_rec_depth / 2,
                        "ERROR in 'max_recursion_depth()': max depth failed. Your depth (" + str(
                            maze.max_recursion_depth)
                        + ") is too low for the maze:\n"
                        + print_maze(maze._maze) + "\n -->")

    def test_max_rec_depth2(self):
        maze, myMazeStruct = analyze_maze(7)
        self.assertTrue(maze.max_recursion_depth <= myMazeStruct.max_rec_depth,
                        "ERROR in 'max_recursion_depth()' in Test (2) max. recursion depth failed. Your depth (" + str(
                            maze.max_recursion_depth)
                        + ") is greater than max. depth (" + str(myMazeStruct.max_rec_depth) + ") for the maze:\n"
                        + print_maze(maze._maze))
        self.assertTrue(maze.max_recursion_depth > myMazeStruct.max_rec_depth / 3,
                        "ERROR in 'max_recursion_depth()': max depth failed. Your depth (" + str(
                            maze.max_recursion_depth)
                        + ") is too low for the maze:\n"
                        + print_maze(maze._maze) + "\n -->")

    def test_max_rec_depth3(self):
        maze, myMazeStruct = analyze_maze(9)
        self.assertTrue(maze.max_recursion_depth <= myMazeStruct.max_rec_depth,
                        "ERROR in 'max_recursion_depth()' in Test (2) max. recursion depth failed. Your depth (" + str(
                            maze.max_recursion_depth)
                        + ") is greater than max. depth (" + str(myMazeStruct.max_rec_depth) + ") for the maze:\n"
                        + print_maze(maze._maze))
        self.assertTrue(maze.max_recursion_depth > myMazeStruct.max_rec_depth / 1.2,
                        "ERROR in 'max_recursion_depth()': max depth failed. Your depth (" + str(
                            maze.max_recursion_depth)
                        + ") is too low for the maze:\n"
                        + print_maze(maze._maze) + "\n -->")

    def test_diagonal_movement_with_exit(self):
        maze, myMazeStruct = analyze_maze(8)
        self.assertEqual(len(myMazeStruct.list_of_exits), len(maze.exits), "ERROR Exit(s) found (" +
                         print_list_of_exits(maze.exits)
                         + print_maze(maze._maze))

    def test_maze_map_one_exit(self):
        maze, myMazeStruct = analyze_maze(7)

        for i in range(myMazeStruct.range_row):
            for j in range(myMazeStruct.range_col):
                self.assertEqual(maze2list(myMazeStruct.maze_solved)[i][j], maze._maze[i][j],
                                 "ERROR in Test of maze map (large):"
                                 " Wrong char found at position [x=" + str(j) + ",y=" + str(i) + "] - expected '"
                                 + maze2list(myMazeStruct.maze_solved)[i][j] + "' but found '" + maze._maze[i][
                                     j] + "'\n"
                                 + print_maze(maze._maze))

    def test_maze_map_multiple_exits(self):
        maze, myMazeStruct = analyze_maze(10)

        for i in range(myMazeStruct.range_row):
            for j in range(myMazeStruct.range_col):
                self.assertEqual(maze2list(myMazeStruct.maze_solved)[i][j], maze._maze[i][j],
                                 "ERROR in Test of maze map (large):"
                                 " Wrong char found at position [x=" + str(i) + ",y=" + str(j) + "] - expected '"
                                 + maze2list(myMazeStruct.maze_solved)[i][j] + "' but found '" + maze._maze[i][
                                     j] + "'\n"
                                 + print_maze(maze._maze))

    def test_maze_map_diagonal(self):
        maze, myMazeStruct = analyze_maze(8)

        for i in range(myMazeStruct.range_row):
            for j in range(myMazeStruct.range_col):
                self.assertEqual(maze2list(myMazeStruct.maze_solved)[i][j], maze._maze[i][j],
                                 "ERROR in Test of maze map (large):"
                                 " Wrong char found at position [x=" + str(i) + ",y=" + str(j) + "] - expected '"
                                 + maze2list(myMazeStruct.maze_solved)[i][j] + "' but found '" + maze._maze[i][
                                     j] + "'\n"
                                 + print_maze(maze._maze))

    def check_exits(self, error_msg, list_solution, list_student, stud_maze):
        self.assertEqual(len(list_solution), len(list_student),
                         error_msg + "Wrong number of exits! Exit(s) found @: " + print_list_of_exits(
                             list_student) + "\n" + print_maze(stud_maze))
        for i in range(len(list_solution)):
            found = False
            for stud_exit in list_student:
                if list_solution[i][0] == stud_exit[0]:
                    if list_solution[i][1] == stud_exit[1]:
                        found = True
                        break
            self.assertTrue(found, error_msg + "Exit @[x=" + str(list_solution[i][0]) + ",y=" + str(
                list_solution[i][1]) + "] not found!\n" + print_maze(stud_maze))


if __name__ == "__main__":
    unittest.main()
