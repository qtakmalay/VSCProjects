import math
from .. problem import Problem
from .. datastructures.priority_queue import PriorityQueue


# please ignore this
def get_solver_mapping():
    return dict(
        astar_ec=ASTAR_Euclidean,
        astar_mh=ASTAR_Manhattan
    )

# def reconstruct_path(self, came_from, current):
#         path = []
#         while current is not None:
#             path.append(current)
#             current = came_from[current]
#         path.reverse()
#         return path

class ASTAR(object):
    # TODO, Exercise 2:
    # implement A* search (ASTAR)
    # - use the provided PriorityQueue where appropriate
    # - to put items into the PriorityQueue, use 'pq.put(<priority>, <item>)'
    # - to get items out of the PriorityQueue, use 'pq.get()'
    # - use a 'set()' to store nodes that were already visited
    def solve(self, problem):
        open_set = PriorityQueue()
        closed_set = set()
        start = problem.get_start_node()
        goal = problem.get_end_node()
        open_set.put(self.heuristic(start, goal), start)
        # g_costs = {start: 0}
        # came_from = {start: None}
        while open_set.has_elements(): 
            current = open_set.get()
            if problem.is_end(current):
                return current
            closed_set.add(current)
            for successor in problem.successors(current):
                if successor not in closed_set and successor not in open_set:
                    cost = successor.cost + current.cost
                    open_set.put(cost + self.heuristic(successor, goal), successor)
                    
        return None


# please note that in an ideal world, the heuristics should actually be part
# of the problem definition, as it assumes domain knowledge about the structure
# of the problem, and defines a distance to the goal state


# this is the ASTAR variant with the euclidean distance as a heuristic
# it is registered as a solver with the name 'astar_ec'
class ASTAR_Euclidean(ASTAR):
    def heuristic(self, current, goal):
        cy, cx = current.state
        gy, gx = goal.state
        return math.sqrt((cy - gy) ** 2 + (cx - gx) ** 2)


# this is the ASTAR variant with the manhattan distance as a heuristic
# it is registered as a solver with the name 'astar_mh'
class ASTAR_Manhattan(ASTAR):
    def heuristic(self, current, goal):
        cy, cx = current.state
        gy, gx = goal.state
        return math.fabs((cy - gy)) + math.fabs(cx - gx)
