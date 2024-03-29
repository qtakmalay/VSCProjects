from .. problem import Problem
from .. datastructures.queue import Queue
from tqdm import tqdm

# please ignore this
def get_solver_mapping():
    return dict(bfs=BFS)


class BFS(object):
    # TODO, exercise 1:
    # - implement Breadth First Search (BFS)
    # - use 'problem.get_start_node()' to get the node with the start state
    # - use 'problem.is_end(node)' to check whether 'node' is the node with the end state
    # - use a set() to store already visited nodes
    # - use the 'queue' datastructure that is already imported as the 'fringe'/ the 'frontier'
    # - use 'problem.successors(node)' to get a list of nodes containing successor states
    def solve(self, problem):
        visited = set()
        fringe = Queue()
        start_node = problem.get_start_node()
        
        if problem.is_end(start_node):
            return start_node
        
        fringe.put(start_node)
        visited.add(start_node)

        while len(fringe) > 0:
            current_node = fringe.get()
            for successor in problem.successors(current_node):
                if successor not in visited:
                    if problem.is_end(successor):
                        return successor
                    fringe.put(successor)
                    visited.add(successor)
                    
        return start_node





    
