from .. problem import Problem
from .. datastructures.priority_queue import PriorityQueue


def get_solver_mapping():
    return dict(ucs=UCS)


class UCS(object):
    # TODO, excercise 2:
    # - implement Uniform Cost Search (UCS), a variant of Dijkstra's Graph Search
    # - use the provided PriorityQueue where appropriate
    # - to put items into the PriorityQueue, use 'pq.put(<priority>, <item>)'
    # - to get items out of the PriorityQueue, use 'pq.get()'
    # - store visited nodes in a 'set()'
    def solve(self, problem):
        fringe = PriorityQueue()
        visited = set()
        fringe_nodes = set()  
        node_costs = {}  
        start_node = problem.get_start_node()
        fringe.put(0, start_node)  
        fringe_nodes.add(start_node) 
        node_costs[start_node] = 0  
            

        while len(fringe) > 0:
            current_cost, current_node = fringe.get(include_priority=True)
            fringe_nodes.remove(current_node) 
                    
            if problem.is_end(current_node):
                return current_node
                    
            if current_node not in visited:
                visited.add(current_node)
                for successor in problem.successors(current_node):
                    if successor not in visited:
                        action_taken = successor.action  
                        action_cost = problem.action_cost(current_node.state, action_taken)
                        new_cost = current_cost + action_cost
                        if successor not in fringe_nodes or node_costs.get(successor, float('inf')) > new_cost:
                            fringe.put(new_cost, successor)
                            fringe_nodes.add(successor)  
                            node_costs[successor] = new_cost  
        return None





