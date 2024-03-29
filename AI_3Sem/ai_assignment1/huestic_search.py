from queue import PriorityQueue, Queue, LifoQueue

# Define the tree structure in a graph
graph = {
    'A': {'B': 3, 'C': 2, 'D': 2},
    'B': {'E': 4},
    'C': {'F': 4},
    'D': {'G': 5, 'H': 4},
    'E': {},
    'F': {},
    'G': {},
    'H': {}
}

# Define the heuristic for each node
heuristics = {
    'A': 2,
    'B': 1,
    'C': 0,
    'D': 3,
    'E': 0,
    'F': 3,
    'G': 2,
    'H': 4
}

# Define the goal node
goal = 'C'

# A* Search
def a_star_search(graph, start, goal):
    # The queue stores tuples of (f_score, node, path_cost, path)
    # f_score is the sum of path_cost and heuristic, which A* uses to prioritize nodes
    frontier = PriorityQueue()
    frontier.put((heuristics[start], start, 0, [start]))
    explored = set()

    while not frontier.empty():
        f_score, current_node, path_cost, path = frontier.get()
        explored.add(current_node)

        # Check if we have reached the goal
        if current_node == goal:
            return path, path_cost

        # Check all neighbors of the current node
        for neighbor, cost in graph[current_node].items():
            new_cost = path_cost + cost
            new_path = path + [neighbor]
            if neighbor not in explored:
                f_score = new_cost + heuristics[neighbor]
                frontier.put((f_score, neighbor, new_cost, new_path))
                explored.add(neighbor)

    return None, None

# Uniform Cost Search
def uniform_cost_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put((0, start, [start]))
    explored = set()

    while not frontier.empty():
        path_cost, current_node, path = frontier.get()
        explored.add(current_node)

        if current_node == goal:
            return path, path_cost

        for neighbor, cost in graph[current_node].items():
            if neighbor not in explored:
                new_cost = path_cost + cost
                new_path = path + [neighbor]
                frontier.put((new_cost, neighbor, new_path))
                explored.add(neighbor)

    return None, None

# Greedy Best-First Search
def greedy_best_first_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put((heuristics[start], start, [start]))
    explored = set()

    while not frontier.empty():
        _, current_node, path = frontier.get()
        explored.add(current_node)

        if current_node == goal:
            return path, heuristics[current_node]

        for neighbor, _ in graph[current_node].items():
            if neighbor not in explored:
                frontier.put((heuristics[neighbor], neighbor, path + [neighbor]))
                explored.add(neighbor)

    return None, None

# Breadth-First Search
def breadth_first_search(graph, start, goal):
    frontier = Queue()
    frontier.put((start, [start]))
    explored = set()

    while not frontier.empty():
        current_node, path = frontier.get()
        explored.add(current_node)

        if current_node == goal:
            return path

        for neighbor in graph[current_node]:
            if neighbor not in explored:
                frontier.put((neighbor, path + [neighbor]))
                explored.add(neighbor)

    return None

# Depth-First Search
def depth_first_search(graph, start, goal):
    frontier = LifoQueue()
    frontier.put((start, [start]))
    explored = set()

    while not frontier.empty():
        current_node, path = frontier.get()
        explored.add(current_node)

        if current_node == goal:
            return path

        for neighbor in graph[current_node]:
            if neighbor not in explored:
                frontier.put((neighbor, path + [neighbor]))
                explored.add(neighbor)

    return None

# Test each search algorithm
a_star_path, a_star_cost = a_star_search(graph, 'A', goal)
ucs_path, ucs_cost = uniform_cost_search(graph, 'A', goal)
gbfs_path, gbfs_cost = greedy_best_first_search(graph, 'A', goal)
bfs_path = breadth_first_search(graph, 'A', goal)
dfs_path = depth_first_search(graph, 'A', goal)

a_star_path, a_star_cost, ucs_path, ucs_cost, gbfs_path, gbfs_cost, bfs_path, dfs_path
