def max_value(state, alpha, beta):
    if terminal_test(state):
        return utility(state), None
    v = float('-inf')
    for a in actions(state):
        min_val, _ = min_value(result(state, a), alpha, beta)
        if min_val > v:
            v = min_val
            best_action = a
        if v >= beta:
            return v, best_action
        alpha = max(alpha, v)
    return v, best_action

def min_value(state, alpha, beta):
    if terminal_test(state):
        return utility(state), None
    v = float('inf')
    for a in actions(state):
        max_val, _ = max_value(result(state, a), alpha, beta)
        if max_val < v:
            v = max_val
            best_action = a
        if v <= alpha:
            return v, best_action
        beta = min(beta, v)
    return v, best_action

def terminal_test(state):
    # In our case, the terminal test is simply whether the state is an integer
    # which represents a utility value at a leaf node.
    return isinstance(state, int)

def utility(state):
    # In a terminal state, the utility is the value of the state itself.
    return state

def actions(state):
    # Actions are the possible moves from the current state, which are the children nodes.
    return state if isinstance(state, list) else []

def result(state, action):
    # The result of taking an action in a state is simply that action
    # as we are directly passing the possible subsequent states (children nodes).
    return action

# Define the game tree
# Note: Normally, the game tree would be defined with objects or more complex data structures
# For simplicity, we use nested lists, where leaf nodes are integer utilities
# and other nodes are lists of their children (possible actions).
game_tree = [
    [1, 2, 1],  # Children of A1
    [2, 2, 0],  # Children of A2
    [3, 1, 2]   # Children of A3
]

# Applying the algorithm
best_value, best_action = max_value(game_tree, float('-inf'), float('inf'))
print(best_value)
print( best_action)
