from .. environment import Environment, Outcome
import numpy as np


def eps_greedy(rng, qs, epsilon):
    # this function makes an epsilon greedy decision
    # it trades off exploration and exploitation
    # exploration: trying out new options that may lead
    #              to better outcomes in the future
    # exploitation: choosing the best option based on past experience
    if rng.uniform(0, 1) < epsilon:
        # - with probability p == epsilon, an action is
        # chosen uniformly at random
        # use rng.randint as the source of randomness
        action = rng.randint(0, len(qs))
    else:
        # - with probability p == 1 - epsilon, the action
        #   having the currently largest q-value estimate is chosen;
        #   in case of multiple actions having the largest q-value,
        #   choose the first (leftmost) action in the array
        action = np.argmax(qs)
    return action


class QLearning():
    def train(self, env: Environment):
        ########################################
        # please leave untouched
        rng = np.random.RandomState(1234)
        alpha = 0.2
        epsilon = 0.3
        gamma = env.get_gamma()
        n_episodes = 10000
        ########################################

        ########################################
        # initialize the Q-'table'
        Q = np.zeros((env.get_n_states(), env.get_n_actions()))
        ########################################

        # TODO #################################
        for episode in range(1, n_episodes + 1):
            # TODO, exercise 3, generate an episode
            # with an eps_greedy policy, and then
            # implement the q-learning update here

            # you interact with the environment
            # ONLY via the methods
            #      'state = env.reset()'
            # and
            #      'state, reward, done = env.step(action)'
            #
            # 'state = env.reset()' is used to
            # reset the environment at the start
            # of an episode
            #
            # 'state, reward, done = env.step(action)'
            # is used to tell the environment that your
            # agent has decided to do 'action'.
            # the environment will then tell you, in
            # which state you actually ended up in,
            # what the immediate reward was, and whether
            # or not the episode ended.
            #
            # you can assume that states are encoded as
            # integers ranging from 0 to env.get_n_states() - 1.
            # you can assume that actions are encoded as
            # integers ranging from 0 to env.get_n_actions() - 1.
            # this means that you can use the states and actions directly to
            # index the correct element of Q (numpy array)
            #
            #
            state = env.reset()
            done = False

            while not done:
                action = eps_greedy(rng, Q[state], epsilon)
                next_state, reward, done = env.step(action)

                if action is not None:
                    best_next_action = np.argmax(Q[next_state])
                    td_target = reward + gamma * Q[next_state][best_next_action] * (1 - int(done))
                    td_error = td_target - Q[state][action]
                    Q[state][action] += alpha * td_error

                state = next_state


        ########################################

        ########################################

        # compute a deterministic policy from the Q value function
        policy = np.zeros((env.get_n_states(), env.get_n_actions()), dtype=np.int64)
        policy[np.arange(len(policy)), np.argmax(Q, axis=1)] = 1
        # the state value function V can be computed easily from Q
        # by taking the action that leads to the max future reward
        V = np.max(Q, axis=1)

        ########################################

        return Outcome(n_episodes, policy, V=V, Q=Q)
