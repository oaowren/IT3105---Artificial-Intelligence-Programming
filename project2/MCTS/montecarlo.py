
"""
Dette er hovedsaklig basert på algoritmen fra
http://www.cs.cornell.edu/courses/cs6700/2016sp/lectures/CS6700-UCT.pdf
side 5-7.
"""

import numpy as np

class MCTS:

    def __init__(self, root, nn):
        self.root = root
        self.tree_policy = {}
        self.states= {}
        self.state_action = {}
        self.c = 1
        self.nn = nn

    def update(self, state, action, reward):
        self.state[state]["N"] +=1
        self.state_action_pair[(state,action)]["N"] +=1
        self.state_action_pair[(state,action)]["Q"] += (reward - self.get_Q(state, action))/self.get_N(state, action)
        return

    def get_N(self, state, action=None):
        if action:
            if (state,action) not in self.state_action_pairs:
                self.state_action_pair[(state,action)]["N"] = 0
            return self.state_action_pair[(state,action)]["N"]
        if state not in self.state:
            self.state[state]["N"] = 0
        return self.state[state]["N"]

    def get_Q(self, state, action):
        return self.state_Action_pair[(state,action)]["Q"]

    def exploration_bonus(self, state, action):
        exploration_bonus =self.c*np.sqrt(np.log(self.get_N(state))/self.get_N(state,action))
        return exploration_bonus

    def rollout_action(self, state, epsilon, player):
        split_state = np.concatenate([player], [int(i) for i in state.split()])
        preds = self.nn.predict([split_state])
        return self.nn.epsilon_best_action(preds, epsilon)




