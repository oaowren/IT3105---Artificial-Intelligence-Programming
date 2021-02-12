import random

class TableLookupCritic():

    def __init__(self, board, lr, eligibility_decay, discount_factor):
        self.board = board
        self.gamma = discount_factor
        self.lam = eligibility_decay
        self.alpha = lr
        self.delta = 0
        self.expected_reward = {}
        self.eligibility = {}
    
    def update_expected_reward(self, sequence):
        if len(sequence) == 2:
            self.eligibility[sequence[0][0]] = self.gamma * self.lam
        self.eligibility[sequence[-1][0]] = 1
        for state,_ in sequence:
            self.expected_reward[state] += self.alpha * self.delta * self.eligibility[state]
            self.eligibility[state] *= self.gamma * self.lam

    def calculate_td_error(self, old_state, new_state, reward):
        for state in [old_state, new_state]:
            if state not in self.expected_reward:
                self.expected_reward[state] = random.uniform(0, 0.2)
        self.delta = reward + self.gamma*self.expected_reward[new_state] - self.expected_reward[old_state]
        return self.delta
