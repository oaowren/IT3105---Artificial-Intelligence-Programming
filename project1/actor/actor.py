import random


class Actor:
    def __init__(
        self,
        lr,
        eligibility_decay,
        discount_factor,
        initial_epsilon,
        epsilon_decay_rate,
    ):
        self.eps = initial_epsilon
        self.eps_dec = epsilon_decay_rate

        self.alpha = lr
        self.lam = eligibility_decay
        self.gamma = discount_factor
        self.policy = {}
        self.eligibility = {}


    def select_action(self, board):
        moves = board.get_all_legal_moves()
        random_choice = random.random()
        if random_choice < self.eps:
            return random.choice(moves)
        else:
            best_move = moves[0]
            best_reward = -10000

            for move in moves:
                if (board.board_state(), move) not in self.policy:
                    self.policy[(board.board_state(), move)] = 0
                move_reward = self.policy[(board.board_state(), move)]
                if move_reward > best_reward:
                    best_move = move
                    best_reward = move_reward
            return best_move
    
    def update(self, delta, sequence):
        self.eligibility[sequence[-1]] = 1
        
        for state,action in sequence:
            if not (state,action) in self.policy:
                self.policy[(state,action)] = 0
            if not (state, action) in self.eligibility:
                self.eligibility[(state,action)] = 1
            self.policy[(state,action)] += self.alpha * delta * self.eligibility[(state,action)]
            self.eligibility[(state,action)] *= self.gamma * self.lam

