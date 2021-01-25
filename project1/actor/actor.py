import random

class Actor():

    def __init__(self, board, lr=0.5, eligibility_decay=0, discount_factor=0.5, initial_epsilon=0.1, epsilon_decay_rate=0.1):
        self.board = board
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay_rate
        self.policy = {}

    def select_action(self):
        moves = self.board.get_all_legal_moves()
        random_choice = random.random()
        if random_choice < self.epsilon:
            return random.choice(moves)
        else:
            best_move = moves[0]
            for move in moves:
                try: 
                    try_move = self.policy[(self.board, move)]
                    if try_move > best_move:
                        best_move = try_move
                except KeyError:
                    pass
            return best_move

