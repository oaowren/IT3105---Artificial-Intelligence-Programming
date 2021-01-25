import random
import copy

class Actor():

    def __init__(self, lr=0.5, eligibility_decay=0, discount_factor=0.5, initial_epsilon=0.1, epsilon_decay_rate=0.1):
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay_rate
        self.policy = {}

    def init_policy(self, board):
        if board.check_losing_state() or board.check_winning_state():
            return 0
        else:
            moves = board.get_all_legal_moves()
            print(moves)
            for move in moves:
                self.policy[(board.board_state(), move)] = 0
                board_copy = copy.deepcopy(board)
                board_copy.make_move(move)
                self.init_policy(board_copy)
            return 0

    def select_action(self, board):
        moves = board.get_all_legal_moves()
        random_choice = random.random()
        if random_choice < self.epsilon:
            return random.choice(moves)
        else:
            best_move = moves[0]
            best_reward = -10000
            for move in moves:
                move_reward = self.policy[(board.board_state(), move)]
                if move_reward > best_reward:
                    best_move = move
                    best_reward = move_reward
            return best_move
    
    def update_policy(self, board, move, reward):
        self.policy[(board.board_state(), move)] = reward

