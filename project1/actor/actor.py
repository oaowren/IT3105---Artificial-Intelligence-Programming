import random
import copy


class Actor:
    def __init__(
        self,
        lr=0.5,
        eligibility_decay=0,
        discount_factor=0.5,
        initial_epsilon=0.1,
        epsilon_decay_rate=0.1,
    ):
        self.eps = initial_epsilon
        self.eps_dec = epsilon_decay_rate
        self.discount = discount_factor
        self.lr = lr
        self.eli_dec = eligibility_decay
        self.policy = {}
        self.eligibility = {}

    def init_policy(self, board):
        if board.check_losing_state() or board.check_winning_state():
            return 0
        else:
            moves = board.get_all_legal_moves()
            for move in moves:
                self.policy[(board.board_state(), move)] = 0
                board_copy = copy.deepcopy(board)
                board_copy.make_move(move)
                self.init_policy(board_copy)
            return 0

    def select_action(self, board):
        moves = board.get_all_legal_moves()
        random_choice = random.random()
        if random_choice < self.eps:
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

    def update_policy(self, board, move, td_error):
        self.policy[(board.board_state(), move)] = (
            self.policy[(board.board_state(), move)]
            + self.lr * td_error * self.eligibility[(board.board_state(), move)]
        )

    def update_eligibility(self, board, move, elig):
        if elig == 1:
            self.eligibility[(board.board_state(), move)] = elig
        else:
            self.eligibility[(board.board_state(), move)] = (
                self.discount
                * self.eli_dec
                * self.eligibility[(board.board_state(), move)]
            )

    def reset_eligibility(self, board):
        if board.check_losing_state() or board.check_winning_state():
            return 0
        else:
            moves = board.get_all_legal_moves()
            for move in moves:
                self.eligibility[(board.board_state(), move)] = 0
                board_copy = copy.deepcopy(board)
                board_copy.make_move(move)
                self.reset_eligibility(board_copy)
            return 0
