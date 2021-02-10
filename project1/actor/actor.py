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
        self.alpha = 0.8
        self.gamma = 0.8
        self.lam = 0.99
        self.policy = {}
        self.eligibility = {}

    def init_policy(self, board):
        moves = board.get_all_legal_moves()
        for move in moves:
            self.policy[(board.board_state(), move)] = 0

    def select_action(self, board):
        moves = board.get_all_legal_moves()
        random_choice = random.random()
        if random_choice < self.eps:
            return random.choice(moves)
        else:
            best_move = moves[0]
            best_reward = -10000
            for move in moves:
                if not (board.board_state(), move) in self.policy:
                    self.policy[(board.board_state(), move)] = 0
                move_reward = self.policy[(board.board_state(), move)]
                if move_reward > best_reward:
                    best_move = move
                    best_reward = move_reward
            # print(best_reward)
            return best_move

    def update_policy(self, board_state, move, td_error):
        self.policy[(board_state, move)] = (
            self.policy[(board_state, move)]
            + self.lr * td_error * self.eligibility[(board_state, move)]
        )

    def update_eligibility(self, board_state, move, elig):
        if elig == 1:
            self.eligibility[(board_state, move)] = elig
        else:
            self.eligibility[(board_state, move)] = (
                self.discount
                * self.eli_dec
                * self.eligibility[(board_state, move)]
            )
    def update(self, delta, sequence):
        """
        This updates all the evaluations of states in the episode sequence based on eligibility traces
        """
        self.eligibility[sequence[-1]] = 1
        
        for state,action in sequence:
            if not (state,action) in self.policy:
                self.policy[(state,action)] = 0.1
            self.policy[(state,action)] += self.alpha * delta * self.eligibility[(state,action)]
            self.eligibility[(state,action)] *= self.gamma * self.lam

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
