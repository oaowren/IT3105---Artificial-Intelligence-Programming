import copy

class Critic:
    def __init__(
        self,
        lr,
        eligibility_decay,
        discount_factor,
    ):
        self.lr = lr
        self.discount_factor = discount_factor
        self.eli_dec = eligibility_decay
        self.eligibility = {}
        self.values = 0 # Datastruktur


    def init_tl(self):
        #TODO: Iterate game states, set win to R and loss to -R
        return self.lr

    def td_error(self, reward, val_state, val_next_state):
        return reward + self.discount_factor*val_next_state - val_state

    def update_eligibility(self, board_state, elig):
        if elig == 1:
            self.eligibility[board_state] = elig
        else:
            self.eligibility[board_state] = (
                self.discount_factor
                * self.eli_dec
                * self.eligibility[board_state]
            )

    def reset_eligibility(self, board):
        if board.check_losing_state() or board.check_winning_state():
            return 0
        else:
            moves = board.get_all_legal_moves()
            for move in moves:
                self.eligibility[board.board_state()] = 0
                board_copy = copy.deepcopy(board)
                board_copy.make_move(move)
                self.reset_eligibility(board_copy)
            return 0
    
    def update_value(self, board_state, td_error):
        self.values = self.values(board_state) + self.lr * td_error * self.eligibility[board_state]
