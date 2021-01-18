class TableLookupCritic():

    def __init__(self, board, lr, eligibility_decay, discount_factor):
        self.board = board
        self.sap = {}
    

    def generate_game_states(self, board):
        if board.check_winning_state():
            return 1
        elif board.check_losing_state():
            return -1
        else: 
            moves = board.get_all_legal_moves()
            for move in moves:
                board_copy = board
                board_copy.make_move(move)
                self.sap[board_copy]=self.generate_game_states(board_copy)
        return self.sap
