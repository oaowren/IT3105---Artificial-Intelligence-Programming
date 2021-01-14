import numpy as np


class Board():

    def __init__(self, board_type="D", board_size=4, open_cells=[(2, 2)]):
        # Set board type to access in check_legal_move
        self.board_type = board_type
        if board_type == "D":
            self.board = np.array([[1 for i in range(board_size)]
                                   for j in range(board_size)])
        elif board_type == "T":
            self.board = np.array([[1 for i in range(n+1)]
                                   for n in range(board_size)])
        else:
            raise Exception(
                "Board type must be either 'D' (Diamond) or 'T' (Triangle)")
        try:
            for cell in open_cells:
                self.board[cell[0]][cell[1]] = 0
        except IndexError:
            raise Exception("Index of open cells must all be within the board")

    def print_board(self):
        for i in self.board:
            print(i)
        print("-------------------\n")

    def make_move(self, move_from, move_to):
        # A move consists of two tuples (x,y) where x denotes the number of the row and y the column
        middle = ((move_from[0] + move_to[0])//2,
                  (move_from[1] + move_to[1])//2)
        if not self.check_legal_move(move_from, move_to, middle):
            raise Exception(
                "Illegal move. A peg must jump over a peg to an empty spot to be a legal move.")
        self.pre_move()
        self.board[move_from] = 0
        self.board[middle] = 0
        # 2 indicates recently moved peg
        self.board[move_to] = 2

    def check_legal_move(self, move_from, move_to, middle):
        if self.board_type == "D" and abs(move_to[0] + move_to[1] - move_from[0] + move_from[1]) == 4:
            return False
        if self.board_type == "T" and move_to[0] + move_to[1] - move_from[0] + move_from[1] == 0:
            return False
        if move_from[0] == move_to[0]:
            if abs(move_to[1] - move_from[1]) != 2:
                return False
        if move_from[1] == move_to[1]:
            if abs(move_to[0] - move_from[0]) != 2:
                return False

        return self.board[move_from] and not self.board[move_to] and self.board[middle]

    def pre_move(self):
        # Remove the previous 2 to only show the last move
        index = np.where(self.board == 2)
        if len(index[0]) != 0:
            self.board[index] = 1
