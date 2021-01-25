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

    def make_move(self, move):
        # A move consists of two tuples (x,y) where x denotes the number of the row and y the column
        middle = ((move[0][0] + move[1][0])//2,
                  (move[0][1] + move[1][1])//2)
        if not self.check_legal_move(move):
            raise Exception(
                "Illegal move. A peg must jump over a peg to an empty spot to be a legal move.")
        self.pre_move()
        self.board[move[0][0]][move[0][1]] = 0
        self.board[middle[0]][middle[1]] = 0
        # 2 indicates recently moved peg
        self.board[move[1][0]][move[1][1]] = 2

    def board_state(self):
        output = ""
        for i in self.board:
            for n in i:
                if n == 2:
                    # Could end in same game state through different moves, important to not distinguish between these
                    output += "1"
                output += str(n)
        return output

    def check_legal_move(self, move):
        middle = ((move[0][0] + move[1][0])//2,
                  (move[0][1] + move[1][1])//2)
        if self.board[move[0][0]][move[0][1]] == 0 or self.board[middle[0]][middle[1]] == 0:
            return False
        dist = abs(move[1][0] + move[1][1] - (move[0][0] + move[0][1]))
        if move[0][0] != move[1][0] and move[0][1] != move[1][1]:
            if self.board_type == "T":
                return dist == 4 and abs(move[0][0] - move[1][0]) == 2
            if self.board_type == "D":
                return dist == 0 and abs(move[0][0] - move[1][0]) == 2
        else:
            return dist == 2

    def pre_move(self):
        # Remove the previous 2 to only show the last move
        index = self.find_indices(2)
        if len(index) != 0:
            self.board[index[0][0]][index[0][1]] = 1

    def find_indices(self, state):
        indices = []
        for i in range(len(self.board)):
            for n in range(len(self.board[i])):
                if self.board[i][n]==state:
                    indices.append((i,n))
        return indices

    def get_all_legal_moves(self):
        legal_moves = []
        open_slots = self.find_indices(0)
        for slot in open_slots:
            for x in range(len(self.board)):
                for y in range(len(self.board[x])):
                    if self.check_legal_move(((x,y), slot)):
                        legal_moves.append(((x,y), slot))
        return legal_moves

    def check_winning_state(self):
        index1 = self.find_indices(1)
        index2 = self.find_indices(2)
        indices = index1+index2
        return len(indices)==1

    def check_losing_state(self):
        moves=self.get_all_legal_moves()
        return not self.check_winning_state() and len(moves)==0