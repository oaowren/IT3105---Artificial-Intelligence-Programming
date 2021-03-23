import numpy as np
from copy import deepcopy


class Board:
    def __init__(self, board_size, starting_player):
        self.board_size = board_size
        self.board = []
        self.reset_board()
        self.player = starting_player

    def reset_board(self):
        self.board = np.array(
            [[0 for i in range(self.board_size)] for j in range(self.board_size)]
        )

    def get_state(self):
        output = ""
        for i in self.board:
            for n in i:
                if n == 1:
                    output += "1"
                elif n ==2:
                    output += "2"
                else:
                    output +="0"
                output += " "
        return output

    def get_legal_moves(self):
        flat_board = self.flatten_board()
        return [(i // self.board_size, i % self.board_size) for i in range(len(flat_board)) if flat_board[i] == 0]

    def check_legal_move(self, move):
        try:
            return self.board[move[0]][move[1]] == 0
        except IndexError:
            return False

    def make_move(self, move):
        if not self.check_legal_move(move):
            raise Exception("Illegal move provided")
        if self.player != 1 and self.player != 2:
            raise Exception("player must be either 1 or 2")
        self.board[move[0]][move[1]] = self.player
        self.player = self.player % 2 + 1

    def flatten_board(self):
        return self.board.flatten()

    def check_winning_state_player_one(self):
        reachable_nodes = []
        for i in range(self.board_size):
            if self.board[0][i] == 1:
                reachable_nodes.append((0, i))
        for node in reachable_nodes:
            for n in range(-1, 2):
                if (
                    0 <= node[1] + n < self.board_size
                    and self.board[node[0] + 1][node[1] + n] == 1
                ):
                    if node[0] + 1 == self.board_size - 1:
                        return True
                    if (
                        node[0] + 1,
                        node[1] + n,
                    ) not in reachable_nodes:  # Check if node is already added to avoid checking the same nodes twice
                        reachable_nodes.append((node[0] + 1, node[1] + n))

        return False

    def check_winning_state_player_two(self):
        reachable_nodes = []
        for i in range(self.board_size):
            if self.board[i][0] == 2:
                reachable_nodes.append((i, 0))
        for node in reachable_nodes:
            for n in range(-1, 2):
                if (
                    0 <= node[0] + n < self.board_size
                    and self.board[node[0] + n][node[1] + 1] == 2
                ):
                    if node[1] + 1 == self.board_size - 1:
                        return True
                    if (node[0] + n, node[1] + 1) not in reachable_nodes:
                        reachable_nodes.append((node[0] + n, node[1] + 1))

        return False

    def check_winning_state(self, player=0):
        """Since winning state must be checked after every move, this function
        takes in which player played the last move, to only check wheter they
        are in a winning state or not. This is to reduce the complexity of the
        algorithm.
        """
        if player == 1:
            return self.check_winning_state_player_one()
        elif player == 2:
            return self.check_winning_state_player_two()
        elif player == 0:
            return (
                self.check_winning_state_player_one()
                or self.check_winning_state_player_two()
            )

    def get_reward(self, player):
        if (player == 1 and self.check_winning_state_player_one()) \
            or (player == 2 and self.check_winning_state_player_two()):
            return 1
        else:
            return -1

    def clone(self):
        return deepcopy(self)
