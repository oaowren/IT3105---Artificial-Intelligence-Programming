from MCTS.node import Node
import numpy as np
from copy import *


class Board:
    def __init__(self, board_size, starting_player):
        self.board_size = board_size
        self.board = []
        self.player = starting_player
        self.reset_board(self.player)

    def reset_board(self, starting_player):
        self.board = np.array(
            [[0 for i in range(self.board_size)] for j in range(self.board_size)]
        )
        self.player = starting_player

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

    def get_legal_moves(self, board):
        # Assumes a nxn board
        board_size = len(board)
        flat_board = self.flatten_board(board)
        return [(i // board_size, i % board_size) for i in range(len(flat_board)) if flat_board[i] == 0]

    def check_legal_move(self, move, board):
        try:
            return board[move[0]][move[1]] == 0
        except IndexError:
            return False

    def get_child_states(self, player, state):
        moves = self.get_legal_moves(state)
        states = [self.get_next_state(state, move, player) for move in moves]
        return [Node(state, action, player) for (state, action) in zip(states, moves)]    

    def get_next_state(self, board, move, player):
        if not self.check_legal_move(move, board):
            raise Exception(f"Illegal move provided: {move} {self.flatten_board(board)}")
        if player != 1 and player != 2:
            raise Exception("player must be either 1 or 2")
        new_board = copy(board)
        new_board[move[0]][move[1]] = player
        return new_board

    def make_move(self, move, player):
        if not self.check_legal_move(move, self.board):
            raise Exception(f"Illegal move provided: {move} {self.flatten_board(self.board)}")
        if self.player != 1 and self.player != 2:
            raise Exception("player must be either 1 or 2")
        self.board[move[0]][move[1]] = player


    def flatten_board(self, board):
        return board.flatten()

    def check_winning_state_player_one(self, board):
        board_size = len(board)
        reachable_nodes = []
        for i in range(board_size):
            if board[0][i] == 1:
                reachable_nodes.append((0, i))
        for node in reachable_nodes:
            for n in range(-1, 2):
                if (
                    0 <= node[1] + n < board_size
                    and board[node[0] + 1][node[1] + n] == 1
                ):
                    if node[0] + 1 == board_size - 1:
                        return True
                    if (
                        node[0] + 1,
                        node[1] + n,
                    ) not in reachable_nodes:  # Check if node is already added to avoid checking the same nodes twice
                        reachable_nodes.append((node[0] + 1, node[1] + n))

        return False

    def check_winning_state_player_two(self, board):
        board_size = len(board)
        reachable_nodes = []
        for i in range(board_size):
            if board[i][0] == 2:
                reachable_nodes.append((i, 0))
        for node in reachable_nodes:
            for n in range(-1, 2):
                if (
                    0 <= node[0] + n < board_size
                    and board[node[0] + n][node[1] + 1] == 2
                ):
                    if node[1] + 1 == self.board_size - 1:
                        return True
                    if (node[0] + n, node[1] + 1) not in reachable_nodes:
                        reachable_nodes.append((node[0] + n, node[1] + 1))

        return False

    def check_winning_state(self, board, player=0):
        """Since winning state must be checked after every move, this function
        takes in which player played the last move, to only check wheter they
        are in a winning state or not. This is to reduce the complexity of the
        algorithm.
        """
        if player == 1:
            return self.check_winning_state_player_one(board)
        elif player == 2:
            return self.check_winning_state_player_two(board)
        elif player == 0:
            return (
                self.check_winning_state_player_one(board)
                or self.check_winning_state_player_two(board)
                or len(self.get_legal_moves(board)) == 0
            )

    def get_reward(self, player):
        if (player == 1 and self.check_winning_state_player_one(self.board)) \
            or (player == 2 and self.check_winning_state_player_two(self.board)):
            return 1
        else:
            return -1

    def clone(self):
        return deepcopy(self)
