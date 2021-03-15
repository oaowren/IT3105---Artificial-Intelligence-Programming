from .board import Board

class GameSimulator:

    def __init__(self, board_size, starting_player, tree):
        self.board = Board(board_size)
        self.board_size = board_size
        self.starting_player = starting_player
        self.player = starting_player
        self.tree = tree
        self.state_action = {}

    def change_player(self):
        self.player = 1 if self.player == 2 else 2

    def initialize_root(self, root):
        player = root[0]
        state_split = root[1].split()
        state = [[int(i) for i in state_split[n*self.board_size:(n+1)*self.board_size]] for n in range(self.board_size)]
        print(state)
        self.player = player
        self.board.board = state


    def rollout_game(self, epsilon):
        while not self.board.check_winning_state(0):
            next_move = self.tree.rollout_action(self.board.board_state(), epsilon, self.player)
            self.state_action[(self.board.board_state(), self.player)] = next_move
            self.board.make_move(next_move, self.player)
            self.change_player()

    def reset(self):
        self.board = Board(self.board_size)
        self.state_action = {}
        self.player = self.starting_player