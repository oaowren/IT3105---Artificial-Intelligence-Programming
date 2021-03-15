class GameSimulator:

    def __init__(self, board, starting_player, tree):
        self.board = board.clone()
        self.player = starting_player
        self.tree = tree
        self.state_action = {}

    def change_player(self):
        self.player = 1 if self.player == 2 else 2

    def rollout_game(self, epsilon):
        while not self.board.check_winning_state(0):
            next_move = self.tree.rollout_action(self.board.board_state(), epsilon, self.player)
            self.state_action[(self.board.board_state(), self.player)] = next_move
            self.board.make_move(next_move, self.player)
            self.change_player()

    def reset_states(self):
        self.state_action = {}