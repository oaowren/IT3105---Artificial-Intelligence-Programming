from .board import Board

class GameSimulator:

    def __init__(self, playing_board, board_size, starting_player, tree):
        self.board = Board(board_size)
        self.playing_board = playing_board
        self.board_size = board_size
        self.starting_player = starting_player
        self.player = starting_player
        self.tree = tree
        self.state_action = {}

    def change_player(self):
        self.player = self.player % 2 + 1

    def initialize_root(self, root):
        player = root[0]
        state_split = root[1].split()
        state = [[int(i) for i in state_split[n*self.board_size:(n+1)*self.board_size]] for n in range(self.board_size)]
        self.player = player
        self.board.board = state


    def rollout_game(self, epsilon):
        while not self.board.check_winning_state(0):
            next_move = self.tree.rollout_action(self.board.board_state(), epsilon, self.player)
            self.board.make_move(next_move, self.player)
            self.change_player()

    def tree_search(self):
        sequence = self.tree.traverse(self.board)
        self.player = sequence[-1][0] % 2 + 1 
        for i in sequence:
            self.state_action[(i[0],i[1])] = i[2]

    def sim_games(self, epsilon, number_of_search_games):
        for _ in number_of_search_games:
            self.tree_search()
            self.rollout_game(epsilon)
            rewards = {1:self.board.get_reward(1), 2: self.board.get_reward(2)}
            for key in self.state_action.keys():
                self.tree.update(key, self.state_action[key], rewards[key[0]])
        return self.tree.get_distribution(self.playing_board)

    def reset(self):
        self.board = Board(self.board_size)
        self.state_action = {}
        self.player = self.starting_player