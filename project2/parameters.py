class Parameters:
    def __init__(self):
        # Board parameters
        self.board_size = 6
        self.starting_player = 0 # 0 for mix
        # MCTS parameters
        self.number_of_games = 100
        self.number_of_search_episodes = 100
        self.epsilon = 1
        self.epsilon_decay = 0.97
        # ANET parameters
        self.lr = 0.01
        self.nn_dims = (60, 60, 40)
        self.activation_function = "relu"
        self.optimizer = "adam"
        # TOPP parameters
        self.number_of_cached_anet = 5
        self.topp = False
        self.topp_games = 20