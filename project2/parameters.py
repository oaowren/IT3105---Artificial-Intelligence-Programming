class Parameters:
    def __init__(self):
        # Board parameters
        self.board_size = 6
        self.starting_player = 1
        # MCTS parameters
        self.number_of_games = 100
        self.number_of_search_episodes = 500
        self.epsilon = 1
        self.epsilon_decay = 1
        # ANET parameters
        self.lr = 0.0001
        self.nn_dims = (15, 20, 25, 30, 30, 20, 15)
        self.activation_function = "sigmoid"
        self.optimizer = "adam"
        # TOPP parameters
        self.number_of_cached_anet = 5
        self.topp = False
        self.topp_games = 20