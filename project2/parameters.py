class Parameters:
    def __init__(self):
        # Board parameters
        self.board_size = 4
        self.starting_player = 0 # 0 for mix
        # MCTS parameters
        self.number_of_games = 20
        self.number_of_search_episodes = 30
        self.epsilon = 1
        self.epsilon_decay = 0.98
        # ANET parameters
        self.lr = 0.001
        self.batch_size = 64
        self.nn_dims = (10, 20, 40, 15, 9)
        self.activation_function = "sigmoid"
        self.optimizer = "adam"
        self.sigma = 1.5
        self.sigma_decay = 0.98
        # TOPP parameters
        self.number_of_cached_anet = 5
        self.topp = True
        self.topp_games = 10
        self.oht = False