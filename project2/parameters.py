class Parameters:
    def __init__(self):
        # Board parameters
        self.board_size = 10
        # MCTS parameters
        self.number_of_episodes = 100
        self.number_of_search_episodes = 1000
        self.epsilon = 0.99
        self.epsilon_decay = 0.95
        # ANET parameters
        self.lr = 0.001
        self.nn_dims = (15, 15, 20, 25, 30, 25, 20, 15)
        self.activation_function = "sigmoid"
        self.optimizer = "adam"
        self.model_name = ""
        # TOPP parameters
        self.number_of_cached_anet = 5
        self.number_of_games = 100
