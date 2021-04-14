class Parameters:
    def __init__(self):
        # Board parameters
        self.board_size = 5
        self.starting_player = 0 # 0 for mix
        # MCTS parameters
        self.number_of_games = 200
        self.number_of_search_episodes = 600
        self.epsilon = 1
        self.epsilon_decay = 0.99
        # ANET parameters
        self.lr = 0.005
        self.batch_size = 64
        self.nn_dims = (18, 24, 18)
        self.activation_function = ["sigmoid", "tanh"] # [0] = actor, [1] = critic
        self.optimizer = "adam"
        # With sigma=1.5 and decay=0.97, first chance of critic eval is at episode 14
        self.sigma = 2
        self.sigma_decay = 0.996
        # TOPP parameters
        self.number_of_cached_anet = 5 # + 1 for episode 0
        self.topp = True
        self.topp_games = 4
        self.visualize_last_game = True
        # Whether or not to play OHT
        self.oht = False
        # Episode number of actor to play OHT
        self.oht_episode = 160