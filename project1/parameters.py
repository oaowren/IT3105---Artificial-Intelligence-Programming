class Parameters():

    def __init__(self):
        # ------ VARIABLES --------
        # Board and Game Variables
        self.board_type = "D"  # "T" or "D"
        self.board_size = 5
        # For board_type = "D" and board_size = 4, open_cells must be either (1,2) or (2,1)
        self.open_cells = [(1, 1)]
        self.number_of_episodes = 600
        # Rewards
        self.winning_reward = 20
        self.losing_reward_per_peg = -3
        self.discount_per_step = -0.1
        # Visualization
        self.display_episode = True  # Display final run
        self.display_delay = 2  # Number of seconds between board updates in visualization

        # Critic Variables
        self.critic_method = "NN"  # "TL" or "NN"
        # First input parameter must be equal to number of holes on board, e.g. type D size 4 = 16
        self.critic_nn_dims = (25, 25, 30, 10, 1)
        self.lr_critic = 0.001
        self.eligibility_decay_critic = 0.95
        self.discount_factor_critic = 0.95

        # Actor Variables
        self.lr_actor = 0.1
        self.eligibility_decay_actor = 0.9
        self.discount_factor_actor = 0.95
        self.epsilon = 0.9
        self.epsilon_decay = 0.96

        #Board vizualization
        self.height = 800
        self.width = 1000
        # -------------------------

    def scenario_triangle_size5_nn(self):
        self.board_type = "T"
        self.board_size = 5
        self.open_cells = [(2, 1)]
        self.number_of_episodes = 750
        self.winning_reward = 100
        self.losing_reward_per_peg = -5
        self.discount_per_step = -1
        self.critic_method = "NN"
        self.critic_nn_dims = (15, 25, 30, 10, 1)
        self.lr_critic = 0.0001
        self.eligibility_decay_critic = 0.96
        self.discount_factor_critic = 0.96
        self.lr_actor = 0.01
        self.eligibility_decay_actor = 0.9
        self.discount_factor_actor = 0.95
        self.epsilon = 0.9
        self.epsilon_decay = 0.99

    def scenario_diamond_size4_nn(self):
        self.board_type = "D"
        self.board_size = 4
        self.open_cells = [(2, 1)]
        self.number_of_episodes = 1000
        self.winning_reward = 20
        self.losing_reward_per_peg = -1
        self.discount_per_step = -0.1
        self.critic_method = "NN"
        self.critic_nn_dims = (16, 25, 30, 10, 1)
        self.lr_critic = 0.001
        self.eligibility_decay_critic = 0.95
        self.discount_factor_critic = 0.95
        self.lr_actor = 0.1
        self.eligibility_decay_actor = 0.9
        self.discount_factor_actor = 0.95
        self.epsilon = 0.9
        self.epsilon_decay = 0.99

    def scenario_triangle_size5_tl(self):
        self.board_type = "T"
        self.board_size = 5
        self.open_cells = [(2, 2)]
        self.number_of_episodes = 600
        self.winning_reward = 20
        self.losing_reward_per_peg = -3
        self.discount_per_step = -0.1
        self.critic_method = "TL"
        self.lr_critic = 0.01
        self.eligibility_decay_critic = 0.95
        self.discount_factor_critic = 0.95
        self.lr_actor = 0.1
        self.eligibility_decay_actor = 0.9
        self.discount_factor_actor = 0.95
        self.epsilon = 0.9
        self.epsilon_decay = 0.96

    def scenario_diamond_size4_tl(self):
        self.board_type = "D"
        self.board_size = 4
        self.open_cells = [(2, 1)]
        self.number_of_episodes = 600
        self.winning_reward = 20
        self.losing_reward_per_peg = -3
        self.discount_per_step = -0.1
        self.critic_method = "TL"
        self.lr_critic = 0.01
        self.eligibility_decay_critic = 0.95
        self.discount_factor_critic = 0.95
        self.lr_actor = 0.1
        self.eligibility_decay_actor = 0.9
        self.discount_factor_actor = 0.95
        self.epsilon = 0.9
        self.epsilon_decay = 0.96

    def test_tl(self):
        self.board_type = "D"
        self.board_size = 4
        self.open_cells = [(0, 0)]
        self.number_of_episodes = 600
        self.winning_reward = 20
        self.losing_reward_per_peg = -10
        self.discount_per_step = -0.1
        self.critic_method = "TL"
        self.lr_critic = 0.01
        self.eligibility_decay_critic = 0.95
        self.discount_factor_critic = 0.95
        self.lr_actor = 0.1
        self.eligibility_decay_actor = 0.9
        self.discount_factor_actor = 0.95
        self.epsilon = 0.9
        self.epsilon_decay = 0.96

    def test_nn(self):
        self.board_type = "D"
        self.board_size = 4
        self.open_cells = [(0, 0)]
        self.number_of_episodes = 600
        self.winning_reward = 20
        self.losing_reward_per_peg = -10
        self.discount_per_step = -0.1
        self.critic_method = "NN"
        self.critic_nn_dims = (16, 25, 30, 10, 1)
        self.lr_critic = 0.01
        self.eligibility_decay_critic = 0.95
        self.discount_factor_critic = 0.95
        self.lr_actor = 0.1
        self.eligibility_decay_actor = 0.9
        self.discount_factor_actor = 0.95
        self.epsilon = 0.9
        self.epsilon_decay = 0.96