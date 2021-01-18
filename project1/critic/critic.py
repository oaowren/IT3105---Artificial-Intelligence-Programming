from .table_lookup_critic import TableLookupCritic

class Critic():

    def __init__(self, board, method="TL", nn_dimensions=None, lr=0.5, eligibility_decay=0, discount_factor=0.5):
        self.board = board
        if method == "TL":
            self.critic = TableLookupCritic(board, lr, eligibility_decay, discount_factor)
        elif method == "NN":
            # TODO: NN-critic
            self.critic = None
        else: 
            raise Exception("Method must be either 'NN' (Neural Network) or 'TL' (Table Lookup)")
    
    def generate_game_states(self):
        return self.critic.generate_game_states(self.board)