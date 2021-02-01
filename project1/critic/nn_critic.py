from tensorflow import keras as ks
import critic.splitgd as SGD
import random
import copy

class CriticNN:
    def __init__(
        self,
        lr,
        nn_dims,
        eligibility_decay,
        discount_factor,
        input_length
    ):
        self.lr = lr
        self.nn_dims = nn_dims
        self.discount_factor = discount_factor
        self.eli_dec = eligibility_decay
        self.eligibility = {}
        self.model = self.init_nn(input_length)
        self.values = {}


    def init_nn(self, input_length):
        model = ks.Sequential()
        model.add(ks.layers.Embedding(input_length = input_length, input_dim = 10000, output_dim = self.nn_dims[0]))
        for i in self.nn_dims[1:]:
            model.add(ks.layers.Dense(i))
        model.compile(optimizer=ks.optimizers.SGD(learning_rate=(self.lr)), loss=ks.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        model.summary()
        return SGD.SplitGD(model, self)

    def modify_gradients(self, gradients):
        
        return gradients

    def td_error(self, reward, val_state, val_next_state):
        return reward + self.discount_factor*val_next_state - val_state

    def update_eligibility(self, board_state, elig):
        if elig == 1:
            self.eligibility[board_state] = elig
        else:
            self.eligibility[board_state] = (
                self.discount_factor
                * self.eli_dec
                * self.eligibility[board_state]
            )

    def reset_eligibility(self, board):
        return 0

    def init_value(self, board):
        moves = board.get_all_legal_moves()
        for move in moves:
            self.values[board.board_state()] = random.uniform(0.0, 0.2)
            board_copy = copy.deepcopy(board)
            board_copy.make_move(move)
            self.init_value(board_copy)
        return 0