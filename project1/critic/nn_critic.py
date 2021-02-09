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
        input_length,
        board
    ):
        self.lr = lr
        self.nn_dims = nn_dims
        self.discount_factor = discount_factor
        self.eli_dec = eligibility_decay
        self.eligibility = {}
        self.model = self.init_nn(input_length)
        self.values = {}
        self.td_error = 0
        self.current_state = None


    def init_nn(self, input_length):
        model = ks.Sequential()
        model.add(ks.layers.Embedding(input_length = input_length, input_dim = 10000, output_dim = self.nn_dims[0]))
        for i in self.nn_dims[1:]:
            model.add(ks.layers.Dense(i))
        model.compile(optimizer=ks.optimizers.SGD(learning_rate=(self.lr)), loss=ks.losses.MeanSquaredError(), metrics=['accuracy'])
        model.summary()
        return SGD.SplitGD(model, self)

    def modify_gradients(self, gradients):
        dvs = []
        for i in range(len(gradients)):
            dvs.append(gradients[i] / (-2*self.td_error))
        elig = []
        for i in range(len(dvs)):
            elig.append(dvs[i] + self.eligibility[self.current_state] * self.eli_dec * self.discount_factor)
        self.eligibility[self.current_state] = elig
        for i in range(len(gradients)):
            gradients[i] = gradients[i] + self.lr * self.td_error * self.eligibility[self.current_state][i]
        return gradients

    def calculate_td_error(self, reward, current_state, next_state):
        self.current_state = current_state
        if next_state not in self.values.keys():
            self.values[next_state] = 0
        if current_state not in self.values.keys():
            self.values[current_state] = 0
        self.td_error = reward + self.discount_factor*self.values[next_state] - self.values[current_state]
        return self.td_error

    def update_eligibility(self, board_state, elig):
        self.eligibility[board_state] = elig

    def reset_eligibility(self, board):
        return 0

    def update_value(self, state, value):
        self.values[state] = value
