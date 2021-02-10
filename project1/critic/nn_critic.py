import tensorflow as tf
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
        model.add(ks.Input(shape=(input_length, )))
        for i in self.nn_dims:
            model.add(ks.layers.Dense(i))
        model.compile(optimizer=ks.optimizers.SGD(learning_rate=(self.lr)), loss=ks.losses.MeanSquaredError(), metrics=['accuracy'])
        model.summary()
        return SGD.SplitGD(model, self)

    def modify_gradients(self, gradients):
        dvs = []
        for i in range(1,len(gradients)):
            result = tf.math.divide(gradients[i], -2*self.td_error)
            dvs.append(result)
        elig = []
        for i in range(len(dvs)):
            try: 
                elig.append(dvs[i] + self.eligibility[self.current_state][i] * self.eli_dec * self.discount_factor)
            except TypeError:
                elig.append(dvs[i] + self.eligibility[self.current_state] * self.eli_dec * self.discount_factor)
        self.eligibility[self.current_state] = elig
        for i in range(1,len(gradients)):
            gradients[i] += self.lr * self.td_error * elig[i-1]
        return gradients

    def calculate_td_error(self, current_state, next_state, reward):
        self.current_state = current_state
        if next_state not in self.values.keys():
            self.values[next_state] = random.uniform(0, 0.2)
        if current_state not in self.values.keys():
            self.values[current_state] = random.uniform(0, 0.2)
        self.td_error = reward + self.discount_factor*self.values[next_state] - self.values[current_state]
        return self.td_error

    def update_eligibility(self, board_state, elig):
        self.eligibility[board_state] = elig

    def update_expected_reward(self, sequence):
        self.model.fit([[int(x) for x in state.split()] for state, reward in sequence], [reward for _, reward in sequence])
        for (state, _) in sequence:
            pred = self.model.model.predict([[int(x) for x in state.split()]])
            self.values[state] = pred[0][0]

    def reset_eligibility(self):
        self.eligibility = {}

    def update_value(self, state, value):
        self.values[state] = value
