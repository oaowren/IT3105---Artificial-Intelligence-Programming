import tensorflow as tf
from tensorflow import keras as ks
import critic.splitgd as SGD
import random

class CriticNN:
    def __init__(
        self,
        lr,
        nn_dims,
        eligibility_decay,
        discount_factor
    ):
        self.alpha = lr
        self.nn_dims = nn_dims
        self.gamma = discount_factor
        self.lam = eligibility_decay
        self.eligibility = {}
        self.model = self.init_nn(nn_dims[0])
        self.expected_reward = {}
        self.delta = 0
        self.current_state = None


    def init_nn(self, input_length):
        model = ks.Sequential()
        model.add(ks.Input(shape=(input_length, )))
        for i in self.nn_dims:
            model.add(ks.layers.Dense(i))
        model.compile(optimizer=ks.optimizers.SGD(learning_rate=(self.alpha)), loss=ks.losses.MeanSquaredError(), metrics=['accuracy'])
        model.summary()
        return SGD.SplitGD(model, self)

    def modify_gradients(self, gradients):
        dvs = []
        for i in range(1,len(gradients)):
            result = tf.math.divide(gradients[i], -2*self.delta)
            dvs.append(result)
        elig = []
        for i in range(len(dvs)):
            elig.append(dvs[i] + self.eligibility[self.current_state])
        for i in range(1,len(gradients)):
            gradients[i] += self.alpha * self.delta * elig[i-1]
        return gradients

    def calculate_td_error(self, old_state, new_state, reward):
        self.current_state = old_state
        for state in [old_state, new_state]:
            if state not in self.expected_reward:
                self.expected_reward[state] = random.uniform(0, 0.2)
        td_error = reward + self.gamma*self.expected_reward[new_state] - self.expected_reward[old_state]
        self.delta = td_error
        return td_error

    def update_expected_reward(self, sequence):
        if len(sequence) == 2:
            self.eligibility[sequence[0][0]] = self.gamma * self.lam
        self.eligibility[sequence[-1][0]] = 1
        for state, _ in sequence[:-1]:
            self.eligibility[state] *=self.gamma * self.lam

    def update_model(self, sequence):
        inputs = [[int(x) for x in state.split()] for state, _ in sequence[:-1]]
        targets = [self.expected_reward[state] * self.gamma + reward for state, reward in sequence[1:]]
        self.model.fit(inputs, targets)
        preds = self.model.model.predict([[int(x) for x in state.split()] for state, _ in sequence])
        for i in range(len(sequence)-1):
            self.expected_reward[sequence[i][0]] = preds[i][0]