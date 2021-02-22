from tensorflow import keras as ks

class NeuralNet():

    def __init__(self, nn_dims, board_size, lr, activation, optimizer):
        self.model = self.init_model(nn_dims, board_size, lr, activation, optimizer)

    def init_model(self, nn_dims, board_size, lr, activation, optimizer):
        model = ks.Sequential()
        activation_function = self.select_activation(activation)
        if activation_function is None:
            raise ValueError("Invalid activation function provided (must be either 'linear', 'sigmoid', 'tanh' or 'relu')")
        model.add(ks.Input(shape=(board_size**2, )))
        model.add(ks.layers.Dense(board_size**2, activation=activation_function))
        for i in nn_dims:
            model.add(ks.layers.Dense(i, activation=activation_function))
        model.add(ks.layers.Dense(board_size**2, activation="softmax"))
        o = self.select_optimizer(optimizer)
        if o is None:
            raise ValueError("Invalid optimizer provided (must be either 'adagrad', 'sgd', 'rmsprop' or 'adam')")
        model.compile(optimizer=o(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=["accuracy"])
        model.summary()
        return model

    def select_activation(self, activation_function):
        acts = {
            "linear": ks.activations.linear,
            "sigmoid": ks.activations.sigmoid,
            "tanh": ks.activations.tanh,
            "relu": ks.activations.relu
        }
        return acts.get(activation_function, None)

    def select_optimizer(self, optimizer):
        optimizers = {
            "adagrad": ks.optimizers.Adagrad,
            "sgd": ks.optimizers.SGD,
            "rmsprop": ks.optimizers.RMSprop,
            "adam": ks.optimizers.Adam
        }
        return optimizers.get(optimizer, None)