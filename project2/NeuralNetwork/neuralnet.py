from tensorflow import keras as ks

# Static values used to select activation function
acts = {
    "linear": ks.activations.linear,
    "sigmoid": ks.activations.sigmoid,
    "tanh": ks.activations.tanh,
    "relu": ks.activations.relu
}

# Static values used to select optimizer
optimizers = {
    "adagrad": ks.optimizers.Adagrad,
    "sgd": ks.optimizers.SGD,
    "rmsprop": ks.optimizers.RMSprop,
    "adam": ks.optimizers.Adam
}

class NeuralNet():

    def __init__(self, nn_dims, board_size, lr, activation, optimizer, load_saved_model=False, model_name="", episode_number=0):
        if load_saved_model:
            self.model = self.load_saved_model(model_name, episode_number)
        else: 
            self.model = self.init_model(nn_dims, board_size, lr, activation, optimizer)

    def init_model(self, nn_dims, board_size, lr, activation, optimizer):
        model = ks.Sequential()
        activation_function = self.select_activation(activation)
        if activation_function is None:
            raise ValueError("Invalid activation function provided (must be either '{0}')".format("', '".join(acts.keys())))
        model.add(ks.Input(shape=(board_size**2, )))
        model.add(ks.layers.Dense(board_size**2, activation=activation_function))
        for i in nn_dims:
            model.add(ks.layers.Dense(i, activation=activation_function))
        model.add(ks.layers.Dense(board_size**2, activation="softmax"))
        o = self.select_optimizer(optimizer)
        if o is None:
            raise ValueError("Invalid optimizer provided (must be either '{0}')".format("', '".join(optimizers.keys())))
        model.compile(optimizer=o(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=["accuracy"])
        model.summary()
        return model

    def select_activation(self, activation_function):
        return acts.get(activation_function, None)

    def select_optimizer(self, optimizer):
        return optimizers.get(optimizer, None)

    def save_model(self, model_name, episode_number):
        self.model.save("project2/models/{0}{1}.h5".format(model_name, episode_number))
        print("Model {0}{1} saved succesfully".format(model_name, episode_number))

    def load_saved_model(self, model_name, episode_number):
        model = ks.models.load_model("project2/models/{0}{1}.h5".format(model_name, episode_number))
        print("Model {0}{1} loaded succesfully".format(model_name, episode_number))
        return model