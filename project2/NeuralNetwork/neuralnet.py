from tensorflow import keras as ks
import numpy as np

# Static values used to select activation function
acts = {
    "linear": ks.activations.linear,
    "sigmoid": ks.activations.sigmoid,
    "tanh": ks.activations.tanh,
    "relu": ks.activations.relu,
}

# Static values used to select optimizer
optimizers = {
    "adagrad": ks.optimizers.Adagrad,
    "sgd": ks.optimizers.SGD,
    "rmsprop": ks.optimizers.RMSprop,
    "adam": ks.optimizers.Adam,
}


class NeuralNet:
    def __init__(
        self,
        nn_dims,
        board_size,
        lr,
        activation,
        optimizer,
        load_saved_model=False,
        model_name="",
        episode_number=0,
    ):
        if load_saved_model:
            try:
                self.model = self.load_saved_model(model_name, episode_number)
            except OSError:
                raise ValueError(
                    "Failed to load model named {0}{1}, did you provide correct model name and episode number?".format(
                        model_name, episode_number
                    )
                )
        else:
            self.model = self.init_model(nn_dims, board_size, lr, activation, optimizer)
        self.topp = load_saved_model
        self.board_size = board_size

    def init_model(self, nn_dims, board_size, lr, activation, optimizer):
        model = ks.Sequential()
        activation_function = acts.get(activation, None)
        if activation_function is None:
            raise ValueError(
                "Invalid activation function provided (must be either '{0}')".format(
                    "', '".join(acts.keys())
                )
            )
        model.add(ks.Input(shape=(board_size ** 2)))
        # Ensure that input has correct shape
        model.add(ks.layers.Dense(board_size ** 2, activation=activation_function))
        for i in nn_dims:
            model.add(ks.layers.Dense(i, activation=activation_function))
        # Ensure that output has correct shape and activation softmax
        model.add(ks.layers.Dense(board_size ** 2, activation="softmax"))
        opt = optimizers.get(optimizer, None)
        if opt is None:
            raise ValueError(
                "Invalid optimizer provided (must be either '{0}')".format(
                    "', '".join(optimizers.keys())
                )
            )
        model.compile(
            optimizer=opt(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()
        return model

    def fit(self, inputs, targets, batch_size=None, epochs=1, verbosity=1):
        if self.topp:
            raise Exception("Model should not train during TOPP")
        train_x, train_y, valid_x, valid_y = self.train_test_split(inputs, targets)
        train_x, train_y = self.random_minibatch(train_x, train_y, len(train_x))
        self.model.fit(
            train_x, train_y, epochs=epochs, verbose=verbosity, batch_size=batch_size
        )
        self.model.evaluate(valid_x, valid_y, verbose=verbosity)

    def predict(self, inputs, flat_board):
        predictions = self.model.predict(inputs)
        pred_length = len(predictions)
        illegal_moves_removed = np.array(
            [
                [
                    predictions[n][i] if flat_board[i] == 0 else 0
                    for i in range(len(predictions[n]))
                ]
                for n in range(pred_length)
            ]
        )
        return np.array([NeuralNet.normalize(illegal_moves_removed[i]) for i in range(pred_length)])

    def best_action(self, normalized_predictions):
        i = np.argmax(normalized_predictions)
        return (i % self.board_size, i//self.board_size)

    def save_model(self, model_name, episode_number):
        self.model.save("project2/models/{0}{1}.h5".format(model_name, episode_number))
        print("Model {0}{1} saved succesfully".format(model_name, episode_number))

    def load_saved_model(self, model_name, episode_number):
        model = ks.models.load_model(
            "project2/models/{0}{1}.h5".format(model_name, episode_number)
        )
        print("Model {0}{1} loaded succesfully".format(model_name, episode_number))
        return model

    def random_minibatch(self, inputs, targets, size=25):
        indices = np.random.randint(len(inputs), size=size)
        return inputs[indices], targets[indices]

    def train_test_split(self, inputs, targets, split=0.1, randomize=True):
        vc = round(split * len(inputs))
        if split > 0:
            pairs = list(zip(inputs, targets))
            if randomize:
                np.random.shuffle(pairs)
            vcases = pairs[0:vc]
            tcases = pairs[vc:]
            return (
                np.array([tc[0] for tc in tcases]),
                np.array([tc[1] for tc in tcases]),
                np.array([vc[0] for vc in vcases]),
                np.array([vc[1] for vc in vcases]),
            )
        else:
            return inputs, targets, [], []

    @staticmethod
    def normalize(arr):
        # Assumes input of 1d np-array
        arrsum = sum(arr)
        return arr/arrsum
