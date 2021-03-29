import random
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
        nn_dims = (10),
        board_size = 3,
        lr = 0.01,
        activation = "sigmoid",
        optimizer  = "adam",
        load_saved_model=False,
        episode_number=0,
    ):
        self.board_size = board_size
        if load_saved_model:
            try:
                self.model = self.load_saved_model(episode_number)
            except OSError:
                raise ValueError(
                    "Failed to load model named {0}{1}, did you provide episode number?".format(f"{self.board_size}x{self.board_size}_ep", episode_number)
                )
        else:
            self.model = self.init_model(nn_dims, board_size, lr, activation, optimizer)
        self.topp = load_saved_model

    def init_model(self, nn_dims, board_size, lr, activation, optimizer):
        activation_function = acts.get(activation, None)
        if activation_function is None:
            raise ValueError(
                "Invalid activation function provided (must be either '{0}')".format(
                    "', '".join(acts.keys())
                )
            )
        x = ks.layers.Input(shape=(board_size ** 2 + 1))
        input_layer = x
        # Ensure that input has correct shape
        input_layer = ks.layers.Dense(board_size ** 2 + 1, activation=activation_function)(input_layer)
        for i in nn_dims:
            input_layer = ks.layers.Dense(i, activation=activation_function)(input_layer)
        # Ensure that output has correct shape and activation softmax
        actor_output = ks.layers.Dense(board_size ** 2)(input_layer)
        actor_output = ks.layers.Activation(activation="softmax", name="actor_output")(actor_output)
        critic_output = ks.layers.Dense(1)(input_layer)
        critic_output = ks.layers.Activation(activation="tanh", name="critic_output")(critic_output)
        opt = optimizers.get(optimizer, None)
        if opt is None:
            raise ValueError(
                "Invalid optimizer provided (must be either '{0}')".format(
                    "', '".join(optimizers.keys())
                )
            )
        model = ks.Model(inputs=x, outputs=[actor_output, critic_output])
        losses = {
            "actor_output": "kl_divergence",
            "critic_output": "mse",
        }
        loss_weights = {"actor_output": 1.0, "critic_output": 1.0}
        model.compile(
            optimizer=opt(learning_rate=lr),
            loss=losses,
            loss_weights=loss_weights,
            metrics=["accuracy"]
        )
        model.summary()
        return model

    def fit(self, inputs, targets, batch_size=64, epochs=1, verbosity=0):
        if self.topp:
            raise Exception("Model should not train during TOPP")
        train_x, train_y, valid_x, valid_y = self.train_test_split(inputs, targets)
        train_x, train_y = self.random_minibatch(train_x, train_y, batch_size)
        self.model.fit(
            inputs, targets, epochs=epochs, verbose=verbosity, batch_size=batch_size
        )
        e = self.model.evaluate(valid_x, valid_y, verbose=verbosity)
        if len(e) == 5:
            print(format('Loss: %.2f\nActor loss: %.2f\nCritic loss: %.2f\nActor accuracy: %.2f\nCritic accuracy:%.2f' % (e[0], e[1], e[2], e[3], e[4])))

    def predict(self, inputs):
        predictions = self.model.predict(inputs)
        pred_length = len(predictions[0])
        illegal_moves_removed = np.array(
            [NeuralNet.normalize(np.array([
                predictions[0][n][i] if inputs[0][i+1] == 0 else 0
                for i in range(len(predictions[0][n]))
            ]))
            for n in range(pred_length)
            ]
        )
        return illegal_moves_removed, predictions[1]

    def best_action(self, normalized_predictions):
        i = np.argmax(normalized_predictions[0])
        return NeuralNet.convert_to_2d_move(i, self.board_size)

    def save_model(self, model_name, episode_number):
        self.model.save("project2/models/{0}{1}.h5".format(model_name, episode_number))
        print("Model {0}{1} saved succesfully".format(model_name, episode_number))

    def load_saved_model(self, episode_number):
        model = ks.models.load_model(
            "project2/models/{0}{1}.h5".format(f"{self.board_size}x{self.board_size}_ep", episode_number)
        )
        print("Model {0}{1} loaded succesfully".format(f"{self.board_size}x{self.board_size}_ep", episode_number))
        return model

    def random_minibatch(self, inputs, targets, size=10):
        if (size >= len(inputs)):
            return inputs, targets
        weight = np.linspace(0, 1, len(inputs))
        index = [i for i in range(len(inputs))]
        indices = np.array(random.choices(index, weights=weight, k=size))
        actor_targets = targets["actor_output"][indices]
        critic_targets = targets["critic_output"][indices]
        return inputs[indices], {"actor_output": actor_targets, "critic_output": critic_targets}

    def train_test_split(self, inputs, targets, split=0.1, randomize=True):
        vc = round(split * len(inputs))
        actor_targets = targets["actor_output"]
        critic_targets = targets["critic_output"]
        if split > 0:
            pairs = list(zip(inputs, actor_targets, critic_targets))
            if randomize:
                np.random.shuffle(pairs)
            vcases = pairs[0:vc]
            tcases = pairs[vc:]
            return (
                np.array([tc[0] for tc in tcases]),
                {"actor_output": np.array([tc[1] for tc in tcases]),
                 "critic_output": np.array([tc[2] for tc in tcases])},
                np.array([vc[0] for vc in vcases]),
                {"actor_output": np.array([vc[1] for vc in vcases]),
                 "critic_output": np.array([vc[2] for vc in vcases])},
            )
        else:
            return inputs, targets, [], {"actor_output":[], "critic_output":[]}

    @staticmethod
    def convert_to_2d_move(index, board_size):
        return (index//board_size, index % board_size)

    @staticmethod
    def normalize(arr):
        # Assumes input of 1d np-array
        arrsum = sum(arr)
        return arr/arrsum
