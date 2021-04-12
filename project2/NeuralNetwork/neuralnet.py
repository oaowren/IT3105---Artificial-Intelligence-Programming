from utils import Utils
from tensorflow import keras as ks
import tensorflow as tf
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
        epsilon = 0,
        sigma = 0,
        nn_dims = (10),
        board_size = 3,
        lr = 0.01,
        activation = "sigmoid",
        optimizer  = "adam",
        load_saved_model=False,
        episode_number=0,
    ):
        self.epsilon = epsilon
        self.sigma = sigma
        self.board_size = board_size
        temp_opt = optimizers.get(optimizer, None)
        self.opt = temp_opt(learning_rate=lr)
        self.activation = acts.get(activation, None)
        self.actor_loss = ks.losses.CategoricalCrossentropy()
        self.critic_loss = ks.losses.MeanSquaredError()
        if load_saved_model:
            try:
                self.model = self.load_saved_model(episode_number)
            except OSError:
                raise ValueError(
                    "Failed to load model named {0}{1}, did you provide episode number?".format(f"{self.board_size}x{self.board_size}_ep", episode_number)
                )
        else:
            self.model = self.init_model(nn_dims, board_size)
        self.topp = load_saved_model

    def init_model(self, nn_dims, board_size):
        if self.activation is None:
            raise ValueError(
                "Invalid activation function provided (must be either '{0}')".format(
                    "', '".join(acts.keys())
                )
            )
        x = ks.layers.Input(shape=(board_size ** 2 + 1))
        input_layer = x
        # Ensure that input has correct shape
        input_layer = ks.layers.Dense(board_size ** 2 + 1, activation=self.activation)(input_layer)
        for i in nn_dims:
            input_layer = ks.layers.Dense(i, activation=self.activation)(input_layer)
        # Ensure that output has correct shape and activation softmax
        actor_output = ks.layers.Dense(board_size ** 2)(input_layer)
        actor_output = ks.layers.Activation(activation="softmax", name="actor_output")(actor_output)
        critic_output = ks.layers.Dense(1)(input_layer)
        critic_output = ks.layers.Activation(activation="tanh", name="critic_output")(critic_output)
        if self.opt is None:
            raise ValueError(
                "Invalid optimizer provided (must be either '{0}')".format(
                    "', '".join(optimizers.keys())
                )
            )
        model = ks.Model(inputs=x, outputs=[actor_output, critic_output])
        losses = {
            "actor_output": self.actor_loss,
            "critic_output": self.critic_loss,
        }
        loss_weights = {"actor_output": 1.0, "critic_output": 1.0}
        model.compile(
            optimizer=self.opt,
            loss=losses,
            loss_weights=loss_weights
        )
        model.summary()
        return model

    def fit(self, batch):
        if self.topp:
            raise Exception("Model should not train during TOPP")
        inputs = np.array([np.concatenate(([node.player], Utils.flatten_board(node.state))) for node, _, _ in batch])
        actor_target = np.array([r[1] for r in batch])
        critic_target = np.array([r[2] for r in batch])
        targets = {"actor_output":actor_target, "critic_output":critic_target}
        self.model.fit(inputs, targets, verbose=1, batch_size=64)

    def predict(self, inputs):
        predictions = self.model(inputs)
        pred_length = len(predictions[0])
        illegal_moves_removed = np.array(
            [Utils.normalize(np.array([
                predictions[0][n][i] if inputs[0][i+1] == 0 else 0
                for i in range(len(predictions[0][n]))
            ]))
            for n in range(pred_length)
            ]
        )
        return illegal_moves_removed, predictions[1]

    def get_critic_eval(self, flat_state, player):
        split_state = np.concatenate(([player], flat_state))
        _, value = self.predict(np.array([split_state]))
        return value[0][0]

    def get_actor_eval(self, flat_state, player):
        split_state = np.concatenate(([player], flat_state))
        preds, _ = self.predict(np.array([split_state]))
        return preds

    def best_action(self, normalized_predictions):
        i = np.argmax(normalized_predictions[0])
        return Utils.convert_to_2d_move(i, self.board_size)

    def save_model(self, model_name, episode_number):
        self.model.save("project2/models/{0}{1}.keras".format(model_name, episode_number))
        print("Model {0}{1} saved succesfully".format(model_name, episode_number))

    def load_saved_model(self, episode_number):
        model = ks.models.load_model(
            "project2/models/{0}{1}.keras".format(f"{self.board_size}x{self.board_size}_ep", episode_number), compile=False
        )
        print("Model {0}{1} loaded succesfully".format(f"{self.board_size}x{self.board_size}_ep", episode_number))
        return model

def safelog(tensor,base=0.0001):
    return tf.math.log(tf.math.maximum(tensor,base))

def deepnet_cross_entropy(targets,outs):
    return tf.reduce_mean(tf.reduce_sum(-1 * targets * tf.nn.log_softmax(outs), axis = -1))