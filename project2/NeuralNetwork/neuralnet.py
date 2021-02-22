from tensorflow import keras as ks

class NeuralNet():

    def __init__(self, nn_dims, board_size, lr):
        self.model = self.init_model(nn_dims, board_size, lr)

    def init_model(self, nn_dims, board_size, lr):
        model = ks.Sequential()
        model.add(ks.Input(shape=(board_size**2, )))
        model.add(ks.layers.Dense(board_size**2))
        for i in nn_dims:
            model.add(ks.layers.Dense(i))
        model.add(ks.layers.Dense(board_size**2))
        model.add(ks.layers.Activation("softmax"))
        model.compile(optimizer=ks.optimizers.Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=["accuracy"])
        model.summary()
        return model