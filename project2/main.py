from NeuralNetwork.neuralnet import NeuralNet
from parameters import Parameters

p = Parameters()

if __name__ == "__main__":
    nn = NeuralNet(p.nn_dims, p.board_size, p.lr, p.activation_function, p.optimizer)
