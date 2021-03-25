import math
from .neuralnet import NeuralNet
import random

class RandomPlayer():

    def __init__(self, board_size):
        self.board_size = board_size

    def predict(self, board):
        return board[0][1:]

    def best_action(self, preds):
        possible_inds = [i for i in range(len(preds)) if preds[i] == 0]
        index = random.choice(possible_inds)
        move = NeuralNet.convert_to_2d_move(index, self.board_size)
        return move