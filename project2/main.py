from NeuralNetwork.neuralnet import NeuralNet
from parameters import Parameters
from board.board import Board
from board.board_visualizer import BoardVisualizer
import numpy as np
import time
import random

p = Parameters()

if __name__ == "__main__":
    nn = NeuralNet(p.nn_dims, p.board_size, p.lr, p.activation_function, p.optimizer)
    board = Board(p.board_size)
    board_visualizer = BoardVisualizer()
    board.board[0][3] = 1
    board.board[2][0] = 2
    flat_board = board.flatten_board()
    rando_values = np.array([[random.randint(0, 50) for _ in range(p.board_size**2)] for _ in range(50)])
    rando_targets = np.array([[random.uniform(0, 0.2) for _ in range(p.board_size**2)] for _ in range(50)])
    nn.fit(rando_values, rando_targets, epochs=25, batch_size=10)
    rando_pred = np.array([[random.randint(0, 50) for _ in range(p.board_size**2)]])
    print(nn.predict(rando_pred, flat_board))
    # print(board.check_winning_state_player_one())
    # board.board[1][1] = 1
    # board.board[2][3] = 2
    # print(board.check_winning_state(1))
    # board.board[2][1] = 1
    # board.board[3][1] = 2
    # print(board.check_winning_state())
    # board.board[3][0] = 1
    # board.board[3][2] = 2
    # print(board.board)
    # print(board.check_winning_state())
    # board_visualizer.draw_board(board.board)
    # time.sleep(6)

