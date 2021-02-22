from NeuralNetwork.neuralnet import NeuralNet
from parameters import Parameters
from board.board import Board
from board.board_visualizer import BoardVisualizer
import time

p = Parameters()

if __name__ == "__main__":
    nn = NeuralNet(p.nn_dims, p.board_size, p.lr, p.activation_function, p.optimizer)
    board = Board(4)
    board_visualizer = BoardVisualizer()
    board.board[0][3] = 1
    board.board[2][0] = 2
    print(board.check_winning_state_player_one())
    board.board[1][1] = 1
    board.board[2][3] = 2
    print(board.check_winning_state(1))
    board.board[2][1] = 1
    board.board[3][1] = 2
    print(board.check_winning_state())
    board.board[3][0] = 1
    board.board[3][2] = 2
    print(board.board)
    print(board.check_winning_state())
    board_visualizer.draw_board(board.board)
    time.sleep(6)

