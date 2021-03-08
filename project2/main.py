from NeuralNetwork.neuralnet import NeuralNet
from parameters import Parameters
from board.board import Board
from board.board_visualizer import BoardVisualizer
import numpy as np
import random
import time

p = Parameters()
# Initialize save interval, RBUF, ANET and board (state manager)
save_interval = p.number_of_games // p.number_of_cached_anet
rbuf = []
nn = NeuralNet(p.nn_dims, p.board_size, p.lr, p.activation_function, p.optimizer)
board = Board(p.board_size)
board_visualizer = BoardVisualizer()


def run_search_game():
    pass
    # Use tree policy to search from root to a leaf, update MC-board with each move
    # argmax(nn.predict(MC-board.state)) to select rollout actions until a final state F, update MC-board with each move
    # MCTS backprop from F to root


def run_full_game():
    board.reset_board()
    # Initialize MCT to root in same state as board_starting_state
    while not board.check_winning_state:
        # Initialize Monte Carlo game board (copy of board) to same state as root
        for search_game in range(p.number_of_search_episodes):
            run_search_game()
        # Save distribution D of visit counts in MCT
        # rbuf.append(root, D)
        # select move a based on D
        # Perform a, moving board to new state
        # MCT: retain subtree rooted at new state, discard everything else
        # root = new state
    # nn.fit([r[0] for r in rbuf], [NeuralNet.normalize(r[1]) for r in rbuf])


if __name__ == "__main__":
    """
    for game in range(p.number_of_games):
        run_full_game()
        if game % save_interval == 0:
            nn.save_model("model", game)
    """

    board.make_move((0,1), 1)
    board.make_move((1,0),1)
    flat_board = board.flatten_board()
    rando_values = np.array(
        [[random.randint(0, 50) for _ in range(p.board_size ** 2)] for _ in range(50)]
    )
    rando_targets = np.array(
        [[random.uniform(0, 0.2) for _ in range(p.board_size ** 2)] for _ in range(50)]
    )
    for i in range(len(rando_values)):
        rbuf.append([rando_values[i], rando_targets[i]])
    nn.fit([r[0] for r in rbuf], [NeuralNet.normalize(r[1]) for r in rbuf])
    rando_pred = np.array([[random.randint(0, 50) for _ in range(p.board_size ** 2)]])
    preds = nn.predict(rando_pred, flat_board)
    print(preds)
    print(nn.best_action(preds))
    # print(board.check_winning_state_player_one())
    # board.make_move((1,1), 1)
    # board.make_move((2,3), 2)
    # board.make_move((2,1), 1)
    # board.make_move((3,1), 2)
    # board.make_move((3,0), 2)
    # board.make_move((3,2), 2)
    # print(board.board)
    # print(board.check_winning_state())
    # board_visualizer.draw_board(board.board)
    # time.sleep(6)
