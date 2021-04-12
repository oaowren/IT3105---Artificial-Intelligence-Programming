from copy import copy
import math
from rbuf import RBUF
from utils import Utils
from MCTS.node import Node
from topp import TOPP
from NeuralNetwork.neuralnet import NeuralNet
from parameters import Parameters
from board.board import Board
from board.board_visualizer import BoardVisualizer
from Client_side.BasicClientActor import BasicClientActor
from MCTS.montecarlo import MCTS
import numpy as np

p = Parameters()
# Initialize save interval, RBUF, ANET and board (state manager)
save_interval = p.number_of_games // p.number_of_cached_anet
rbuf = RBUF()
nn = NeuralNet(p.epsilon, p.sigma, p.nn_dims, p.board_size, p.lr, p.activation_function, p.optimizer)
board = Board(p.board_size, p.starting_player)
board_visualizer = BoardVisualizer()
topp = TOPP()


def run_full_game(starting_player):
    # Starting state
    board.reset_board(starting_player)
    tree = MCTS(nn, p.board_size, starting_player)
    tree.set_root(Node(copy(board.board), None, starting_player))
    player = starting_player
    next_node = tree.root
    while not board.check_winning_state(board.board):
        no_of_legal_moves = len(board.get_legal_moves(board.board))
        dynamic_range = int(p.number_of_search_episodes/(math.log(no_of_legal_moves+2, p.board_size)))
        for i in range(dynamic_range):
            leaf = tree.traverse()
            node = tree.expand_tree(leaf)
            reward = tree.rollout_game(node)
            tree.update(node, reward)
        D = tree.get_distribution()
        D = check_for_winning_move(board, D, player)
        # Add to replay buffer
        rbuf.add((tree.root, D, tree.root.Q))
        # Select move based on D
        next_move = tree.get_best_move(D)
        next_node = tree.get_node_from_move(next_node, next_move)
        board.make_move(next_move, player)
        player = player % 2 + 1
        tree.set_root(next_node)
    nn.fit(rbuf.get_random_batch(p.batch_size))

def check_for_winning_move(board, D, player):
    if (sum(Utils.flatten_board(board.board)) == 0):
        index = Utils.get_mid_index(board.board_size)
        D = [1.0 if ind == index else 0.0 for ind in range(len(D))]
        return np.array(D)
    for i, p in enumerate(D):
        if p > 0.5:
            move = Utils.convert_to_2d_move(i, board.board_size)
            board_copy = board.clone()
            board_copy.board[move[0]][move[1]] = player
            if board_copy.check_winning_state(board_copy.board):
                D = [1.0 if ind == i else 0.0 for ind in range(len(D))]
                return np.array(D)
            # board_copy = board.clone()
            # board_copy.board[move[0]][move[1]] = player % 2 + 1
            # if board_copy.check_winning_state(board_copy.board):
            #     D = [1.0 if ind == i else 0.0 for ind in range(len(D))]
            #     return np.array(D)
    return D


if __name__ == "__main__":
    if (p.topp):
        episodes = [i*save_interval for i in range(p.number_of_cached_anet, -1, -1)]
        actors = [NeuralNet(board_size=p.board_size, load_saved_model=True, episode_number=i) for i in episodes]
        topp.run_topp(board, episodes, actors, p.topp_games, board_visualizer)
    elif p.oht:
        bsa = BasicClientActor(verbose=False)
        bsa.connect_to_server()
    else:
        for game in range(p.number_of_games):
            if game % save_interval == 0:
                nn.save_model(f"{p.board_size}x{p.board_size}_ep", game)
            print("Game no. " + str(game+1))
            run_full_game(game % 2 + 1 if p.starting_player==0 else p.starting_player)
            nn.epsilon *= p.epsilon_decay
            nn.sigma *= p.sigma_decay
        nn.save_model(f"{p.board_size}x{p.board_size}_ep", p.number_of_games)
