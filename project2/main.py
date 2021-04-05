from copy import copy
import math
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
rbuf = []
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
    while not board.check_winning_state(board.board):
        # Return distribution
        no_of_legal_moves = len(board.get_legal_moves(board.board))
        dynamic_range = int(p.number_of_search_episodes/(math.log(no_of_legal_moves+2, p.board_size)))
        for i in range(p.number_of_search_episodes):
            leaf = tree.traverse()
            node = tree.expand_tree(leaf)
            reward = tree.rollout_game(node)
            tree.update(node, reward)
        D = tree.get_distribution()
        # Add to replay buffer
        rbuf.append((tree.root, D, tree.root.Q))
        # Select move based on D
        next_node = tree.get_best_move()
        board.make_move(next_node.action, player)
        player = player % 2 + 1
        tree.set_root(next_node)
    inputs = np.array([np.concatenate(([r[0].player], board.flatten_board(r[0].state))) for r in rbuf])
    actor_target = np.array([r[1] for r in rbuf])
    critic_target = np.array([r[2] for r in rbuf])
    targets = {"actor_output": actor_target,
               "critic_output": critic_target}
    nn.fit(inputs, targets, batch_size=p.batch_size)


if __name__ == "__main__":
    if (p.topp):
        episodes = [i*save_interval for i in range(p.number_of_cached_anet + 1)]
        actors = [NeuralNet(board_size=p.board_size, load_saved_model=True, episode_number=i) for i in episodes]
        topp.run_topp(board, episodes, actors, p.topp_games, board_visualizer)
    else:
        for game in range(p.number_of_games):
            if game % save_interval == 0:
                nn.save_model(f"{p.board_size}x{p.board_size}_ep", game)
            print("Game no. " + str(game+1))
            run_full_game(game % 2 + 1 if p.starting_player==0 else p.starting_player)
            nn.epsilon *= p.epsilon_decay
            nn.sigma *= p.sigma_decay
        nn.save_model(f"{p.board_size}x{p.board_size}_ep", p.number_of_games)
