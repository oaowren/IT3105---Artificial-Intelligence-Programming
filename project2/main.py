from topp import TOPP
from NeuralNetwork.neuralnet import NeuralNet
from parameters import Parameters
from board.board import Board
from board.board_visualizer import BoardVisualizer
from board.game_simulator import GameSimulator
from Client_side.BasicClientActor import BasicClientActor
from MCTS.montecarlo import MCTS
from scipy.special import softmax
import numpy as np

p = Parameters()
# Initialize save interval, RBUF, ANET and board (state manager)
save_interval = p.number_of_games // p.number_of_cached_anet
rbuf = {}
nn = NeuralNet(p.nn_dims, p.board_size, p.lr, p.activation_function, p.optimizer)
board = Board(p.board_size, p.starting_player)
board_visualizer = BoardVisualizer()
tree = MCTS((p.starting_player, board.get_state()), nn)
sim = GameSimulator(board, p.board_size, p.starting_player, tree)
topp = TOPP()


def run_full_game(epsilon, sigma, starting_player):
    # Starting state
    board.reset_board(starting_player)
    while not board.check_winning_state():
        # Initialize simulations
        tree.root = board.get_state()
        sim.initialize_root(tree.root, board.player)
        # Return distribution
        D, Q = sim.sim_games(epsilon, sigma, p.number_of_search_episodes)
        # Parse to state representation
        s = str(board.player) + " " + tree.root
        # Select move based on D
        next_move = get_best_move_from_D(D)
        # Add to replay buffer
        rbuf[s] = (D, Q)
        board.make_move(next_move)
        sim.reset(board.player)
    tree.reset()
    # Reset memoization of visited states during rollouts
    inputs = np.array([[int(i) for i in r.split()] for r in rbuf.keys()])
    actor_target = np.array([softmax([i[1] for i in rbuf[key][0]]) for key in rbuf.keys()])
    critic_target = np.array([[rbuf[key][1]] for key in rbuf.keys()])
    targets = {"actor_output": actor_target,
               "critic_output": critic_target}
    nn.fit(inputs, targets, batch_size=p.batch_size)

def get_best_move_from_D(D):
    best_move = None
    most_visits = -1
    for d in D:
        if (d[1] > most_visits):
            best_move = d[0]
            most_visits = d[1]
    return best_move


if __name__ == "__main__":
    if (p.topp):
        episodes = [i*save_interval for i in range(p.number_of_cached_anet + 1)]
        actors = [NeuralNet(board_size=p.board_size, load_saved_model=True, episode_number=i) for i in episodes]
        topp.run_topp(board, episodes, actors, p.topp_games, board_visualizer)
    else:
        epsilon = p.epsilon
        sigma = p.sigma
        for game in range(p.number_of_games):
            if game % save_interval == 0:
                nn.save_model(f"{p.board_size}x{p.board_size}_ep", game)
            print("Game no. " + str(game+1))
            run_full_game(epsilon, sigma, game % 2 + 1 if p.starting_player==0 else p.starting_player)
            epsilon *= p.epsilon_decay
            sigma *= p.sigma_decay
        nn.save_model(f"{p.board_size}x{p.board_size}_ep", p.number_of_games)
