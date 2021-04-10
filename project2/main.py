from topp import TOPP
from NeuralNetwork.neuralnet import NeuralNet
from parameters import Parameters
from board.board import Board
from board.board_visualizer import BoardVisualizer
from board.game_simulator import GameSimulator
from Client_side.BasicClientActor import BasicClientActor
from MCTS.montecarlo import MCTS
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

# TODO: Flytt logikk ut fra game_simulator til mcts, se om roten har noe nytte
# Board kan gjøres statisk og ta inn [[], []] - board i stedet



def run_full_game(epsilon, sigma, starting_player):
    # Starting state
    board.reset_board(starting_player)
    while not board.check_winning_state():
        # Initialize simulations
        tree.root = board.get_state()
        sim.initialize_root(tree.root, board.player)
        # Return distribution
        D, Q = sim.sim_games(epsilon, sigma, p.number_of_search_episodes)
        D = check_for_winning_move(board, D)
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
    actor_target = np.array([[i[1] for i in rbuf[key][0]] for key in rbuf.keys()])
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

def check_for_winning_move(board, D):
    if (sum(board.flatten_board()) == 0):
        D = [(el[0], 1.0 if el[0] == (board.board_size//2, board.board_size//2) else 0.0) for el in D]
        return D
    for i, el in enumerate(D):
        # Set limit at 0.5 as this means this move will be taken based on D no matter what, increase the weight for rbuf
        if el[1] > 0.5:
            # Check for winning move
            board_copy = board.clone()
            board_copy.make_move(el[0])
            if board_copy.check_winning_state():
                D = [(el[0], 1.0 if ind == i else 0.0) for ind, el in enumerate(D)]
                return D
            # Check for opponent winning move
            board_copy = board.clone()
            board_copy.make_move(el[0], board_copy.player % 2 + 1)
            if board_copy.check_winning_state():
                D = [(el[0], 1.0 if ind == i else 0.0) for ind, el in enumerate(D)]
                return D
    return D


if __name__ == "__main__":
    if (p.topp):
        episodes = [i*save_interval for i in range(p.number_of_cached_anet + 1)]
        actors = [NeuralNet(board_size=p.board_size, load_saved_model=True, episode_number=i) for i in episodes]
        topp.run_topp(board, episodes, actors, p.topp_games, board_visualizer, visualize_last_game=p.visualize_last_game)
    elif p.oht:
        bsa = BasicClientActor(p.oht_episode, verbose=False)
        bsa.connect_to_server()
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
