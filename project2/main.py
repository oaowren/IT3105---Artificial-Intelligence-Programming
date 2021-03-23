from NeuralNetwork.neuralnet import NeuralNet
from parameters import Parameters
from board.board import Board
from board.board_visualizer import BoardVisualizer
from board.game_simulator import GameSimulator
from Client_side.BasicClientActor import BasicClientActor
from MCTS.montecarlo import MCTS
import numpy as np
import time

p = Parameters()
# Initialize save interval, RBUF, ANET and board (state manager)
save_interval = p.number_of_games // p.number_of_cached_anet
rbuf = {}
nn = NeuralNet(p.nn_dims, p.board_size, p.lr, p.activation_function, p.optimizer)
board = Board(p.board_size, p.starting_player)
board_visualizer = BoardVisualizer()
tree = MCTS((p.starting_player, board.get_state()), nn)
sim = GameSimulator(board, p.board_size, p.starting_player, tree)
topp = p.topp


def run_full_game(epsilon):
    board.reset_board()
    while not board.check_winning_state():
        tree.root = board.get_state()
        sim.initialize_root(tree.root, board.player)
        D = sim.sim_games(epsilon, p.number_of_search_episodes)
        rbuf[str(board.player) + " " + tree.root] = D
        next_move = get_best_move_from_D(D)
        board.make_move(next_move)
        sim.reset()
    nn.fit([[int(i) for i in r.split()] for r in rbuf.keys()],\
         [NeuralNet.normalize(np.array([i[1] for i in rbuf[key]])) for key in rbuf.keys()])

def get_best_move_from_D(D):
    best_move = None
    most_visits = -1
    for d in D:
        if (d[1] > most_visits):
            best_move = d[0]
            most_visits = d[1]
    return best_move

def run_topp_game(actor1, actor2, starting_player, visualize=True):
    player_no = starting_player
    player = actor1 if player_no == 1 else actor2
    if visualize:
        board_visualizer.draw_board(board.board)
        time.sleep(1)
    while not board.check_winning_state():
        split_state = np.concatenate(([player_no], [int(i) for i in board.get_state().split()]))
        preds = actor1.predict(np.array([split_state]))
        move = player.best_action(preds)
        board.make_move(move)
        player_no = player_no % 2 + 1
        player = player = actor1 if player_no == 1 else actor2
        if visualize:
            board_visualizer.draw_board(board.board)
            time.sleep(1)
    winning_player = 1 if board.check_winning_state_player_one() else 2
    print(f'Player {winning_player} wins!')
    if visualize:
        board_visualizer.draw_board(board.board)
        time.sleep(1)


if __name__ == "__main__":
    if (p.topp):
        actor1 = NeuralNet(board_size=p.board_size, load_saved_model=True, episode_number=p.actor1_episode)
        actor2 = NeuralNet(board_size=p.board_size, load_saved_model=True, episode_number=p.actor2_episode)
        run_topp_game(actor1, actor2, 1)
    else:
        epsilon = p.epsilon
        for game in range(p.number_of_games):
            run_full_game(epsilon)
            print("Game no. " + str(game+1))
            epsilon *= p.epsilon_decay
            if game % save_interval == 0:
                nn.save_model(f"{p.board_size}x{p.board_size}_ep", game)
