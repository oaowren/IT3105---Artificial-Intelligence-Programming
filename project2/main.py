from NeuralNetwork.neuralnet import NeuralNet
from parameters import Parameters
from board.board import Board
from board.board_visualizer import BoardVisualizer
from board.game_simulator import GameSimulator
from Client_side.BasicClientActor import BasicClientActor
from MCTS.montecarlo import MCTS
from scipy.special import softmax
import numpy as np
import time

p = Parameters()
# Initialize save interval, RBUF, ANET and board (state manager)
save_interval = p.number_of_games // p.number_of_cached_anet
rbuf = {}
state_rewards = {}
nn = NeuralNet(p.nn_dims, p.board_size, p.lr, p.activation_function, p.optimizer)
board = Board(p.board_size, p.starting_player)
board_visualizer = BoardVisualizer()
tree = MCTS((p.starting_player, board.get_state()), nn)
sim = GameSimulator(board, p.board_size, p.starting_player, tree)
topp = p.topp


def run_full_game(epsilon, sigma, starting_player):
    board.reset_board(starting_player)
    game_sequence = []
    while not board.check_winning_state():
        tree.root = board.get_state()
        sim.initialize_root(tree.root, board.player)
        D = sim.sim_games(epsilon, sigma, p.number_of_search_episodes)
        s = str(board.player) + " " + tree.root
        game_sequence.append(s)
        rbuf[s] = D
        next_move = get_best_move_from_D(D)
        board.make_move(next_move)
        sim.reset(board.player)
    tree.memoized_preds = {}
    reward = {1:board.get_reward(1), 2: board.get_reward(2)}
    game_rewards = [reward[int(i.split()[0])] for i in game_sequence]
    for s in game_sequence:
        state_rewards[s] = game_rewards[int(s.split()[0])]
    inputs = np.array([[int(i) for i in r.split()] for r in rbuf.keys()])
    targets = {"actor_output": np.array([softmax([i[1] for i in rbuf[key]]) for key in rbuf.keys()]),
               "critic_output": np.array([[state_rewards[key]] for key in rbuf.keys()])}
    nn.fit(inputs, targets)

def get_best_move_from_D(D):
    best_move = None
    most_visits = -1
    for d in D:
        if (d[1] > most_visits):
            best_move = d[0]
            most_visits = d[1]
    return best_move

def run_topp_game(actor1, actor2, starting_player, visualize=True):
    board.reset_board(starting_player)
    player_no = starting_player
    player = actor1 if player_no == 1 else actor2
    if visualize:
        board_visualizer.draw_board(board.board)
        time.sleep(1)
    while not board.check_winning_state():
        split_state = np.concatenate(([player_no], [int(i) for i in board.get_state().split()]))
        preds = player.predict(np.array([split_state]))[0]
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
    return winning_player


if __name__ == "__main__":
    if (p.topp):
        episodes = [i*save_interval for i in range(p.number_of_cached_anet + 1)]
        actors = [NeuralNet(board_size=p.board_size, load_saved_model=True, episode_number=i) for i in episodes]
        actorscore = [0 for _ in episodes]
        for i in range(len(actors)):
            for n in range(i+1, len(actors)):
                player1 = 0
                player2 = 0
                print(f"Actor[{episodes[i]} episodes] vs. actor[{episodes[n]} episodes]")
                for game in range(p.topp_games):
                    winner = run_topp_game(actors[i], actors[n], game % 2 + 1, visualize= False)
                    player1 += 1 if winner == 1 else 0
                    player2 += 1 if winner == 2 else 0
                print(f"Actor[{episodes[i]} episodes] won {player1} times.\nActor[{episodes[n]} episodes] won {player2} times.\n")
                actorscore[i]+=player1
                actorscore[n]+=player2
        print("---------FINAL SCORES-----------")
        for i in range(len(episodes)):
            print(f"Actor[{episodes[i]} episodes]: {actorscore[i]} wins")
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
