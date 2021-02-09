from game.board import Board
from game.board_visualizer import BoardVisualizer
from actor.actor import Actor
from critic.critic import Critic
from critic.nn_critic import CriticNN
from critic.table_lookup_critic import TableLookupCritic
import numpy as np
import time
import copy


# ------ VARIABLES --------
# Board and Game Variables
board_type = "T"  # "T" or "D"
board_size = 3
open_cells = [(2, 0),(2,2)]
number_of_episodes = 25
display_episode = number_of_episodes - 1  # Display final run
display_delay = 2  # Number of seconds between board updates in visualization

# Critic Variables
critic_method = "NN"  # "TL" or "NN"
critic_nn_dims = (9, 20, 30, 5, 1)
lr_critic = 0.1
eligibility_decay_critic = 0.1
discount_factor_critic = 0.1
table_lookup = False

# Actor Variables
lr_actor = 0.1
eligibility_decay_actor = 0.1
discount_factor_actor = 0.1
epsilon = 0.1
epsilon_decay = 0.1
# -------------------------

# ------- FUNCTIONS -------
def find_saps(board):
    saps = []
    if board.check_losing_state() or board.check_winning_state():
        return saps
    else:
        moves = board.get_all_legal_moves()
        for move in moves:
            board_copy = copy.deepcopy(board)
            board_copy.make_move(move)
            saps.append((board.board_state(), move))
            saps = saps + find_saps(board_copy)
        return saps

def create_critic(method, nn_dimensions, lr, eligibility_decay, discount_factor, board, table_lookup):
    if table_lookup:
        return TableLookupCritic(board, lr, eligibility_decay, discount_factor)
    return Critic(method,nn_dimensions, lr, eligibility_decay, discount_factor)

def run_game_instance(board, actor, critic, visualize=False):
    actor.init_policy(board)
    # TODO: Run below code for each episode
    actor.reset_eligibility(board)
    # TODO: critic.reset_eligibility(board)
    action = actor.select_action(board)
    # Repeat for each step of the episode
    state_and_rewards = []
    state_and_rewards.append((board.board_state(), 0))
    state_and_action = []
    state_and_action.append((board.board_state(), action))
    while True:
        prev_state, prev_action = board.board_state(), action
        board.make_move(action)
        reward = board.get_reward()
        state_and_rewards.append((board.board_state(), reward))
        state_and_action.append((board.board_state(), action))
        # TODO: Give reinforcement for current state
        if board.check_losing_state() or board.check_winning_state():
            break
        action = actor.select_action(board)
        actor.update_eligibility(prev_state, prev_action, 1)
        # TODO: Critic set eligibility to 1
        if True: #if critic and actor should update TODO fix this
            td_error = critic.calculate_td_error(prev_state, board.board_state(), reward)
            critic.update_expected_reward(state_and_rewards)
            actor.update(td_error, state_and_action)

        if visualize:
            boardVisualizer.draw_board(board.board, board.board_type)
            time.sleep(display_delay)  # Sleep to display the board for some time
    board.print_board()
    board.reset_board()
    board.print_board()



# -------------------------

# Main-function for running everything
if __name__ == "__main__":

    # Initialize all components
    board = Board(board_type=board_type, board_size=board_size, open_cells=open_cells)
    boardVisualizer = BoardVisualizer(width=1000, height=800)
    actor = Actor(
        lr=lr_actor,
        eligibility_decay=eligibility_decay_actor,
        discount_factor=discount_factor_actor,
        initial_epsilon=epsilon,
        epsilon_decay_rate=epsilon_decay,
    )
    critic = create_critic(
        method=critic_method,
        nn_dimensions=critic_nn_dims,
        lr=lr_critic,
        eligibility_decay=eligibility_decay_critic,
        discount_factor=discount_factor_critic,
        board=board,
        table_lookup = True
    )

    # Draw initial board state
    boardVisualizer.draw_board(board.board, board.board_type)
    time.sleep(display_delay)  # Sleep to display the board for some time
    sequence = {}

    # Run episodes
    # TODO: Init V(s) for Critic
    for i in range(number_of_episodes):
        run_game_instance(board,actor,critic)
