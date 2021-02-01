from game.board import Board
from game.board_visualizer import BoardVisualizer
from actor.actor import Actor
from critic.critic import Critic
from critic.nn_critic import CriticNN
import time
import copy


# ------ VARIABLES --------
# Board and Game Variables
board_type = "D"  # "T" or "D"
board_size = 3
open_cells = [(2, 2), (1, 1)]
number_of_episodes = 25
display_episode = number_of_episodes - 1  # Display final run
display_delay = 2  # Number of seconds between board updates in visualization

# Critic Variables
critic_method = "NN"  # "TL" or "NN"
critic_nn_dims = (15, 20, 30, 5, 1)
lr_critic = 0.1
eligibility_decay_critic = 0.1
discount_factor_critic = 0.1

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
    critic = (
        Critic(
            lr=lr_critic,
            eligibility_decay=eligibility_decay_critic,
            discount_factor=discount_factor_critic,
        )
        if critic_method == "TL"
        else (
            CriticNN(
                lr_critic,
                critic_nn_dims,
                eligibility_decay_critic,
                discount_factor_critic,
                len(board.board_state()),
            )
        )
    )
    critic.model.modify_gradients([1,2,4,5])

    # Draw initial board state
    boardVisualizer.draw_board(board.board, board.board_type)
    time.sleep(display_delay)  # Sleep to display the board for some time

    # Run episodes
    # TODO: Init V(s) for Critic
    actor.init_policy(board)
    # TODO: Run below code for each episode
    actor.reset_eligibility(board)
    # TODO: critic.reset_eligibility(board)
    action = actor.select_action(board)
    # Repeat for each step of the episode
    while True:
        prev_state, prev_action = board.board_state(), action
        board.make_move(action)
        # TODO: Give reinforcement for current state
        if board.check_losing_state() or board.check_winning_state():
            break
        action = actor.select_action(board)
        actor.update_eligibility(prev_state, prev_action, 1)
        td_error = 1  # TODO: Critic calculate TD-error
        # TODO: Critic set eligibility to 1
        for sap in find_saps(board):
            # TODO: Critic update V(s)
            # TODO: Critic update eligiblity
            actor.update_policy(sap[0], sap[1], td_error)
            actor.update_eligibility(sap[0], sap[1], 0)

        boardVisualizer.draw_board(board.board, board.board_type)
        time.sleep(display_delay)  # Sleep to display the board for some time

    boardVisualizer.draw_board(board.board, board.board_type)
    time.sleep(display_delay + 5)  # Sleep to display the board for some time
