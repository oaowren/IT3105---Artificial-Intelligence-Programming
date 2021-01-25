from game.board import Board
from game.board_visualizer import BoardVisualizer
from actor.actor import Actor
from critic.critic import Critic
import time


# ------ VARIABLES --------
# Board and Game Variables
board_type = "D"                # "T" or "D"
board_size = 3
open_cells = [(2,2), (1,1)]
number_of_episodes = 25
display_episode = number_of_episodes - 1    # Display final run
display_delay = 2               # Number of seconds between board updates in visualization

# Critic Variables
critic_method = "TL"            # "TL" or "NN"
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

# Main-function for running everything
if __name__ == "__main__":

    # Initialize all components
    board = Board(board_type=board_type, board_size=board_size, open_cells=open_cells)
    boardVisualizer = BoardVisualizer(width=1000, height=800)
    actor = Actor(lr=lr_actor, eligibility_decay=eligibility_decay_actor, discount_factor=discount_factor_actor, initial_epsilon=epsilon, epsilon_decay_rate=epsilon_decay)
    actor.init_policy(board)
    critic = Critic(method=critic_method, nn_dimensions=critic_nn_dims, lr=lr_critic, eligibility_decay=eligibility_decay_critic, discount_factor=discount_factor_critic)
    
    # Draw initial board state
    boardVisualizer.draw_board(board.board, board.board_type)
    time.sleep(display_delay) #Sleep to display the board for some time

    # Run episodes
    while not board.check_losing_state() and not board.check_winning_state():
        board.print_board()
        selected_move = actor.select_action(board)
        board.make_move(selected_move)
        boardVisualizer.draw_board(board.board, board.board_type)
        time.sleep(display_delay) #Sleep to display the board for some time

