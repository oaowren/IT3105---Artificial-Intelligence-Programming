from game.board import Board
from game.board_visualizer import BoardVisualizer
from actor.actor import Actor
import time


# ------ VARIABLES --------
# Board and Game Variables
board_type = "D"                # "T" or "D"
board_size = 4
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

# Main-function for running everything
if __name__ == "__main__":
    board = Board(board_type=board_type, board_size=board_size, open_cells=open_cells)
    boardVisualizer = BoardVisualizer(width=1000, height=800)
    actor = Actor(board)
    boardVisualizer.draw_board(board.board, board.board_type)
    time.sleep(display_delay) #Sleep to display the board for some time
    while not board.check_losing_state() and not board.check_winning_state():
        board.print_board()
        selected_move = actor.select_action()
        board.make_move(selected_move)
        boardVisualizer.draw_board(board.board, board.board_type)
        time.sleep(display_delay) #Sleep to display the board for some time

