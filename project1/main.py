from game.board import Board
from game.board_visualizer import BoardVisualizer
from actor.actor import Actor
from critic.nn_critic import CriticNN
from critic.table_lookup_critic import TableLookupCritic
from parameters import Parameters
from matplotlib import pyplot as plt
import time


# ------ VARIABLES --------
p = Parameters()
scenario = p.scenario_size4_diamond_tl()
if scenario is not None:
    scenario()
# Board and Game Variables
board_type = p.board_type
board_size = p.board_size
open_cells = p.open_cells
number_of_episodes = p.number_of_episodes
# Rewards
winning_reward = p.winning_reward
losing_reward_per_peg = p.losing_reward_per_peg
discount_per_step = p.discount_per_step
# Visualization
display_episode = p.display_episode
display_delay = p.display_delay

# Critic Variables
critic_method = p.critic_method
critic_nn_dims = p.critic_nn_dims
lr_critic = p.lr_critic
eligibility_decay_critic = p.eligibility_decay_critic
discount_factor_critic = p.discount_factor_critic

# Actor Variables
lr_actor = p.lr_actor
eligibility_decay_actor = p.eligibility_decay_actor
discount_factor_actor = p.discount_factor_actor
epsilon = p.epsilon
epsilon_decay = p.epsilon_decay

# Visualization size
height = p.height
width = p.width

# -------------------------

# ------- FUNCTIONS -------
def create_critic(method, nn_dimensions, lr, eligibility_decay, discount_factor):
    if method == "TL":
        return TableLookupCritic(lr, eligibility_decay, discount_factor)
    return CriticNN(
        lr, nn_dimensions, eligibility_decay, discount_factor
    )


def run_game_instance(board, actor, critic, remaining_pegs, visualize=False):
    actor.eligibility = {}
    critic.eligibility = {}
    action = actor.select_action(board)
    state_and_rewards = []
    state_and_rewards.append((board.board_state(), board.get_reward()))
    state_and_action = []
    state_and_action.append((board.board_state(), action))
    if visualize:
        boardVisualizer.draw_board(board.board, board.board_type)
        time.sleep(display_delay)  # Sleep to display the board for some time
    while True:
        prev_state, prev_action = board.board_state(), action
        board.make_move(action)
        reward = board.get_reward()
        state_and_rewards.append((board.board_state(), reward))
        state_and_action.append((prev_state, prev_action))
        td_error = critic.calculate_td_error(prev_state, board.board_state(), reward)
        critic.update_expected_reward(state_and_rewards)
        actor.update(td_error, state_and_action)
        if visualize:
            boardVisualizer.draw_board(board.board, board.board_type)
            time.sleep(display_delay)  # Sleep to display the board for some time
        if board.check_losing_state() or board.check_winning_state():
            if critic_method == "NN":
                critic.update_model(state_and_rewards)
            remaining_pegs.append(board.get_remaining_pegs())
            if board.check_winning_state():
                board.reset_board()
                print("WIN")
                return 1
            board.reset_board()
            return 0
        action = actor.select_action(board)


# -------------------------

# Main-function for running everything
if __name__ == "__main__":

    # Initialize all components
    board = Board(
        rewards=[winning_reward, losing_reward_per_peg, discount_per_step],
        board_type=board_type,
        board_size=board_size,
        open_cells=open_cells,
    )
    boardVisualizer = BoardVisualizer(width, height)
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
        discount_factor=discount_factor_critic
    )
    remaining_pegs = []

    # Run episodes
    for i in range(number_of_episodes):
        print("Running training episode: {}".format(i + 1))
        run_game_instance(board, actor, critic, remaining_pegs, False)
        actor.eps *= epsilon_decay
    if display_episode:
        actor.eps = 0
        run_game_instance(board, actor, critic, remaining_pegs, True)
    plt.plot(remaining_pegs)
    plt.show()
