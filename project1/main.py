from game.board import Board
from critic.critic import Critic


if __name__ == "__main__":
    board = Board(board_type="T", board_size=3, open_cells=[(0,0), (1,1), (2,0)])
    critic = Critic(board)
    board.print_board()
    print(critic.generate_game_states())
