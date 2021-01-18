from game.board import Board
from game.board_visualizer import BoardVisualizer
import time


if __name__ == "__main__":
    board = Board(board_type="D", board_size=7, open_cells=[(2,2), (1,1)])
    boardVisualizer = BoardVisualizer(width=1000, height=800)
    boardVisualizer.draw_board(board.board, board.board_type)
    time.sleep(5) #Sleep to display the board for some time
    board.print_board()
    board.make_move((4, 0), (2, 2))
    boardVisualizer.draw_board(board.board, board.board_type)
    time.sleep(5) #Sleep to display the board for some time
    board.print_board()
    board.make_move((3, 0), (1, 2))
    board.print_board()
