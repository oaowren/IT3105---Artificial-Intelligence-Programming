from game.board import Board


if __name__ == "__main__":
    board = Board(board_type="T", board_size=5, open_cells=[(2,2)])
    board.print_board()
    board.make_move((4, 0), (2, 2))
    board.print_board()
    board.make_move((3, 0), (1, 2))
    board.print_board()
