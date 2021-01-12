from game.board import Board


if __name__ == "__main__":
    board = Board()
    board.print_board()
    board.make_move((2, 0), (2, 2))
    board.print_board()
    board.make_move((0, 0), (2, 0))
    board.print_board()
