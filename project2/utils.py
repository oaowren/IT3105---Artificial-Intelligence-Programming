import random

class Utils:

    @staticmethod
    def get_mid_index(board_size):
        if board_size % 2 == 1:
            mid = board_size//2
            return board_size * mid + mid
        mid1 = board_size//2
        mid2 = board_size//2 - 1
        return random.choice([board_size*mid1 + mid1, board_size*mid1 + mid2, board_size*mid2 + mid1, board_size*mid2 + mid2])

    @staticmethod
    def convert_to_2d_move(index, board_size):
        return (index//board_size, index % board_size)

    @staticmethod
    def normalize(arr):
        # Assumes input of 1d np-array
        arrsum = sum(arr)
        result = arr/arrsum
        return result

    @staticmethod
    def flatten_board(board):
        # Assumes input of 2d np-array
        return board.flatten()