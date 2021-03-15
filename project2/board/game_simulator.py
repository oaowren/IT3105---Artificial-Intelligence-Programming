class GameSimulator:

    def __init__(self, board, starting_player):
        self.board = board.clone()
        self.player = starting_player

    def change_player(self):
        self.player = 1 if self.player == 2 else 2

    

    