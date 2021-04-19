import time
import numpy as np

class TOPP:

    def run_topp_game(self, board, actor1, actor2, starting_player, board_visualizer, visualize=True):
        board.reset_board(starting_player)
        player_no = starting_player
        player = actor1 if player_no == 1 else actor2
        number_of_moves = 0
        if visualize:
            board_visualizer.draw_board(board.board)
            time.sleep(0.5)
        while not board.check_winning_state():
            split_state = np.concatenate(([player_no], [int(i) for i in board.get_state().split()]))
            preds = player.predict(np.array([split_state]))[0]
            move = player.best_action(preds)
            board.make_move(move)
            number_of_moves += int(player_no == 1)
            player_no = player_no % 2 + 1
            player = player = actor1 if player_no == 1 else actor2
            if visualize:
                board_visualizer.draw_board(board.board)
                time.sleep(0.5)
        winning_player = 1 if board.check_winning_state_player_one() else 2
        print(f'Player {winning_player} wins!')
        print(number_of_moves)
        if visualize:
            board_visualizer.draw_board(board.board)
            time.sleep(0.5)
        return winning_player

    def run_topp(self, board, episodes, actors, topp_games, visualizer, visualize_last_game=True):
        actorscore = [0 for _ in episodes]
        for i in range(len(actors)):
            for n in range(i+1, len(actors)):
                player1 = 0
                player2 = 0
                print(f"Actor[{episodes[i]} episodes] vs. actor[{episodes[n]} episodes]")
                for game in range(topp_games):
                    winner = self.run_topp_game(board, actors[i], actors[n], game % 2 + 1, visualizer, visualize= visualize_last_game and game==topp_games - 1)
                    player1 += 1 if winner == 1 else 0
                    player2 += 1 if winner == 2 else 0
                print(f"Actor[{episodes[i]} episodes] won {player1} times.\nActor[{episodes[n]} episodes] won {player2} times.\n")
                actorscore[i]+=player1
        for i in range(len(actors)-1, -1, -1):
            for n in range(i-1, -1, -1):
                player1 = 0
                player2 = 0
                print(f"Actor[{episodes[i]} episodes] vs. actor[{episodes[n]} episodes]")
                for game in range(topp_games):
                    winner = self.run_topp_game(board, actors[i], actors[n], game % 2 + 1, visualizer, visualize= visualize_last_game and game==topp_games - 1)
                    player1 += 1 if winner == 1 else 0
                    player2 += 1 if winner == 2 else 0
                print(f"Actor[{episodes[i]} episodes] won {player1} times.\nActor[{episodes[n]} episodes] won {player2} times.\n")
                actorscore[i]+=player1
        print("--------- FINAL SCORES AS PLAYER 1 --------")
        for i in range(len(episodes)):
            print(f"Actor[{episodes[i]} episodes]: {actorscore[i]} wins")