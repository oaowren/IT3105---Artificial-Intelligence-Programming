
"""
Dette er hovedsaklig basert på algoritmen fra
http://www.cs.cornell.edu/courses/cs6700/2016sp/lectures/CS6700-UCT.pdf
side 5-7.
"""

import numpy as np

class MCTS:

    def __init__(self, root, nn):
        self.root = root
        self.tree_policy = {}
        self.states= {}
        self.state_action = {}
        self.c = 1
        self.nn = nn

    def update(self, state, action, reward):
        self.state[state]["N"] +=1
        self.state_action[(state,action)]["N"] +=1
        self.state_action[(state,action)]["Q"] += (reward - self.get_Q(state, action))/self.get_N(state, action)
        return

    def get_N(self, state, action=None):
        if action:
            if (state,action) not in self.state_action:
                self.state_action[(state,action)]["N"] = 0
            return self.state_action[(state,action)]["N"]
        if state not in self.state:
            self.state[state]["N"] = 0
        return self.state[state]["N"]

    def get_Q(self, state, action):
        return self.state_action[(state,action)]["Q"]

    def exploration_bonus(self, state, action):
        exploration_bonus =self.c*np.sqrt(np.log(self.get_N(state))/self.get_N(state,action))
        return exploration_bonus

    def get_distribution(self, board):
        moves = board.get_legal_moves()
        state = board.get_state()
        dist = []
        for move in moves:
            dist.append(self.get_N(state, move))
        return dist

    def rollout_action(self, state, epsilon, player):
        split_state = np.concatenate(([player], [int(i) for i in state.split()]))
        preds = self.nn.predict(np.array([split_state]))
        print(preds)
        return self.nn.epsilon_best_action(preds, epsilon)

    def expand_tree(self, board, player):
        state = board.get_state()
        legal_moves = board.get_legal_moves()
        self.states[state] = {"N":0, "A": legal_moves, "P": player}
        for move in legal_moves:
            board_copy = board.clone()
            board_copy.make_move(move)
            self.state_action[(state, move)] = {"N": 0, "Q": 0, "P": player, "State": board_copy}

    def select_action(self, board, player):
        #Get max value for player 1, and min value for player 2
        state = board.get_state()
        moves = board.get_legal_moves()
        if(player == 1):
            values = [self.get_max_value_move(state, move) for move in moves]
            index = values.index(max(values))
        else:
            values = [self.get_min_value_move(state, move) for move in moves]
            index = values.index(min(values))
        return moves[index]

    def get_max_value_move(self, board, move):
        return self.get_Q(board, move) + self.exploration_bonus(board, move)

    def get_min_value_move(self, board, move):
        return self.get_Q(board, move) - self.exploration_bonus(board, move)

    def traverse(self, board, player):
        traversal_sequence = []
        while not board.check_winning_state() and board.get_state() in self.states:
            move = self.select_action(board, player)
            traversal_sequence.append((player, board.get_state(), move))
            board.make_move(move)
        print("seq")
        print(traversal_sequence)
        return traversal_sequence

