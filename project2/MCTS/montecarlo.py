from NeuralNetwork.neuralnet import NeuralNet
import random
"""
Dette er hovedsaklig basert på algoritmen fra
http://www.cs.cornell.edu/courses/cs6700/2016sp/lectures/CS6700-UCT.pdf
side 5-7.
"""

import numpy as np

class MCTS:

    def __init__(self, root, nn):
        self.root = root
        self.states= {}
        self.state_action = {}
        self.c = 1
        self.nn = nn
        self.memoized_preds = {}

    def update(self, state, action, reward):
        self.states[state]["N"] +=1
        self.state_action[(state,action)]["N"] +=1
        self.states[state]["Q"] += (reward - self.get_Q_state(state))/(1 + self.get_N(state, action))
        self.state_action[(state,action)]["Q"] += (reward - self.get_Q(state, action))/(1 + self.get_N(state, action))
        return

    def get_N(self, state, action=None):
        if action:
            if (state,action) not in self.state_action:
                self.state_action[(state, action)] = {"N": 0, "Q": 0}
            return self.state_action[(state,action)]["N"]
        if state not in self.states:
            self.states[state] = {"N":0, "Q": 0}
        return self.states[state]["N"]

    def get_Q(self, state, action):
        return self.state_action[(state,action)]["Q"]

    def get_Q_state(self, state):
        return self.states[state]["Q"]


    def exploration_bonus(self, state, action):
        exploration_bonus =self.c*np.sqrt(np.log(self.get_N(state))/(self.get_N(state,action)))
        return exploration_bonus

    def get_distribution(self, board):
        moves = [NeuralNet.convert_to_2d_move(i, board.board_size) for i in range(board.board_size ** 2)]
        state = board.get_state()
        dist = [self.get_N(state, move) for move in moves]
        dist = NeuralNet.normalize(np.array(dist))
        output = [(moves[i], dist[i]) for i in range(len(moves))]
        return output, self.get_Q_state(state)

    def rollout_action(self, board, epsilon, player):
        if random.random() < epsilon:
            return self.random_action(board)
        state = board.get_state()
        if (player, state) in self.memoized_preds:
            return self.nn.best_action(self.memoized_preds[(player,state)])
        split_state = np.concatenate(([player], [int(i) for i in state.split()]))
        preds = self.nn.predict(np.array([split_state]))
        self.memoized_preds[(player, state)] = preds[0]
        return self.nn.best_action(preds[0])

    def critic_evaluate(self, board, player):
        state = board.get_state()
        split_state = np.concatenate(([player], [int(i) for i in state.split()]))
        preds = self.nn.predict(np.array([split_state]))
        return preds[1][0][0]

    def random_action(self, board):
        return random.choice(board.get_legal_moves())

    def expand_tree(self, board):
        state = board.get_state()
        legal_moves = board.get_legal_moves()
        self.states[state] = {"N":0, "Q": 0}
        for move in legal_moves:
            board_copy = board.clone()
            board_copy.make_move(move)
            self.state_action[(state, move)] = {"N": 0, "Q": 0}

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
        return traversal_sequence

    def reset(self):
        self.states={}
        self.state_action={}
        self.memoized_preds = {}

