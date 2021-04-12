from utils import Utils
from board.board import Board

import random
import numpy as np
"""
Dette er hovedsaklig basert på algoritmen fra
http://www.cs.cornell.edu/courses/cs6700/2016sp/lectures/CS6700-UCT.pdf
side 5-7.
"""


class MCTS:

    def __init__(self, actor, board_size, starting_player, c=1):
        self.board = Board(board_size, starting_player)
        self.root = None
        self.c = c
        self.actor = actor

    def set_root(self, node):
        self.root = node

    def update(self, goal_node, reward):
        node = goal_node
        node.N += 1
        while node != self.root:
            node.parent.N += 1
            node.Q += (reward - node.Q) / node.N
            node = node.parent

    def get_distribution(self):
        size = self.board.board_size
        dist = np.zeros(size**2)
        children = self.root.get_children()
        for child in children:
            index = child.action[0] * size + child.action[1]
            dist[index] += child.N
        return Utils.normalize(dist)

    def rollout_game(self, node):
        current_state, player = node.state, node.player
        reward = 0
        if random.random() > self.actor.sigma:
            reward = self.actor.get_critic_eval(Utils.flatten_board(current_state), player)
        else:
            while not self.board.check_winning_state(current_state):
                action = None
                if random.random() < self.actor.epsilon:
                    action = self.random_action(current_state)
                else:
                    preds = self.actor.get_actor_eval(Utils.flatten_board(current_state), player)
                    action = self.actor.best_action(preds)
                current_state = self.board.get_next_state(current_state, action, player)
                player = player % 2 + 1
            reward = self.board.get_reward(current_state)
        return reward

    def random_action(self, state):
        return random.choice(self.board.get_legal_moves(state))

    def expand_tree(self, node):
        if self.board.check_winning_state(node.state):
            return node
        node.children = self.board.get_child_states(node.player, node.state)
        for child in node.children:
            child.set_parent(node)
        return node

    def select_action(self, root):
        #Get max value for player 1, and min value for player 2
        moves = [(node, self.get_value_move(node, root.player)) for node in root.children]

        if(root.player == 1):
            root, _ = max(moves, key=lambda x: x[1])
        else:
            root, _ = min(moves, key=lambda x: x[1])
        return root

    def get_value_move(self, node, player):
        c = self.c if player == 1 else -self.c
        val = node.Q + c * np.sqrt(np.log(node.parent.N)/(1 + node.N))
        return val

    def traverse(self):
        root = self.root
        children = root.get_children()
        while len(children) != 0 and not self.board.check_winning_state(root.state):
            root = self.select_action(root)
            children = root.get_children()
        return root

    def get_best_move(self, distribution):
        index = np.argmax(distribution)
        return Utils.convert_to_2d_move(index, self.board.board_size)

    def get_node_from_move(self, node, move):
        for child in node.children:
            if child.action == move:
                return child


