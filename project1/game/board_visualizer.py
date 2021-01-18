import networkx as nx
import matplotlib.pyplot as plt
from board import Board

board = Board()

G=nx.Graph()
for i in range(len(board.board)):
    for n in range(len(board.board[i])):
        elem = board.board[i][n]
        G.add_node(i*(len(board.board))+n+1)
        """if elem == 1:
            G.add_node((n+1)*(i+1))
        elif elem == 2:
            G.add_node(n)
        else: 
            G.add_node(n)"""

nx.draw(G)
plt.show()