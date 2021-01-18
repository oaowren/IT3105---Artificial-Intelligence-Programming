import networkx as nx
import matplotlib.pyplot as plt
from board import Board

class BoardVisualizer:

    def __init__(self, display_episode, delay, open_cells=[(2,2)]):
        # Extend to create different boards
        self.display_episode = display_episode
        self.delay = delay
        self.board = Board(open_cells=self.open_cells)
        self.graph = nx.Graph()

    def draw_board(self):
        self.graph.add_nodes_from(self.open_cells, color = "black")
        nx.draw(self.graph)
        plt.show()

    def get_empty_positions(self):

