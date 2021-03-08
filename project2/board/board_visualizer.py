import math
import matplotlib.pyplot as plt
import pygame


class BoardVisualizer:
    def __init__(self, width=1000, height=800):
        self.width = width
        self.height = height

    def draw_board(self, board):
        height = self.height
        width = self.width
        horisontal_spacing = find_horisontal_spacing(board, width)

        # Create blank screen to be drawn upon
        screen = pygame.display.set_mode((width, height))
        screen.fill((255, 255, 255))

        for row in range(len(board)):
            horisontal_position = (width / 2) - (
                (len(board[row])+1) * horisontal_spacing
            ) + row*horisontal_spacing/2
            circle_size = 125/len(board)
            for col in range(len(board[row])):
                if board[row][col] == 1:
                    pygame.draw.circle(
                        screen,
                        (0, 0, 255),
                        (horisontal_position, height * ((row + 1) / (len(board) + 1))),
                        circle_size,
                    )
                elif board[row][col] == 0:
                    pygame.draw.circle(
                        screen,
                        (0, 0, 0),
                        (horisontal_position, height * ((row + 1) / (len(board) + 1))),
                        circle_size,
                    )
                elif board[row][col] == 2:
                    pygame.draw.circle(
                        screen,
                        (255, 0, 0),
                        (horisontal_position, height * ((row + 1) / (len(board) + 1))),
                        circle_size,
                    )
                else:
                    raise Exception(
                        "Expected a value of 0, 1 or 2 in cell, instead recieved: "
                        + str(board[row][col])
                    )
                horisontal_position += horisontal_spacing * 2

        pygame.display.update()


def find_horisontal_spacing(board, width):
    """Using the maximum number of nodes to be placed horisontally and width
    of the screen, find a distance that can be used to seperate each node
    equally horisontally."""

    max_rows = 1
    for row in board:
        if len(row) > max_rows:
            max_rows = len(row)
    return (width - 100) / (
        max_rows * 2 + math.ceil(len(board)/2)
    )  # Using width -100, to make sure the circles aren't drawn outside of the screen.
