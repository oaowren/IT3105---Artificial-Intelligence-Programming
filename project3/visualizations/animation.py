import matplotlib.pyplot as plt
import numpy as np
import math

x = np.linspace(-1.2, 0.6, 100)

def plot_curve(starting_pos):
    plt.figure()
    plt.plot(x, y_func(x))
    plt.plot(starting_pos, y_func(starting_pos), 'o')
    plt.show()


def y_func(x):
    return np.cos(3*(x + math.pi/2))

    