from visualizations.animation import plot_curve
from state_manager.mountain_car import MountainCar

mc = MountainCar()

if __name__ == "__main__":
    plot_curve(mc.pos)