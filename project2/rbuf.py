from typing import Deque
import numpy as np
import random

class RBUF:

    def __init__(self, max_length=800):
        self.buffer = Deque()
        self.max_length = max_length

    def add(self, state_distribution):
        if len(self.buffer) > self.max_length:
            self.buffer.popleft()
        self.buffer.append(state_distribution)

    def get_random_batch(self, size):
        if size > len(self.buffer):
            return self.buffer
        weights = np.linspace(0, 1, len(self.buffer))
        return random.choices(self.buffer, weights=weights, k=size)