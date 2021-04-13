from typing import Deque
import numpy as np
import random

class RBUF:

    def __init__(self, max_size=256):
        self.buffer = Deque()
        self.max_size = max_size

    def get_random_batch(self, batch_size):
        if batch_size > len(self.buffer):
            return self.buffer

        weights = np.linspace(0, 1, len(self.buffer))
        return random.choices(self.buffer, weights=weights, k=batch_size)

    def add(self, case):
        if len(self.buffer) > self.max_size:
            self.buffer.popleft()
        self.buffer.append(case)