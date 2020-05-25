import random
from collections import deque

import numpy as np


class Buffer:

    def __init__(self):
        self.memory = deque(maxlen=2000)

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def get_random_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return [np.array([_[i] for _ in batch]) for i in range(5)]
