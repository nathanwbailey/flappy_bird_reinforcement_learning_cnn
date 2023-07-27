from collections import deque
import random

class CacheRecall():
    def __init__(self, memory_size) -> None:
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
    
    def cache(self, data):
        self.memory.append(data)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)