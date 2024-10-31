from abc import ABC, abstractmethod

class ActionSpace(ABC):
    def __init__(self, shape, dtype, low, high):
        self.shape = shape
        self.dtype = dtype
        self.low = low
        self.high = high

    @abstractmethod
    def sample(self):
        pass
