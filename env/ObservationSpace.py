from abc import ABC

class ObservationSpace(ABC):
    def __init__(self, shape, dtype, low, high):
        self.shape = shape
        self.dtype = dtype
        self.low = low
        self.high = high