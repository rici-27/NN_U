import numpy as np

class Tensor:
    def __init__(self, elements, dtype=np.float64):
        self.elements = elements
        self.deltas = np.zeros_like(self.elements)
