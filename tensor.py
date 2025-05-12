import numpy as np

class Tensor:
    def __init__(self, elements, dtype=np.float64):
        self.elements = elements
        self.deltas = np.zeros([np.shape(self.elements)[0], np.shape(self.elements)[0]])
        # wir sollen deltas lazy initiieren, was heißt das?

    # def __repr__(self):
    #     return f"Tensor(elements={self.elements})"