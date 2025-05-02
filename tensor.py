import numpy as np

class Tensor:
    def __init__(self, elements: np.ndarray, deltas: np.ndarray):
    ## prüfen ob man so np.array vorgeben kann
        self.elements = elements
        self.deltas = deltas
        # wir sollen deltas lazy initiieren, was heißt das?