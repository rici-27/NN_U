import numpy as np

# Tensor klasse zum Abspeichern von Daten

class Tensor:
    def __init__(self, elements):
        self.elements = elements
        self.deltas = np.zeros_like(self.elements)
