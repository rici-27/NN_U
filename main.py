import numpy as np
from abc import ABC, abstractmethod
from functions import sigmoid

# Erstellen der Layer
a = np.array([1, 2, 3])
print(np.multiply([[0, 4], [1, 2]], [[10, 12], [3, 5]]))
print(np.multiply([[0, 4], [1, 2]], [[10, 12], [3, 5]]).T)


print(sigmoid(a))