import numpy as np
from abc import ABC, abstractmethod
from functions import sigmoid
import tensorflow as tf

# Erstellen der Layer
a = np.array([1, 2, 1])
b = np.array([2,0,2])
#print(np.multiply([[0, 4], [1, 2]], [[10, 12], [3, 5]]))
#print(np.multiply([[0, 4], [1, 2]], [[10, 12], [3, 5]]).T)

#print(np.matmul(a.T, b))

#print((1+b)/2)

print(np.diag(2 * np.exp(a)))

tensors = []
for i in range(7):
    f"tensor_{i}" = i
    tensors.append(f"tensor_{i}")

print(tensors)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()