import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    return max(x, 0)

def tanH(x):
    return 1 - 2/(1 + np.exp(2 * x))

