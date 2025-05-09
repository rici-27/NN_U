import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def ReLu(x):
    return np.max(x, 0)

def tanH(x):
    return 1 - 2/(1 + np.exp(2 * x))

def softmax(x, w):
    return np.exp(x)/w