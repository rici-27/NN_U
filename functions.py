import numpy as np

# Hilfsfunktionen für die Activation Layer und Softmax

# sigmoid Funktion stückweise definiert - für numerische Stabilität - e^x macht Probleme für größe x

def sigmoid(x):
    return np.piecewise(
        x,
        [x > 0],
        [lambda i: 1 / (1 + np.exp(-i)),
            lambda i: np.exp(i) / (1 + np.exp(i))],
    )

def ReLu(x):
    return np.max(x, 0)

def tanH(x):
    return 1 - 2/(1 + np.exp(2 * x))

def softmax(x, w):
    return np.exp(x)/w


def convolut(x, k):
    """length = np.sqrt(x.lenght)
    matrix = x.reshape(length, length)"""



    pass