import numpy as np

# Hilfsfunktionen für die Activation Layer, Softmax Layer und Convolutional Layer

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

# Input sind zwei Matrizen und die Ausgabe ist eine Matrix die durch die Faltung entsteht
def convolut(X, F):
    row = X.shape[0] - F.shape[0] + 1
    col =  X.shape[1] - F.shape[1] + 1
    Matrix = np.zeros((row, col))
    
    for i in range(row):
        for j in range(col):
            submatrix = X[i:(i + F.shape[0]), j:(j + F.shape[1])]
            Matrix[i, j] = np.sum(submatrix * F)
    return Matrix
    
