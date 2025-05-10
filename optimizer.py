from tensor import Tensor
from network import *
from abc import ABC
import numpy as np

class SGDTrainer():

    def __init__(self, learningRate: float, amountEpochs: int, Shuffle: bool=True):
        self.learningRate = learningRate
        self.amountEpochs = amountEpochs
        self.shuffle = Shuffle

    def optimizing(self, network: Network, data):
        for i in range(self.amountEpochs):
            for m in range(len(data)):
                network.forward(data[m])
                network.backprop(data[m])

                for n in range(len(network.layers)):
                    if type(network.layers[n]) == FCN_Layer:
                        network.layers[n].weights =- self.learningRate * network.layers[n].weights.deltas
                        network.layers[n].bias =- self.learningRate * network.layers[n].bias.deltas

                        