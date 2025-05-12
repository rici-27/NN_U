from tensor import Tensor
from network import *
from layer import *
from abc import ABC
import numpy as np
import time

class SGDTrainer():

    def __init__(self, learningRate: float, amountEpochs: int, Shuffle: bool=True):
        self.learningRate = learningRate
        self.amountEpochs = amountEpochs
        self.shuffle = Shuffle

    def optimizing(self, network: Network, data):
        for epoch in range(self.amountEpochs):
            epoch_loss = 0
            start_time = time.time()
            
            # data enthält example und label
            for m in range(len(data[0])):
                network.forward(data[0][m])
                # das wahre label wird auch als tensor übergeben an die backprop
                network.backprop(Tensor(data[0][m]))
                epoch_loss =+ network.loss.elements[0]

                for n in range(len(network.layers)):
                    if type(network.layers[n]) == FCN_Layer:
                        network.layers[n].weight =- self.learningRate * network.layers[n].weight.deltas
                        network.layers[n].bias =- self.learningRate * network.layers[n].bias.deltas

            end_time = time.time()
            print("Epoch:", epoch, ", Loss:", epoch_loss/len(data[0]))  
                    