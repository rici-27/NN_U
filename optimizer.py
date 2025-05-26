from tensor import Tensor
from network import *
from layer import *
from abc import ABC
import numpy as np
import time

# Trainer Klasse um das Netzwerk zu trainieren

class SGDTrainer():

    def __init__(self, learningRate: float, amountEpochs: int, Shuffle: bool=True):
        self.learningRate = learningRate
        self.amountEpochs = amountEpochs
        self.shuffle = Shuffle

    def optimizing(self, network: Network, data):
        for epoch in range(self.amountEpochs):
            epoch_loss = 0
            start_time = time.time()

            for m in range(len(data[0])):
                network.forward(data[0][m])
                network.backprop(Tensor(data[1][m]))
                epoch_loss += network.loss.elements

                for n in range(len(network.layers)):
                    if type(network.layers[n]) == FCN_Layer:
                        network.layers[n].weight.elements -= self.learningRate * network.layers[n].weight.deltas
                        network.layers[n].bias.elements -= self.learningRate * network.layers[n].bias.deltas
                # hier weitere optionen/cases für CNN hinzufügen, oder Methode in layer hinzufügen

            end_time = time.time()
            print("Epoch:", epoch, ", Loss:", epoch_loss/len(data[0]))  
            print("Dauer der Epoche:", round(end_time - start_time, 2), "Sekunden")

                    