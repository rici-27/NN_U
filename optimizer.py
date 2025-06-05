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
        gesamt_zeit = 0
        for epoch in range(self.amountEpochs):
            epoch_loss = 0
            start_time = time.time()

            for m in range(len(data[0])):
                if m % 100 == 0:
                    print(f"Iteration {m}")
                network.forward(data[0][m])
                network.backprop(Tensor(data[1][m]))
                epoch_loss += network.loss.elements

                for n in range(len(network.layers)):
                    if type(network.layers[n]) == FCN_Layer:
                        network.layers[n].weight.elements -= self.learningRate * network.layers[n].weight.deltas
                        network.layers[n].bias.elements -= self.learningRate * network.layers[n].bias.deltas

                    elif type(network.layers[n]) == Conv2DLayer:
                        network.layers[n].weight.elements -= self.learningRate * network.layers[n].weight.deltas
                        network.layers[n].bias.elements -= self.learningRate * network.layers[n].bias.deltas

            end_time = time.time()
            print("Epoch:", epoch, ", Loss:", epoch_loss/len(data[0]))  
            print("Dauer der Epoche:", round(end_time - start_time, 2), "Sekunden")
            gesamt_zeit += end_time - start_time
        print(f"Gesamtzeit des Trainings::", round(gesamt_zeit/60, 2), "Minuten")

                    