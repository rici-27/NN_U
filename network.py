from tensor import Tensor
from layers import *


class Network():

    def __init__(self, input_layer: Layer, layers: list, loss_layer: Layer):
        self.input_layer = input_layer
        self.layers = layers
        self.loss_layer = loss_layer

        # das Network muss auch die Tensoren haben, 
        # die werden abhängig von der Shape in den Layern erzeugt
        # tensoren sind dann in einer liste
        tensors = []

    def forward(self, data):

        self.input_layer.forward(data, self.tensors[0])
        
        for i in range(len(self.layers)):
            self.layers[i].forward(self.tensors[i], self.tensors[i+1])

        # eigentlich wird hier nur iterativ aufgerufen oder
        pass
        #return loss 

    def backprop():
        # hier passiert die backpropagation
        return
       
    def saveParams(self, folder_path):
        pass

    def loadParams(self, folder_path):
        pass

    # hier kommen die loss funktionen rein (abhängig von wahren labels?)
    # layer und input layer