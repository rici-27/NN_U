from tensor import Tensor
from layers import *


class Network():

    def __init__(self, input_layer: Layer, layers: list, loss_layer: Layer):
        self.input_layer = input_layer
        self.layers = layers
        self.loss_layer = loss_layer

        self.predictions = [] # predictions soll am ende 10 einträge enthalten (hot one encoding)
        # vielleicht anders als liste implementieren
        self.loss = Tensor([-1]) # loss ist nur ein skalar

        tensors = []
        # für jede layer in layers einen tensor erzeugen reich!
        # letzter Tensor enthält wahre Datenn (oder ?)

    def forward(self, data):

        self.input_layer.forward(input = data, outTensor = self.tensors[0])
        
        for i in range(len(self.layers)):
            self.layers[i].forward(inTensor = self.tensors[i], outTensor = self.tensors[i+1])
        
        self.loss_layer.forward(inTensor=self.predictions, outTensor=self.loss)

    def backprop(self):

        self.loss_layer.backward(outTensor, inTensor)
        #eigentlich nur alles rückwärts oder
        # hier passiert die backpropagation

        # und dann nochmal vorwärts durch calculate deltas
        return
       
    def saveParams(self, folder_path):
        pass

    def loadParams(self, folder_path):
        pass

    # hier kommen die loss funktionen rein (abhängig von wahren labels?)
    # layer und input layer