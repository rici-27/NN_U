from tensor import Tensor
from layers import *
import numpy as np


class Network():

    def __init__(self, input_layer: Layer, layers: list, loss_layer: Layer, tensor_list = []):
        self.input_layer = input_layer
        self.layers = layers
        self.loss_layer = loss_layer
        self.loss = Tensor(0)

        self.tensor_list = tensor_list


    def forward(self, data):
        # kucken wo data eingeht

        t0= Tensor(np.zeros([self.input_layer.outShape[0]]), np.zeros([self.input_layer.outShape[0]]))
        self.tensor_list.append(t0)

        self.input_layer.forward(input = data, outTensor = self.tensor_list[0])
        
        for i in range(len(self.layers)):
            t = Tensor(np.zeros([self.layers[i].outshape[0]]), np.zeros([self.layers[i].outshape[0]]))
            self.tensor_list.append(t)

            self.layers[i].forward(inTensor = self.tensor_list[i], outTensor = self.tensor_list[i+1])
        
        self.loss_layer.forward(inTensor=self.tensor_list[-1], outTensor=self.loss)

    def backprop(self):


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