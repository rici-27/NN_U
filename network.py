from tensor import Tensor
from layer import *
import numpy as np
import pickle
import os


class Network():

    def __init__(self, input_layer: Layer, layers: list, loss_layer: Layer):
        self.input_layer = input_layer
        self.layers = layers
        self.loss_layer = loss_layer
        
        self.loss = Tensor([-1])
        self.tensor_list = []


    def forward(self, data):
        # kucken wo data eingeht

        self.tensor_list = [] # vllt unnötig
        t0= Tensor(np.zeros([self.input_layer.outShape[0]]))
        self.tensor_list.append(t0)

        self.input_layer.forward(input = data, outTensor = self.tensor_list[0])
        
        for layer in self.layers:
            t = Tensor(np.zeros([layer.outShape[0]]))
            self.tensor_list.append(t)

            layer.forward(inTensor = self.tensor_list[-2], outTensor = self.tensor_list[-1])
        
        self.loss_layer.forward(inTensor=self.tensor_list[-1], outTensor=self.loss)


    def backprop(self, labels):
        self.loss_layer.backward(outTensor=labels, inTensor=self.tensor_list[-1])
        
        for i in reversed(range(0, len(self.layers))):
            self.layers[i].backward(outTensor = self.tensor_list[i+1], inTensor = self.tensor_list[i])
        for i in range(0,len(self.layers)):
            self.layers[i].calculate_delta_weights(inTensor = self.tensor_list[i], outTensor = self.tensor_list[i+1])
            

    def saveParams(self, folder_path):
        dict = {}
        for layer in self.layers:
            if layer.layer_type == "FCN":
                dict[f"FCN_weight_",layer.num] = layer.weight.elements
                dict[f"FCN_bias_",layer.num] = layer.bias.elements
        
        file_path = os.path.join(folder_path, "params.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(dict, f)

        

    def loadParams(self, folder_path):
        pass

