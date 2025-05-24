from tensor import Tensor
from layer import *
import numpy as np
import pickle
import os

# Netzwerk Klasse

class Network():

    def __init__(self, input_layer: Layer, layers: list, loss_layer: Layer):
        self.input_layer = input_layer
        self.layers = layers
        self.loss_layer = loss_layer
        
        self.loss = Tensor([0])
        self.tensor_list = []

    def forward(self, data):
        # kucken wo data eingeht

        self.tensor_list = [] # vllt unnötig
        t0= Tensor(np.zeros([self.input_layer.outShape]))
        self.tensor_list.append(t0)

        self.input_layer.forward(data, outTensor = self.tensor_list[0])
        
        for layer in self.layers:
            t = Tensor(np.zeros([layer.outShape]))
            self.tensor_list.append(t)
            
            layer.forward(inTensor = self.tensor_list[-2], outTensor = self.tensor_list[-1])


    def backprop(self, labels):
        
        self.loss.elements = self.loss_layer.forward(inTensor=self.tensor_list[-1], outTensor=labels)
        
        self.loss_layer.backward(outTensor=labels, inTensor=self.tensor_list[-1])
        
        for i in reversed(range(0, len(self.layers))):
            self.layers[i].backward(outTensor = self.tensor_list[i+1], inTensor = self.tensor_list[i])
        for i in range(0,len(self.layers)):
            if type(self.layers[i]) == FCN_Layer:
                self.layers[i].calculate_delta_weights(inTensor = self.tensor_list[i], outTensor = self.tensor_list[i+1])
            

    def saveParams(self, folder_path):
        dict = {}
        for layer in self.layers:
            if type(layer) == FCN_Layer: # das hier noch ändern und layer_type entfernen
                dict[f"weight_{layer.num}"] = layer.weight.elements
                dict[f"bias_{layer.num}"] = layer.bias.elements
        
        file_path = os.path.join(folder_path, "params.pkl")
        
        with open(file_path, 'wb') as f:
            pickle.dump(dict, f)


    def loadParams(self, folder_path):
        file_path = os.path.join(folder_path, "params.pkl")
        with open(file_path, "rb") as f:
            parameter = pickle.load(f)  
        fcn_layers = [layer for layer in self.layers if type(layer) == FCN_Layer] 
        for (k, layer) in enumerate(fcn_layers):
            layer.weight.elements = parameter[f"weight_{k+1}"]
            layer.bias.elements = parameter[f"bias_{k+1}"]
                


