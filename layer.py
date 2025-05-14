import numpy as np
from abc import ABC, abstractmethod
from functions import tanH, ReLu, sigmoid, softmax
from tensor import Tensor


class Layer(ABC):
    
    def __init__(self, inShape, outShape, layer_type, num):
        self.inShape = inShape
        self.outShape = outShape
        self.layer_type = layer_type
        self.num = num

    @abstractmethod
    def forward(self, inTensor, outTensor):
        pass

    @abstractmethod
    def backward(self, outTensor, inTensor):
        pass 
    
   
class Input_Layer_MNIST(Layer):

    # in den anderen layern init anpassen, wenn notwendig
    def __init__(self, outShape, layer_type = "input_layer"):
        self.outShape = outShape
        self.layer_type = layer_type

    def forward(self, data, outTensor):
        outTensor.elements = data.flatten()

    def backward(self, outTensor, inTensor):
        pass


# Fully connected Layer
class FCN_Layer(Layer):

    def __init__(self, inShape, outShape, num, layer_type = "FCN"):
        self.inShape = inShape
        self.outShape = outShape
        self.num = num
        self.layer_type = layer_type
        self.weight = Tensor(np.random.uniform(
            low=-0.5, high=0.5, size=(self.inShape, self.outShape)))
        self.bias = Tensor(np.random.uniform(
            low=-0.5, high=0.5, size=(self.outShape)))

    def forward(self, inTensor, outTensor):
        outTensor.elements = np.matmul(inTensor.elements, self.weight.elements) + self.bias.elements 

    def backward(self, inTensor, outTensor):
        inTensor.deltas = np.matmul(outTensor.deltas, self.weight.elements.T)

    def calculate_delta_weights(self, inTensor, outTensor):
        self.weight.deltas = np.outer(inTensor.elements, outTensor.deltas)
        self.bias.deltas = outTensor.deltas

class ACT_Layer_sigmoid(Layer):

    def __init__(self, inShape, layer_type = "ACT"):
        self.inShape = inShape
        self.outShape = inShape
        self.layer_type = layer_type

    def forward(self, inTensor, outTensor):
        outTensor.elements = sigmoid(inTensor.elements)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = (outTensor.elements * (1 - outTensor.elements)) * outTensor.deltas


class ACT_Layer_ReLu(Layer):

    def __init__(self, inShape, layer_type = "ACT"):
        self.inShape = inShape
        self.outShape = inShape
        self.layer_type = layer_type

    def forward(self, inTensor, outTensor):
        outTensor.elements = ReLu(inTensor.elements)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = (1+np.sign(inTensor.elements))/2 * outTensor.deltas


class ACT_Layer_tanH(Layer):

    def __init__(self, inShape, layer_type = "ACT"):
        self.inShape = inShape
        self.outShape = inShape
        self.layer_type = layer_type

    def forward(self, inTensor, outTensor):
        outTensor.elements = tanH(inTensor.elements)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = (1- outTensor.elements * outTensor.elements) * outTensor.deltas


class Softmax_Layer(Layer):

    def __init__(self, inShape, layertype = "Softmax"):
        self.inShape = inShape
        self.outShape = inShape
        self.layer_type = layertype

    def forward(self, inTensor, outTensor):
        w = np.sum(np.exp(inTensor.elements))
        outTensor.elements = softmax(inTensor.elements, w)
    
    def backward(self, inTensor, outTensor):
        # das noch effizienter machen :-)
        w = np.sum(np.exp(inTensor.elements))
        derivative_wrt_input = (np.diag(w * np.exp(inTensor.elements)) - np.outer(np.exp(inTensor.elements),  np.exp(inTensor.elements)))/(w**2)
        inTensor.deltas = np.matmul(outTensor.deltas, derivative_wrt_input)

class MSE_Loss_Layer(Layer):

    # # # weiß nicht ob wir shape größe hier brauchen
    
    def __init__(self, layer_type = "Loss"):
        self.layer_type = layer_type

    # outTensor enthält hier die Labels
    # dh der loss wird muss tatsächlich returned werden
    def forward(self, inTensor, outTensor):
        return (1/inTensor.elements.shape[0]) * sum((inTensor.elements - outTensor.elements)**2)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = (2/inTensor.elements.shape[0]) * (inTensor.elements - outTensor.elements)
    

class Cross_Entropy_Loss_Layer(Layer):

    def __init__(self, layer_type = "Loss"):
        self.layer_type = layer_type

    def forward(self, inTensor, outTensor):
        return - sum(np.log(inTensor.elements) * outTensor.elements)
        # hier aufpassen wenn inTensor.elements zu klein ist
        # hier wird summiert, brauche shape ?

    def backward(self, inTensor, outTensor):
        inTensor.deltas = - outTensor.elements / inTensor.elements
        # hier aufpassen dass nicht durch null geteilt wird