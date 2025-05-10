import numpy as np
from abc import ABC, abstractmethod
from functions import tanH, ReLu, sigmoid, softmax

# nochmal schauen ob alles row oder collumn wise

class Layer(ABC):
    
    # noch optimieren
    def __init__(self, inShape, outShape):
        self.inShape = inShape
        self.outShape = outShape
    # unsicher ob ich beide brauche

    @abstractmethod
    def forward(self, inTensor, outTensor):
        pass
    
    # beachte unterschied backward und backprop
    @abstractmethod
    def backward(self, outTensor, inTensor):
        pass 

    @abstractmethod
    def calculate_delta_weights(self, inTensor, outTensor):
        pass
    """ das ist hier das Parameter Update, nicht Teil der Backpropagation """

    @abstractmethod
    def update_weights(self, inTensor, outTensor):
        pass
    
    
class Input_Layer_MNIST(Layer):
    # Soll die Daten in einen von unseren Tensoren unwandeln
    # kriegen wahrscheinlich array und deltas muss erstmal leer initialisiert werden
    """ shape einarbeiten! sind alles vektoren?"""
    # Frage: Wollen wir es mit einer Liste machen? batch dies das (eher nicht)

    def __init__(self, inShape, outShape):
        self.inShape = inShape
        self.outShape = outShape

    def forward(self, data, outTensor):
        pass


# Fully connected Layer
class FCN_Layer(Layer):

    def __init__(self, weight, bias, inShape, outShape):
        self.weight = weight
        self.bias = bias
        self.inShape = inShape
        self.outShape = outShape
    # wichtig: weight und bias auch als tensor speichern

    def forward(self, inTensor, outTensor):
        outTensor.elements = np.mathmul(inTensor.elements, self.weight.elements) + self.bias.elements 

    def backward(self, inTensor, outTensor):
        return np.dot(outTensor.deltas, self.weight.elements.T)
        # output muss dann direkt in korrekten Tensor abgespeichert werden

    def calculate_delta_weights(self, inTensor, outTensor):
        self.weight.deltas = np.outer(inTensor.elements, outTensor.deltas)
        self.bias.deltas = outTensor.deltas
    
    def update_weights(self, inTensor, outTensor):
        pass


class ACT_Layer_sigmoid(Layer):

    def __init__(self, inShape, outShape):
        self.inShape = inShape
        self.outShape = outShape

    # müssen wir hier auch noch in und outshape übergeben?
    def forward(self, inTensor, outTensor):
        outTensor.elements = sigmoid(inTensor.elements)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = (outTensor.elements * (1 - outTensor.elements)) * outTensor.deltas
    # elementweise Multiplikation


class ACT_Layer_ReLu(Layer):

    def __init__(self, inShape, outShape):
        self.inShape = inShape
        self.outShape = outShape

    def forward(self, inTensor, outTensor):
        outTensor.elements = ReLu(inTensor.elements)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = (1+np.sign(inTensor.elements))/2 * outTensor.deltas


class ACT_Layer_tanH(Layer):

    def __init__(self, inShape, outShape):
        self.inShape = inShape
        self.outShape = outShape

    def forward(self, inTensor, outTensor):
        outTensor.elements = tanH(inTensor.elements)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = (1- outTensor.elements * outTensor.elements) * outTensor.deltas


class Softmax_Layer(Layer):

    def __init__(self, inShape, outShape):
        self.inShape = inShape
        self.outShape = outShape

    def forward(inTensor, outTensor):
        w = np.sum(np.exp(inTensor.elements))
        outTensor.elements = softmax(inTensor.elements, w)
    
    def backward(inTensor, outTensor):
        temp = np.exp(inTensor.elements)
        # das noch effizienter machen :-)
        w = np.sum(np.exp(inTensor.elements))
        ableitungnachx = (np.diag(w * np.exp(inTensor.elements)) - np.outer(np.exp(inTensor.elements),  np.exp(inTensor.elements)))/w**2
        inTensor.deltas = np.matmul(ableitungnachx, outTensor.deltas)

class MSE_Loss_Layer(Layer):

    def __init__(self, inShape, outShape):
        self.inShape = inShape
        self.outShape = outShape

    # outTensor enthält hier die Labels
    # dh der loss wird muss tatsächlich returned werden
    def forward(self, inTensor, outTensor):
        return (1/inTensor.elements.shape[0]) * sum((inTensor.elements - outTensor.elements)**2)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = (2/inTensor.elements.shape[0]) * (inTensor.elements - outTensor.elements)

class Cross_Entropy_Loss_Layer(Layer):

    def __init__(self, inShape, outShape):
        self.inShape = inShape
        self.outShape = outShape

    def forward(self, inTensor, outTensor):
        return - sum(np.log(inTensor.elements) * outTensor.elements)
        # hier aufpassen wenn inTensor.elements zu klein ist
        # hier wird summiert, brauche shape ?

    def backward(self, inTensor, outTensor):
        inTensor.deltas = - outTensor.elements / inTensor.elements