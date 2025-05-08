import numpy as np
from abc import ABC, abstractmethod
from functions import tanH, ReLu, sigmoid

# nochmal schauen ob alles row oder collumn wise

class Layer(ABC):
    
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
    
    
class Input_Layer_MNIST(Layer):
    # Soll die Daten in einen von unseren Tensoren unwandeln
    # kriegen wahrscheinlich array und deltas muss erstmal leer initialisiert werden
    """ shape einarbeiten! sind alles vektoren?"""
    # Frage: Wollen wir es mit einer Liste machen? batch dies das (eher nicht)

    def forward(self, data, outTensor):
        pass


# Fully connected Layer
class FCN_Layer(Layer):

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
    # wichtig: weight und bias auch als tensor speichern

    def forward(self, inTensor, outTensor):
        outTensor.elements = np.mathmul(inTensor.elements, self.weight.elements) + self.bias.elements
        # hier wird nichts returned
        # vorsicht, weights und bias sind auch 

    def backward(self, inTensor, outTensor):
        return np.dot(outTensor.deltas, self.weight.elements.T)
        # output muss dann direkt in korrekten Tensor abgespeichert werden

    def calculate_delta_weights(self, inTensor, outTensor):
        # hier wird die ableitung nach w berechnet UND upgedated
        # wir können die delta weights in weights.deltas abspeichern!
        return self.weight.T


class ACT_Layer_sigmoid(Layer):

    # müssen wir hier auch noch in und outshape übergeben?
    def forward(self, inTensor, outTensor):
        outTensor.elements = sigmoid(inTensor.elements)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = (outTensor.elements * (1 - outTensor.elements)) * outTensor.deltas
    # elementweise Multiplikation


class ACT_Layer_ReLu(Layer):

    def forward(self, inTensor, outTensor):
        outTensor.elements = ReLu(inTensor.elements)

    def backward(self, inTensor, outTensor):
        return np.dot(outTensor.deltas, self.calculate_delta_weights(inTensor))


class ACT_Layer_tanH(Layer):

    def forward(self, inTensor, outTensor):
        outTensor.elements = tanH(inTensor.elements)

    def backward(self, inTensor, outTensor):
        inTensor.deltas # fehlt


class Softmax_Layer(Layer):

    def forward(inTensor, outTensor):
        outTensor.elements = Softmax(inTensor.elements)

    # softmax noch definieren
    
    def backward(inTensor, outTensor):
        inTensor.deltas # fehlt


class MSE_Loss_Layer(Layer):


    # outTensor enthält hier die Labels
    # dh der loss wird muss tatsächlich returned werden
    def forward(self, inTensor, outTensor):
        # gibt ein skalar zurück
        # wie ist shape definiert? passt shape[0]
        return (1/inTensor.elements.shape[0]) *  np.sqrt(sum((inTensor.elements - outTensor.elements)**2))
        # hier aufpassen wenn inTensor.elements zu klein ist
        # hier wird summiert, brauche shape ?

    def backward(self, inTensor, outTensor):
        inTensor.deltas # fehlt

class Cross_Entropy_Loss_Layer(Layer):
    # gibt ein skalar zurück

    def forward(self, inTensor, outTensor):
        return - sum(np.log(inTensor.elements) * outTensor.elements)
        # hier aufpassen wenn inTensor.elements zu klein ist
        # hier wird summiert, brauche shape ?

    def backward(self, inTensor, outTensor):
        inTensor.deltas # fehlt