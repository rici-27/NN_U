import numpy as np
from abc import ABC, abstractmethod
from functions import tanH, relu, sigmoid


class Layer(ABC):
    
    # vorsicht, abstractmethod muss for jede (!) Methode neu geschrieben werden, die abstrakt sein soll
    @abstractmethod
    
    # könnten noch Attribut hinzufügen, ob die Layer Parameter enthält oder nicht

    @abstractmethod
    def forward(self, inTensor, outTensor):
        pass
    
    @abstractmethod
    def backward(self, outTensor, inTensor):
        pass 
    # würde mich hier an Benennung aus VL halten,
    # outTensor bezeichnet hier den, der im Netz später kommt,
    # auch wenn er hier als "Input" verarbeitet wird

    @abstractmethod
    def calculate_delta_weights(self, inTensor, outTensor):
        pass
    """ das ist hier eigentlich der Parameter Update, nicht Teil der Backpropagation """
    
    def saveParams():
        pass
    
    def loadParams():
        pass
    
class Input(Layer):
    # Soll die Daten in einen von unseren Tensoren unwandeln
    # kriegen wahrscheinlich array und deltas muss erstmal leer initialisiert werden
    # Frage: Wollen wir es mit einer Liste machen? batch dies das (eher nicht)
    pass


# Fully connected Layer
class FCN(Layer):


    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
    # wichtig: weight und bias auch als tensor speichern

    def forward(self, inTensor, outTensor):
        return np.mathmul(inTensor.elements, self.weight) + self.bias

    def backward(self, inTensor, outTensor):
        return np.dot(outTensor.deltas, self.weight.T)
        # output muss dann direkt in korrekten Tensor abgespeichert werden

    def calculate_delta_weights(self, inTensor, outTensor):
        # hier wird die ableitung nach w berechnet UND upgedated
        return self.weight.T

class ACT_sigmoid(Layer):

    def __init__(self, weight: np.array, bias: int):
        self.weight = weight
        self.bias = bias

    def forward(self, inTensor, outTensor):
        return sigmoid(inTensor.elements)

    def backward(self, inTensor, outTensor):
        return np.dot(outTensor.deltas, self.calculate_delta_weights(inTensor))

    # hier gibt es eigentlich kein delta_weights, kein parameter update
    def calculate_delta_weights(self, inTensor, outTensor):
        vsigmoid = np.vectorize(sigmoid)
        return np.multiply(vsigmoid(inTensor.elements) , (1 - vsigmoid(inTensor.elements)))


class Softmax(Layer):

    def forward(inTensor):
        return
    
    def backward(inTensor, outTensor):
        return
    
    def calculate_delta_weights(inTensor, outTensor):
        return


# irgendwo müssen wir auch den Loss ableiten, gibt es eine "Loss Layer" ?
