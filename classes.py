import numpy as np
from abc import ABC, abstractmethod
from functions import tanH, relu, sigmoid

class Tensor:
    def __init__(self, elements: np.array, deltas: np.array):

        self.elements = elements
        self.deltas = deltas

class Layer(ABC):
    @abstractmethod
    def backward(self, input_path, output_path):
        pass

    def forward(self, input_path, output_path):
        pass

    def calculate_delta_weights(self, input_path, output_path):
        pass

class FCN(Layer):

    def __init__(self, weight: np.array, bias: int):

        self.weight = weight
        self.bias = bias

    def backward(self, input_path, output_path):
        return np.dot(output_path.deltas, self.weight.T)

    def forward(self, input_path: Tensor, output_path: Tensor):
        return np.mathmul(input_path.elements,self.weight) + self.bias

    def calculate_delta_weights(self, input_path, output_path):
        return self.weight.T

class ACT_sigmoid(Layer):

    def __init__(self, weight: np.array, bias: int):
        self.weight = weight
        self.bias = bias

    def backward(self, input_path, output_path):
        return np.dot(output_path.deltas, self.calculate_delta_weights(input_path))

    def forward(self, input_path: Tensor, output_path: Tensor):
        return sigmoid(input_path.elements)

    def calculate_delta_weights(self, input_path, output_path):
        vsigmoid = np.vectorize(sigmoid)
        return np.multiply(vsigmoid(input_path.elements) , (1 - vsigmoid(input_path.elements)))



#   Objectorientierte   programmierung (vererbung)
