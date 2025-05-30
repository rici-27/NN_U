import numpy as np
from abc import ABC, abstractmethod

from keras.src.legacy.backend import bias_add

from functions import tanH, ReLu, sigmoid, softmax, convolut
from tensor import Tensor

# Verschiedene Layer die später für das Netzwerk verwendet werden sollen

class Layer(ABC):
    
    def __init__(self, inShape, outShape, num):
        self.inShape = inShape
        self.outShape = outShape
        self.num = num

    @abstractmethod
    def forward(self, inTensor, outTensor):
        pass

    @abstractmethod
    def backward(self, outTensor, inTensor):
        pass 
    
   
class Input_Layer_MNIST_FCN(Layer):

    def __init__(self, outShape):
        self.outShape = outShape

    def forward(self, data, outTensor):
        outTensor.elements = data.flatten()

    def backward(self, outTensor, inTensor):
        pass
    
# neue Input Layer für CNN:

class Input_Layer_MNIST_CNN(Layer):
    def __init__(self, outShape):
        self.outShape = outShape

    def forward(self, data, outTensor):
        outTensor.elements = data

    def backward(self, outTensor, inTensor):
        pass


class FCN_Layer(Layer):

    def __init__(self, inShape, outShape, num):
        self.inShape = inShape
        self.outShape = outShape
        self.num = num
        self.weight = Tensor(np.random.uniform(
            low=-0.5, high=0.5, size=(self.inShape, self.outShape)))
        self.bias = Tensor(np.random.uniform(
            low=-0.5, high=0.5, size=(self.outShape)))
    def __repr__(self):
        return f"FCN_Layer(inShape = self.inShape, outShape = self.outShape, num = self.num)"

    def forward(self, inTensor, outTensor):
        outTensor.elements = np.matmul(inTensor.elements, self.weight.elements) + self.bias.elements 

    def backward(self, inTensor, outTensor):
        inTensor.deltas = np.matmul(outTensor.deltas, self.weight.elements.T)

    def calculate_delta_weights(self, inTensor, outTensor):
        self.weight.deltas = np.outer(inTensor.elements, outTensor.deltas)
        self.bias.deltas = outTensor.deltas


class ACT_Layer_sigmoid(Layer):

    def __init__(self, inShape):
        self.inShape = inShape
        self.outShape = inShape

    def __repr__(self):
        return f"ACT_Layer_sigmoid(inShape = self.inShape)"

    def forward(self, inTensor, outTensor):
        outTensor.elements = sigmoid(inTensor.elements)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = (outTensor.elements * (1 - outTensor.elements)) * outTensor.deltas


class ACT_Layer_ReLu(Layer):

    def __init__(self, inShape):
        self.inShape = inShape
        self.outShape = inShape

    def __repr__(self):
        return f"ACT_Layer_ReLu(inShape = self.inShape, outShape = self.outShape)"

    def forward(self, inTensor, outTensor):
        outTensor.elements = ReLu(inTensor.elements)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = (1+np.sign(inTensor.elements))/2 * outTensor.deltas


class ACT_Layer_tanH(Layer):

    def __init__(self, inShape):
        self.inShape = inShape
        self.outShape = inShape

    def __repr__(self):
        return f"ACT_Layer_tanH(inShape = self.inShape)"

    def forward(self, inTensor, outTensor):
        outTensor.elements = tanH(inTensor.elements)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = (1- outTensor.elements * outTensor.elements) * outTensor.deltas


class Softmax_Layer(Layer):

    def __init__(self, inShape):
        self.inShape = inShape
        self.outShape = inShape

    def __repr__(self):
        return f"Softmax_Layer(inShape = self.inShape)"

    def forward(self, inTensor, outTensor):
        w = np.sum(np.exp(inTensor.elements))
        outTensor.elements = softmax(inTensor.elements, w)
    
    def backward(self, inTensor, outTensor):
        w = np.sum(np.exp(inTensor.elements))
        derivative_wrt_input = (np.diag(w * np.exp(inTensor.elements)) - np.outer(np.exp(inTensor.elements),  np.exp(inTensor.elements)))/(w**2)
        inTensor.deltas = np.matmul(outTensor.deltas, derivative_wrt_input)


class MSE_Loss_Layer(Layer):

    def forward(self, inTensor, outTensor):
        return (1/inTensor.elements.shape[0]) * sum((inTensor.elements - outTensor.elements)**2)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = (2/inTensor.elements.shape[0]) * (inTensor.elements - outTensor.elements)
    

class Cross_Entropy_Loss_Layer(Layer):

    def __init__(self, layer_type = "Loss"):
        self.layer_type = layer_type

    def forward(self, inTensor, outTensor):
        return - sum(np.log(inTensor.elements + 1e-12) * outTensor.elements)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = - outTensor.elements / inTensor.elements + 1e-12


# UB2: Convolution Layer

class Conv2DLayer(Layer):
    
    def __init__(self, inShape1, inShape2, inShape3, x_length, y_length, amount, num):
        self.num = num
        self.inShape = np.array([inShape1, inShape2, inShape3])
        self.outShape = np.array([self.inShape[0] - x_length + 1, self.inShape[1] - y_length + 1, amount])
        self.x_length = x_length
        self.y_length = y_length
        self.depth = self.inShape[-1]
        self.amount = amount
        self.bias = Tensor(np.random.uniform(low = -0.5, high = 0.5, size =(self.amount)))
        self.weight = Tensor(np.random.uniform(low = -0.5,
                                               high = 0.5,
                                               size =(self.x_length, self.y_length, self.depth, self.amount)))

    def __repr__(self):
        return f"Conv2DLayer(inShape = self.inShape, x_length = self.x_length, y_length = self.y_length, amount = self.amount)"

    def forward(self, inTensor, outTensor):
        for k in range(self.amount):
            for i in range(self.outShape[0]):
                for j in range(self.outShape[1]):
                    submatrix = inTensor.elements[i : i + self.x_length, j : j + self.y_length, :]
                    outTensor.elements[i, j, k] = np.sum(submatrix * self.weight.elements[ :, :, :, k]) + self.bias.elements[k]


    def backward(self, inTensor, outTensor):
        padded_deltas = self.padding(outTensor)
        new_filter = np.rot90(self.weight.elements.transpose(0, 1, 3, 2), k=2)
        for k in range(self.inShape[2]):
            for i in range(self.inShape[0]):
                for j in range(self.inShape[1]):
                    submatrix = padded_deltas[i : i + self.x_length, j : j + self.y_length, :]
                    inTensor.deltas[i, j, k] = np.sum(submatrix * new_filter[ :, :, :, k])
        
    
    def padding(self, outTensor):
        padded_matrix = np.zeros([self.outShape[0] + 2 * self.x_length - 2, self.outShape[1] + 2 * self.y_length - 2, self.amount])
        padded_matrix[self.x_length - 1: self.outShape[0] + (self.x_length-1), self.y_length - 1: self.outShape[1] + (self.y_length-1), :] = outTensor.deltas
        return padded_matrix
    
    def calculate_delta_weights(self, inTensor, outTensor):
        for k in range(self.amount):
            for n in range(self.inShape[2]):
                self.weight.deltas[:, :, n, k] = convolut(inTensor.elements[:, :, n], outTensor.deltas[:, :, k])
        
        for k in range(self.amount):
            self.bias.deltas[k] = np.sum(outTensor.deltas[:,:,k])



class Pooling2D(Layer):
    
    def __init__(self):
        pass

    def __repr__(self):
        pass

        
class Flatten(Layer):
    
    def __init__(self, inShape1, inShape2, inShape3, outShape):
        self.inShape = np.array([inShape1, inShape2, inShape3])
        self.outShape = outShape
        
    def forward(self, inTensor, outTensor):
        outTensor.elements = inTensor.elements.flatten()
        
    def backward(self, inTensor, outTensor):
        inTensor.deltas = outTensor.deltas.reshape(self.inShape[0], self.inShape[1], 1)
        
    def calculate_delta_weights():
        pass

