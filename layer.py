﻿import numpy as np
from abc import ABC, abstractmethod
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
        outTensor.elements = data.reshape(28, 28, 1)

    def backward(self, outTensor, inTensor):
        pass


class FCN_Layer(Layer):

    def __init__(self, inShape, outShape, num):
        self.inShape =  inShape
        self.outShape = outShape
        self.num = num
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

    def __init__(self, inShape):
        self.inShape = inShape
        self.outShape = inShape

    def forward(self, inTensor, outTensor):
        outTensor.elements = sigmoid(inTensor.elements)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = (outTensor.elements * (1 - outTensor.elements)) * outTensor.deltas


class ACT_Layer_ReLu(Layer):

    def __init__(self, inShape):
        self.inShape = inShape
        self.outShape = inShape

    def forward(self, inTensor, outTensor):
        outTensor.elements = ReLu(inTensor.elements)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = (1+np.sign(inTensor.elements))/2 * outTensor.deltas


class ACT_Layer_tanH(Layer):

    def __init__(self, inShape):
        self.inShape = inShape
        self.outShape = inShape

    def forward(self, inTensor, outTensor):
        outTensor.elements = tanH(inTensor.elements)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = (1- outTensor.elements * outTensor.elements) * outTensor.deltas


class Softmax_Layer(Layer):

    def __init__(self, inShape):
        self.inShape = inShape
        self.outShape = inShape

    def forward(self, inTensor, outTensor):
        w = np.sum(np.exp(inTensor.elements))
        outTensor.elements = softmax(inTensor.elements, w)
    
    def backward(self, inTensor, outTensor):
        w = np.sum(np.exp(inTensor.elements))
        derivative_wrt_input = (np.diag(w * np.exp(inTensor.elements)) - np.outer(np.exp(inTensor.elements),  np.exp(inTensor.elements)))/(w**2)
        inTensor.deltas = np.matmul(outTensor.deltas, derivative_wrt_input)


class MSE_Loss_Layer(Layer):

    def __init__(self, inShape, outShape=1):
        self.inShape = inShape
        self.outShape = outShape

    def forward(self, inTensor, outTensor):
        return (1/inTensor.elements.shape[0]) * sum((inTensor.elements - outTensor.elements)**2)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = (2/inTensor.elements.shape[0]) * (inTensor.elements - outTensor.elements)
    

class Cross_Entropy_Loss_Layer(Layer):

    def __init__(self, inShape, outShape=1):
        self.inShape = inShape
        self.outShape = outShape

    def forward(self, inTensor, outTensor):
        return - sum(np.log(inTensor.elements + 1e-12) * outTensor.elements)

    def backward(self, inTensor, outTensor):
        inTensor.deltas = - outTensor.elements / inTensor.elements + 1e-12


class Conv2DLayer(Layer):
    
    def __init__(self, inShape, x_length, y_length, amount, num):
        self.num = num
        self.inShape = inShape
        self.outShape = np.array([self.inShape[0] - x_length + 1, self.inShape[1] - y_length + 1, amount])
        self.x_length = x_length
        self.y_length = y_length
        self.depth = self.inShape[-1]
        self.amount = amount
        self.bias = Tensor(np.random.uniform(low = -0.5, high = 0.5, size =(self.amount)))
        self.weight = Tensor(np.random.uniform(low = -0.5,
                                               high = 0.5,
                                               size =(self.x_length, self.y_length, self.depth, self.amount)))
        self.out = np.zeros((self.outShape[0] * self.outShape[1], self.amount))
        self.out_backward = np.zeros((self.inShape[0] * self.inShape[1], self.depth))

    def padding(self, outTensor):
        padded_matrix = np.zeros([self.outShape[0] + 2 * self.x_length - 2, self.outShape[1] + 2 * self.y_length - 2, self.amount])
        padded_matrix[self.x_length - 1: -(self.x_length-1), self.y_length - 1: -(self.y_length-1), :] = outTensor.deltas
        return padded_matrix

    def forward(self, inTensor, outTensor):
        for f in range(0, self.amount):
            window = np.lib.stride_tricks.sliding_window_view(inTensor.elements, (
                self.x_length, self.y_length, self.depth))
            filter = self.weight.elements[:, :, :,f]
            self.out[:, f] = np.tensordot(window, filter, axes=3).flatten(order='F')
        outTensor.elements = np.reshape(
            self.out, self.outShape, 'F') + self.bias.elements

    def backward(self, outTensor, inTensor):
        padded_deltas = self.padding(outTensor)
        rot_trans_filter = Tensor(
            np.rot90(np.transpose(self.weight.elements, (0, 1, 3, 2)), 2, axes=(0,1)))
        
        for f in range(0, self.depth):
            window = np.lib.stride_tricks.sliding_window_view(padded_deltas, (
                self.x_length, self.y_length, self.amount))
            filter = rot_trans_filter.elements[:, :, :, f]
            self.out_backward[:, f] = np.tensordot(window, filter, axes=3).flatten(order='F')
        inTensor.deltas = np.reshape(self.out_backward, self.inShape, 'F')

    def calculate_delta_weights(self, outTensor, inTensor):
        out = []
        for f in range(0, self.amount):
            for ch in range(0, self.depth):
                window = np.lib.stride_tricks.sliding_window_view(inTensor.elements[:, :, ch],
                                                                    window_shape=np.shape(outTensor.deltas[:, :, f]))
                out = np.append(out, np.transpose(np.tensordot(window, outTensor.deltas[:, :, f])))
        self.weight.deltas = np.reshape(out, shape=np.shape(self.weight.elements), order='F')

        for k in range(0, self.amount):
            self.bias.deltas[k] = np.sum(outTensor.deltas[:, :, k])

class Pooling2D(Layer):
    def __init__(self, inShape, outShape, x_length, y_length, stride=(1, 1)):
        self.inShape = inShape
        self.outShape = outShape
        self.kernel_size = np.array([x_length, y_length])
        self.stride = stride
        self.depth = self.inShape[-1]
        self.mask = np.zeros((self.depth, self.outShape[0] * self.outShape[1]), dtype=int)
        self.inDeltas_flat = np.zeros((self.inShape[0] * self.inShape[1],))
        self.outDeltas_flat = np.zeros((self.outShape[0] * self.outShape[1],))

    def forward(self, inTensor, outTensor):
        for j in range(self.outShape[1]):
            for i in range(self.outShape[0]):
                start_i, start_j = i * self.stride[0], j * self.stride[1]
                end_i, end_j = start_i + self.kernel_size[0], start_j + self.kernel_size[1]
                for ch in range(0, self.depth):
                    outTensor.elements[i, j, ch] = np.max(inTensor.elements[start_i:end_i, start_j:end_j, ch])
                    relativ_index = np.argmax(inTensor.elements[start_i:end_i, start_j:end_j, ch].flatten('F'))
                    self.mask[ch, i + j* self.outShape[0]] = start_i + start_j*(self.inShape[0]) + relativ_index + relativ_index//self.kernel_size[0] * (self.inShape[0] - self.kernel_size[0])

    def backward(self, outTensor, inTensor):
        for ch in range(0, self.depth):
            self.inDeltas_flat *= 0
            self.outDeltas_flat = outTensor.deltas[:, :, ch].flatten('F')
            self.inDeltas_flat[self.mask[ch,:]] = self.outDeltas_flat
            inTensor.deltas[:, :, ch] = np.reshape(self.inDeltas_flat, (self.inShape[0], self.inShape[1]))


class Flatten(Layer):
    
    def __init__(self, inShape, outShape):
        self.inShape = inShape
        self.outShape = outShape

    def forward(self, inTensor, outTensor):
        outTensor.elements = inTensor.elements.flatten()
        
    def backward(self, inTensor, outTensor):
        inTensor.deltas = outTensor.deltas.reshape(*self.inShape)

        

