import numpy as np
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
    def __repr__(self):
        return f"FCN_Layer(inShape = {self.inShape}, outShape = {self.outShape}, num = {self.num})" ## anpassen

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
        self.out = np.zeros((self.outShape[0] * self.outShape[1], self.amount))
        self.out_backward = np.zeros((self.inShape[0] * self.inShape[1], self.inShape[2]))


    def __repr__(self):
        return f"Conv2DLayer(inShape = self.inShape, x_length = self.x_length, y_length = self.y_length, amount = self.amount)"

    # def forward(self, inTensor, outTensor):
    #     for k in range(self.amount):
    #         for i in range(self.outShape[0]):
    #             for j in range(self.outShape[1]):
    #                 submatrix = inTensor.elements[i : i + self.x_length, j : j + self.y_length, :]
    #                 outTensor.elements[i, j, k] = np.sum(submatrix * self.weight.elements[ :, :, :, k]) + self.bias.elements[k]


    # def backward(self, inTensor, outTensor):
    #     padded_deltas = self.padding(outTensor)
    #     new_filter = np.rot90(self.weight.elements.transpose(0, 1, 3, 2), k=2)
    #     for k in range(self.inShape[2]):
    #         for i in range(self.inShape[0]):
    #             for j in range(self.inShape[1]):
    #                 submatrix = padded_deltas[i : i + self.x_length, j : j + self.y_length, :]
    #                 inTensor.deltas[i, j, k] = np.sum(submatrix * new_filter[ :, :, :, k])
        
    
    def padding(self, outTensor):
        padded_matrix = np.zeros([self.outShape[0] + 2 * self.x_length - 2, self.outShape[1] + 2 * self.y_length - 2, self.amount])
        padded_matrix[self.x_length - 1: self.outShape[0] + (self.x_length-1), self.y_length - 1: self.outShape[1] + (self.y_length-1), :] = outTensor.deltas
        return padded_matrix
    
    # def calculate_delta_weights(self, inTensor, outTensor):
    #     for k in range(self.amount):
    #         for n in range(self.inShape[2]):
    #             self.weight.deltas[:, :, n, k] = convolut(inTensor.elements[:, :, n], outTensor.deltas[:, :, k])
    #     for k in range(self.amount):
    #         self.bias.deltas[k] = np.sum(outTensor.deltas[:,:,k])


    def forward(self, inTensor, outTensor):
        # for (inTensor, outTensor) in zip(inTensors, outTensors):
        #     if self.padding == 'Full' or self.padding == 'Half':
        #         self.padded_tensor.elements[self.padding_x:-1*self.padding_x,
        #                                     self.padding_y:-1*self.padding_y, :] = inTensor.elements[:, :, :]
        #     else:
        #         self.padded_tensor.elements = inTensor.elements
        #     # convolution, work with block to save on the channel-for-loop
        for f in range(0, self.amount):
            # create windows of the size of the kernel over the input
            window = np.lib.stride_tricks.sliding_window_view(inTensor.elements, (
                self.x_length, self.y_length, self.depth))
            # compute tensorproduct over the window and the filter
            filter = self.weight.elements[:, :, :,f]
            self.out[:, f] = np.tensordot(window, filter, axes=3).flatten(order='F')
            #print("Größe eines outs:", np.shape(np.tensordot(window, filter, axes=3).flatten(order='F')))
        # reshape into outShape
        #print(np.shape(self.out))
        outTensor.elements = np.reshape(
            self.out, self.outShape, 'F') + self.bias.elements

    def backward(self, outTensor, inTensor):
        # for (inTensor, outTensor) in zip(inTensors, outTensors):
        #     if self.padding == 'None' or self.padding == 'Half':
        #         self.padded_tensor_backward.deltas[self.padding_x:-1*self.padding_x,
        #                                            self.padding_y:-1*self.padding_y, :] = outTensor.deltas[:, :, :]
        #     else:
        #         self.padded_tensor_backward.deltas = outTensor.deltas
        padded_deltas = self.padding(outTensor)
        rot_trans_filter = Tensor(
            np.rot90(np.transpose(self.weight.elements, (0, 1, 3, 2)), 2))
        # convolution, work with block to save on the channel-for-loop
        # out = []
        for f in range(0, self.inShape[2]):
            # create windows of the size of the kernel over the input
            window = np.lib.stride_tricks.sliding_window_view(padded_deltas, (
                self.x_length, self.y_length, self.amount))
            # compute tensorproduct over the window and the filter
            filter = rot_trans_filter.elements[:, :, :, f]
            self.out_backward[:, f] = np.tensordot(window, filter, axes=3).flatten(order='F')
        # reshape into outShape
        inTensor.deltas = np.reshape(self.out_backward, self.inShape, 'F')

    def calculate_delta_weights(self, outTensor, inTensor):
        # filter delta update
        out = []
        for f in range(0, self.outShape[-1]):
            for ch in range(0, self.inShape[2]):
                # get window, depending on the channel and the filter
                window = np.lib.stride_tricks.sliding_window_view(inTensor.elements[:, :, ch],
                                                                    window_shape=np.shape(outTensor.deltas[:, :, f]))
                # compute tensorproduct over the window (inTensor(channel) and the kernel (outTensor.deltas(filter))
                out = np.append(out, np.transpose(np.tensordot(window, outTensor.deltas[:, :, f])))
            # reshape into shape of filter and update
        self.weight.deltas = np.reshape(out, shape=np.shape(self.weight.elements), order='F')

        # bias delta update
        for k in range(0, np.shape(self.bias.elements)[0]):
            self.bias.deltas[k] = np.sum(outTensor.deltas[:, :, k])



class Pooling2D(Layer):
    def __init__(self, inShape1, inShape2, inShape3, outShape1, outShape2, outShape3, x_length, y_length, axis=0, stride=(1, 1)):
        self.inShape = np.array([inShape1, inShape2, inShape3])
        self.outShape = np.array([outShape1, outShape2, outShape3])
        self.kernel_size = np.array([x_length, y_length])
        self.axis = axis
        if self.axis:
            self.order = 'C'
        else:
            self.order = 'F'
        self.stride = stride
        self.num_channels = self.inShape[-1]
        self.mask = np.zeros((self.num_channels, self.outShape[0] * self.outShape[1]), dtype=int)
        self.inDeltas_flat = np.zeros((self.inShape[0] * self.inShape[1],))
        self.outDeltas_flat = np.zeros((self.outShape[0] * self.outShape[1],))

    def forward(self, inTensor, outTensor):
        a0, a1 = self.axis, abs(self.axis - 1)
        for j in range(self.outShape[a1]):
            for i in range(self.outShape[a0]):
                start_i, start_j = i * self.stride[a0], j * self.stride[a1]
                end_i, end_j = start_i + self.kernel_size[a0], start_j + self.kernel_size[a1]
                for ch in range(0, self.num_channels):
                    outTensor.elements[i, j, ch] = np.max(inTensor.elements[start_i:end_i, start_j:end_j, ch])
                    relativ_index = np.argmax(inTensor.elements[start_i:end_i, start_j:end_j, ch].flatten(order=self.order))
                    self.mask[ch, i + j* self.outShape[a0]] = start_i + start_j*(self.inShape[a0]) + relativ_index + relativ_index//self.kernel_size[a0] * (self.inShape[a0] - self.kernel_size[a0])

    def backward(self, outTensor, inTensor):
        for ch in range(0, self.num_channels):
            self.inDeltas_flat *= 0
            self.outDeltas_flat = outTensor.deltas[:, :, ch].flatten(order=self.order)
            self.inDeltas_flat[self.mask[ch,:]] = self.outDeltas_flat
            inTensor.deltas[:, :, ch] = np.reshape(self.inDeltas_flat, (self.inShape[0], self.inShape[1]))


class Flatten(Layer):
    
    def __init__(self, inShape1, inShape2, inShape3, outShape):
        self.inShape = np.array([inShape1, inShape2, inShape3])
        self.outShape = outShape
    
    def __repr__(self):
        return f"Flatten(inShape1 = self.inShape1, inShape2 = self.inShape2, inShape3= self.inShape3)"

    def forward(self, inTensor, outTensor):
        outTensor.elements = inTensor.elements.flatten()
        
    def backward(self, inTensor, outTensor):
        inTensor.deltas = outTensor.deltas.reshape(*self.inShape)

        

