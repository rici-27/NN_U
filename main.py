from tensor import Tensor
from network import *
from abc import ABC
import numpy as np
import time
import keras
from optimizer import SGDTrainer
from layer import *

def run_model(train = True):
    input_layer = Input_Layer_MNIST(784)
    loss_layer = MSE_Loss_Layer # auf random
    
    # hier einfach mal paal layer anlegen
    
    layers = []
    layers.append(FCN_Layer(inShape=784, outShape=196, num = 1))
    layers.append(ACT_Layer_sigmoid(inShape=196))
    layers.append(FCN_Layer(inShape=196, outShape=98, num = 2))
    layers.append(ACT_Layer_tanH(inShape=98))
    layers.append(FCN_Layer(inShape=98, outShape=10, num = 3))
    layers.append(Softmax_Layer(inShape=10))
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train /255
    x_test = x_test /255

    # One-Hot-Encoding für Trainingsdata
    y_train_onehot = np.zeros([y_train.size, 10])
    print(y_train.size)
    y_train_onehot[np.arange(y_train.size), y_train] = 1
    
    train_data = (x_train, y_train_onehot)

    
    network_mnist = Network(input_layer=input_layer, layers=layers, loss_layer=loss_layer)
    
    if train == True:
        trainer = SGDTrainer(0.5, 10)
        trainer.optimizing(network_mnist, train_data)
        network_mnist.saveParams()
        
    network_mnist.loadParams()
    
    mistakes = 0
    for (x, y) in (x_test, y_test):
        network_mnist.forward(x)
        pred = np.argmax(network_mnist.tensor_list[-1].elements)
        if pred != y:
            mistakes =+ 1
    
    print("Accuracy: ", mistakes/len(x_test))    
        
    # hier soll noch was ausgegeben und gespeichert werden
    
if __name__ == "__main__":
    run_model()