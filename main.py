from tensor import Tensor
from network import *
from abc import ABC
import numpy as np
import time
import keras  
from optimizer import SGDTrainer
from layer import *
import argparse

def run_model(folder_path, train = True):
    input_layer = Input_Layer_MNIST(784)
    loss_layer = Cross_Entropy_Loss_Layer() # auf random

    # Hier werden nun die verschiedenen Layer angelegt, input und loss layer werden separat behandelt
    
    layers = []
    layers.append(FCN_Layer(inShape = 784, outShape = 196, num = 1))
    layers.append(ACT_Layer_sigmoid(inShape = 196))
    layers.append(FCN_Layer(inShape = 196, outShape = 98, num = 2))
    layers.append(ACT_Layer_tanH(inShape = 98))
    layers.append(FCN_Layer(inShape = 98, outShape = 10, num = 3))
    layers.append(Softmax_Layer(inShape = 10))
    
    network_mnist = Network(input_layer = input_layer, layers = layers, loss_layer = loss_layer)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train /255
    x_test = x_test /255

    # One-Hot-Encoding für Trainingsdata
    y_train_onehot = np.zeros([y_train.size, 10])
    y_train_onehot[np.arange(y_train.size), y_train] = 1
    
    train_data = (x_train, y_train_onehot)
    
    if train == True:
        trainer = SGDTrainer(0.001, 20)
        trainer.optimizing(network_mnist, train_data)
        network_mnist.saveParams(folder_path)
        
    else:
        network_mnist.loadParams(folder_path)
    
    mistakes = 0

    for (x, y) in zip(x_test, y_test):
        network_mnist.forward(x)
        pred = np.argmax(network_mnist.tensor_list[-1].elements)
        if pred != y:
            mistakes += 1
    
    print("Fehlerquote: ", mistakes/len(y_test))    
        
    # hier soll noch was ausgegeben und gespeichert werden
    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Decide mode for NN and insert folder path for parameter", add_help=False)
#     parser.add_argument("--train", type=bool, default=True, help="train or just evaluate?")
#     parser.add_argument("--folder_path", type=str, default = r"C:\Users\Anwender\Desktop\Neuronale Netze", help="insert folder path for parameters")
#     args = parser.parse_args()
    
#     run_model(args.train, args.folder_path) 


folder_path = r"C:\Users\Simon\Desktop\Neuronale Netze"


run_model(folder_path, True)