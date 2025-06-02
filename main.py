from tensor import Tensor
from network import *
from abc import ABC
import numpy as np
import time
import keras
from optimizer import SGDTrainer
from layer import *
import argparse
import ast


def run_model(folder_path, train=True, net = "FCN"):

    # Abkürzungen für die config Datei
    class_map = {
        "fcn": FCN_Layer,
        "sigmoid": ACT_Layer_sigmoid,
        "tanh": ACT_Layer_tanH,
        "relu": ACT_Layer_ReLu,
        "softmax": Softmax_Layer,
        "cnn": Conv2DLayer,
        "maxpool": Pooling2D,
        "flatten": Flatten
    }

    # Layer für das Netzwerk erstellen

    if net == "FCN":
        input_layer = Input_Layer_MNIST_FCN(np.array([28, 28, 1]))
        loss_layer = Cross_Entropy_Loss_Layer()  
        layers = []

        # Datei einlesen und Objekte erzeugen
        with open("config2.txt", "r") as file:
            for line in file:
                parts = [part.strip() for part in line.strip().split(",")]
                class_name = parts[0]
                args = list(map(int, parts[1:]))

                if class_name in class_map:
                    obj = class_map[class_name](*args)
                    layers.append(obj)
                else:
                    print(f"Unbekannte Klasse: {class_name}")

    if net == "CNN":
        input_layer = Input_Layer_MNIST_CNN(np.array([28, 28, 1])) 
        loss_layer = Cross_Entropy_Loss_Layer() 
        layers = []

        # Datei einlesen und Objekte erzeugen
        with open("config1.txt", "r") as file:
            for line in file:
                parts = [part.strip() for part in line.strip().split(",")]
                class_name = parts[0]
                args = list(map(int, parts[1:]))

                if class_name in class_map:
                    obj = class_map[class_name](*args)
                    layers.append(obj)
                else:
                    print(f"Unbekannte Klasse: {class_name}")


    # Netzwerk definieren
    network_mnist = Network(input_layer=input_layer, layers=layers, loss_layer=loss_layer)

    # Testdaten werden geladen und in ein passendes Format gebracht. /255 aufgrund der Graustufen
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255

    # One-Hot-Encoding für Trainingsdata
    y_train_onehot = np.zeros([y_train.size, 10])
    y_train_onehot[np.arange(y_train.size), y_train] = 1

    train_data = (x_train, y_train_onehot)

    # Netzwerk trainieren bzw. Parameter einlesen
    if train == True:
        trainer = SGDTrainer(0.01, 2)
        trainer.optimizing(network_mnist, train_data)
        network_mnist.saveParams(folder_path, net)

    else:
        network_mnist.loadParams(folder_path, net)

    # Netzwerk testen
    mistakes = 0

    for (x, y) in zip(x_test, y_test):
        network_mnist.forward(x)
        pred = np.argmax(network_mnist.tensor_list[-1].elements)
        if pred != y:
            mistakes += 1

    err = mistakes / len(y_test)

    print("Fehlerquote: ", err)
    print("Accuracy: ", 1-err)

# Hier passenden Ordnerpfad eingeben wie in README beschrieben
folder_path = r"C:\Users\Anwender\Desktop\Neuronale Netze"
#folder_path = r"C:\Users\Simon\Desktop\Neuronale Netze"
#folder_path = f"/Users/ricardabuttmann/Desktop/NN"


# Zweites Argument (True/ False) gibt an, ob der Trainingsmodus aktiviert werden soll
# Drittes Argument ('FCN'/ 'CNN') gibt an, welches Netzwerk genutzt werden soll
run_model(folder_path, True, "FCN")