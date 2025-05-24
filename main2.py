from tensor import Tensor
from network import *
from abc import ABC
import numpy as np
import time
import keras
from optimizer import SGDTrainer
from layer import *
import argparse


def run_model(folder_path, train=True):

    # Layer für das Netzwerk erstellen

    input_layer = Input_Layer_MNIST(784)
    loss_layer = Cross_Entropy_Loss_Layer()  # auf random
    layers = []

    # Config file einlesen

    # Das isr nur dafür da, damit man in der Config die Namen abkürzen kann
    class_map = {
        "fnc": FCN_Layer,
        "sigmoid": ACT_Layer_sigmoid,
        "tanh": ACT_Layer_tanH,
        "relu": ACT_Layer_ReLu,
        "softmax": Softmax_Layer
    }

    # Datei einlesen und Objekte erzeugen
    with open("config.txt", "r") as file:
        content = file.read()

    try:
        with open("config.txt", "r") as file:
            for line in file:
                parts = [part.strip() for part in line.strip().split(",")]
                class_name = parts[0]
                args = list(map(int, parts[1:]))

                if class_name in class_map:
                    obj = class_map[class_name](*args)
                    layers.append(obj)
                else:
                    print(f"Unbekannte Klasse: {class_name}")

    except FileNotFoundError:
        print("Datei config.txt nicht gefunden.")
    except Exception as e:
        print(f"Fehler: {e}")

    # Netzwerk definieren
    network_mnist = Network(input_layer=input_layer, layers=layers, loss_layer=loss_layer)

    # Testdaten werden geladen und in ein passendes Format gebracht. /255 aufgrund der Graustufen
    # --> x_test sollen floats zwischen 0 und 1 sein
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255

    # One-Hot-Encoding für Trainingsdata
    y_train_onehot = np.zeros([y_train.size, 10])
    y_train_onehot[np.arange(y_train.size), y_train] = 1

    train_data = (x_train, y_train_onehot)

    # Netzwerk trainieren bzw. Parameter einlesen
    if train == True:
        trainer = SGDTrainer(0.001, 2)
        trainer.optimizing(network_mnist, train_data)
        network_mnist.saveParams(folder_path)

    else:
        network_mnist.loadParams(folder_path)

    # Netzwerk testen
    mistakes = 0

    for (x, y) in zip(x_test, y_test):
        network_mnist.forward(x)
        pred = np.argmax(network_mnist.tensor_list[-1].elements)
        if pred != y:
            mistakes += 1

    print("Fehlerquote: ", mistakes / len(y_test))

    # hier soll noch was ausgegeben und gespeichert werden


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Decide mode for NN and insert folder path for parameter", add_help=False)
#     parser.add_argument("--train", type=bool, default=True, help="train or just evaluate?")
#     parser.add_argument("--folder_path", type=str, default = r"C:\Users\Anwender\Desktop\Neuronale Netze", help="insert folder path for parameters")
#     args = parser.parse_args()

#     run_model(args.train, args.folder_path)


folder_path = r"C:\Users\Simon\Desktop\Neuronale Netze"

run_model(folder_path, True)