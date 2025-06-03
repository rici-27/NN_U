from tensor import Tensor
from network import *
from abc import ABC
import numpy as np
import time
import keras
from optimizer import SGDTrainer
from layer import *


def run_model(folder_path, train=True, net = "FCN"):


    if net == "FCN":
        input_layer = Input_Layer_MNIST_FCN(np.array([28, 28, 1]))
        loss_layer = Cross_Entropy_Loss_Layer(10)  
        layers = []

        fcn1 = FCN_Layer(784, 196, 1)
        layers.append(fcn1)
        
        sigmoid = ACT_Layer_sigmoid((196,))
        layers.append(sigmoid)
        
        fcn2 = FCN_Layer(196, 98, 2)
        layers.append(fcn2)
        
        tanh = ACT_Layer_tanH((98,))
        layers.append(tanh)
        
        fcn3 = FCN_Layer(98, 10, 3)
        layers.append(fcn3)
        
        softmax = Softmax_Layer(10)
        layers.append(softmax)
        
    if net == "CNN":
        input_layer = Input_Layer_MNIST_CNN(np.array([28, 28, 1])) 
        loss_layer = Cross_Entropy_Loss_Layer(10) 
        layers = []

        cnn1 = Conv2DLayer(28, 28, 1, 2, 2, 3, 1)
        layers.append(cnn1)
        maxpool1 = Pooling2D(27, 27, 3, 14, 14, 3, 2, 2, stride = (2,2))
        layers.append(maxpool1)
        relu1 = ACT_Layer_ReLu((14, 14, 3))
        layers.append(relu1)
        
        cnn2 = Conv2DLayer(14, 14, 3, 5, 5, 6, 2)
        layers.append(cnn2)
        maxpool2 = Pooling2D(10, 10, 3, 5, 5, 6, 2, 2, stride = (2,2))
        layers.append(maxpool2)
        relu2 = ACT_Layer_ReLu((5, 5, 6))
        layers.append(relu2)
        
        flatten = Flatten(5, 5, 6, 150)
        layers.append(flatten)
        
        fcn1 = FCN_Layer(150, 100, 1)
        layers.append(fcn1)
        
        sigmoid = ACT_Layer_sigmoid(100)
        layers.append(sigmoid)
        
        fcn2 = FCN_Layer(100, 50, 2)
        layers.append(fcn2)
        
        tanh = ACT_Layer_tanH(50)
        layers.append(tanh)
        
        fcn3 = FCN_Layer(50, 10, 3)
        layers.append(fcn3)
        
        softmax = Softmax_Layer(10)
        layers.append(softmax)
        
        
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
        trainer = SGDTrainer(0.01, 10)
        trainer.optimizing(network_mnist, train_data)
        network_mnist.saveParams(folder_path, net)

    else:
        network_mnist.loadParams(folder_path, net)

    # Netzwerk testen
    mistakes = 0
    start_time = time.time()

    for (x, y) in zip(x_test, y_test):
        network_mnist.forward(x)
        pred = np.argmax(network_mnist.tensor_list[-1].elements)
        if pred != y:
            mistakes += 1

    err = mistakes / len(y_test)
    end_time = time.time()

    print("Fehlerquote: ", err)
    print("Accuracy: ", 1-err)
    print("Dauer der Auswertung:", round(end_time - start_time, 2), "Sekunden")

# Hier passenden Ordnerpfad eingeben wie in README beschrieben

#folder_path = r"C:\Users\Anwender\Desktop\Neuronale Netze"
#folder_path = r"C:\Users\Simon\Desktop\Neuronale Netze"
folder_path = f"/Users/ricardabuttmann/Desktop/NN/UB1"


# Zweites Argument (True/ False) gibt an, ob der Trainingsmodus aktiviert werden soll
# Drittes Argument ('FCN'/ 'CNN') gibt an, welches Netzwerk genutzt werden soll
run_model(folder_path, False, "FCN")