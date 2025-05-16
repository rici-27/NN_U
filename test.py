#%%
import pickle
import os
from layer import *



layers = []
layers.append(FCN_Layer(inShape=8, outShape=4, num = 1))
layers.append(ACT_Layer_sigmoid(inShape=4))
layers.append(FCN_Layer(inShape=4, outShape=2, num = 2))
layers.append(ACT_Layer_tanH(inShape=2))
layers.append(FCN_Layer(inShape=2, outShape=1, num = 3))
layers.append(Softmax_Layer(inShape=1))


folder_path = r"C:\Users\Anwender\Desktop\Neuronale Netze"



def saveParams(folder_path):
    dict = {}
    for layer in layers:
        if layer.layer_type == "FCN":
            dict[f"weight_{layer.num}"] = layer.weight.elements
            dict[f"bias_{layer.num}"] = layer.bias.elements
    
    file_path = os.path.join(folder_path, "params.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(dict, f)


def loadParams(folder_path):
    file_path = os.path.join(folder_path, "params.pkl")
    with open(file_path, "rb") as f:
        parameter = pickle.load(f)  
    fcn_layers = [layer for layer in layers if layer.layer_type == "FCN"] 
    for (k, layer) in enumerate(fcn_layers):
        layer.weight.elements = parameter[f"weight_{k+1}"]
        layer.bias.elements = parameter[f"bias_{k+1}"]

saveParams(folder_path)
file_path = os.path.join(folder_path, "params.pkl")

with open(file_path, "rb") as f:
        parameter = pickle.load(f) 

#%%

parameter["weight_1"] = np.zeros([8,4])
# %%
fcn_layers = [layer for layer in layers if layer.layer_type == "FCN"] 
for (k, layer) in enumerate(fcn_layers):
        layer.weight.elements = parameter[f"weight_{k+1}"]
        layer.bias.elements = parameter[f"bias_{k+1}"]

# %%
parameter["weight_1"]
# %%
print(layers[0].weight.elements)
# %%
