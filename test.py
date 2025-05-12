import pickle
import os

dict = {}

dict[1] = 2
dict[3] = 4

file_path = r"C:\Users\Anwender\Desktop\Neuronale Netze\params.pkl"

with open(file_path, "wb") as f:
    pickle.dump(dict, f)


