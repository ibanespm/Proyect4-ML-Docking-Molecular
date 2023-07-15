import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import deepsmiles
converter = deepsmiles.Converter(rings=True, branches=True)

cuda0 = torch.device('cuda:0')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class CNN(torch.nn.Module):
  def __init__(self):
    super(CNN,self).__init__()
    self.layer1 =  nn.Sequential(
        nn.Conv1d(in_channels = 34 , out_channels = 128, kernel_size = 3, stride = 1, padding = 2),
        nn.ReLU(),
        nn.Conv1d(in_channels =  128 , out_channels = 64, kernel_size = 3, stride = 1, padding = 2),
        nn.ReLU(),
        nn.Conv1d(in_channels =  64 , out_channels = 32, kernel_size = 3, stride = 1, padding = 2),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2))
    self.fc =  nn.Sequential(
        nn.Linear(4320, 750),
        nn.ReLU(),
        nn.Linear(750, 1),
        nn.Sigmoid()
        )

  def forward(self, x):
    out = self.layer1(x)
    out = out.reshape(out.size(0), -1)
    output = self.fc(out)
    return output


model2 = torch.load('model_.pth') #Agregar , map_location=torch.device('cpu') si es necesario

LIG1 = "O(CCOCCOC)CCOCCO"
print("El ligando 1 es",LIG1)
# LIG1 no se une al receptor 
LIG2 = "O=C1NC2=C([C@@]13N4[C@@H](C[C@@H]3CC(=O)N5C/C(/C(=O)/C(/C5)=C/C=6C=7C(C=CC6)=CC=CC7)=C\C=8C=9C(C=CC8)=CC=CC9)CCC4)C=CC=C2"
print("El ligando 2 es",LIG2)
# LIG1 se une al receptor 
d = {'smiles':[LIG1, LIG2]}
smalldf = pd.DataFrame(data=d)

deep_SMILES = []
for i in range(len(smalldf)):
  deep_smile = converter.encode(smalldf.smiles[i])
  deep_SMILES.append(deep_smile)

smalldf["DeepSmiles"] = deep_SMILES

keys = ['8', 'C', '#', '9', ' ', '0', 'S', '@', '1', '2', 'B', '-', '7', 'O', 'l', ')', '\\', '/', 'I', '5', '6', '+', '3', '=', 'F', 'r', 'N', '4', '%', 'P', 'H', ']', '[', '(']

OHE2_smiles = []

for smile in smalldf.smiles.values[0:2]:
  matrix = np.zeros((264,34)) #34 is the number of unique characters, 254 is the number of the max length of the SMILES string
  for count,char in enumerate(smile):
    for count2,key in enumerate(keys):
      if key == char:
        matrix[count][count2] = 1
  OHE2_smiles.append(matrix.T)

input_shape = (264, 34)

x_val = np.array(OHE2_smiles)
x_val = torch.from_numpy(x_val).float()
x_val = x_val.to(device)

predicted =   model2(x_val)

predicted = predicted.cpu()

predictions_label = [1 if predicted[i] >= 0.5 else 0 for i in range(len(predicted))]
print(predictions_label) #Los datos obtenidos son iguales a los realizados con el modelo original
print("Los valores obtenidos deberían ser [0,1]")