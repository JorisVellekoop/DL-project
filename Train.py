# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:10:24 2021

@author: remco
    --gpu 1
"""
import os
from pathlib import Path
import pandas as pd
import cv2 
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img
import Segnet
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

#GENERAL SETTINGS
view = 0
batch_sz = 4
n_epochs = 3



data_path = Path("CamVidData")

classes = pd.read_csv(data_path / 'class_dict_2.csv', index_col=0)
n_classes = len(classes)

cls2rgb = {cl:list(classes.loc[cl,:]) for cl in classes.index}

img = cv2.imread(str(data_path) + '/train/0001TP_006690.png')
plt.imshow(img)

mask = cv2.imread(str(data_path) + '/train_labels/0001TP_006690_L.png')
mask = cv2.cvtColor((mask).astype(np.uint8), cv2.COLOR_BGR2RGB)

plt.imshow(mask)


def adjust_mask(mask, flat=False):
    semantic_map = []
    for colour in list(cls2rgb.values()):        
        equality = np.equal(mask, colour)# 256x256x3 with True or False
        class_map = np.all(equality, axis = -1)# 256x256 If all True, then True, else False
        semantic_map.append(class_map)# List of 256x256 arrays, map of True for a given found color at the pixel, and False otherwise.
    semantic_map = np.stack(semantic_map, axis=-1)# 256x256xn_classes True only at the found color, and all False otherwise.
   

    return np.float32(semantic_map)# convert to numbers

new_mask = adjust_mask(mask)

def load_CAMVID(data_type='train', enc='ohe', shape='normal'):
  img_path = str(data_path) + '/' + data_type + '/'
  labels_path = str(data_path) + '/' + data_type + '_labels/'
  # without adding target_size=(256,256) in load_img we get Out of mem: 421x960x720x32x4bytes is around 34GB!
  x = np.array([np.array(load_img(str(img_path) + file, target_size=(256,256)))*1./255 for file in sorted(os.listdir(img_path))])
  y = np.array([adjust_mask(np.array(load_img(str(labels_path) + file, target_size=(256,256)))) for file in sorted(os.listdir(labels_path))])

  return x, y

#Load in the data
x_train, y_train = load_CAMVID(data_type='train')
#x_test, y_test = load_CAMVID(data_type='test')# Don't load test for RAM consumption
x_val, y_val = load_CAMVID(data_type='val')

#Convert to Tensor
x_train = torch.Tensor(x_train).permute(0,3,1,2)
print(x_train.shape)
#x_test = torch.Tensor(x_test).permute(0,3,1,2)
x_val = torch.Tensor(x_val).permute(0,3,1,2)
y_train = torch.Tensor(y_train).permute(0,3,1,2)
#y_test = torch.Tensor(y_test).permute(0,3,1,2)
y_val = torch.Tensor(y_val).permute(0,3,1,2)

train_dataset = TensorDataset(x_train,y_train)
val_dataset = TensorDataset(x_val,y_val)
#test_dataset = TensorDataset(x_test,y_test)

dataloader_train = DataLoader(train_dataset, batch_size=2, shuffle=True)
dataloader_val = DataLoader(val_dataset, batch_size=2, shuffle=False)
#dataloader_test = DataLoader(test_dataset, batch_size=12, shuffle=False)


print("Begin trainen")

model = Segnet.CNN(3,n_classes)#.cuda(GPU_ID)

#class_weights = 1.0/train_dataset.get_class_probability()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5*10**(-4))

model.train()

for epoch in range(n_epochs):  # loop over the dataset multiple times
    for i, data in enumerate(dataloader_train, 0):
        # get the inputs; data is a list of [inputs, labels]
        loss = 0.0
        inputs, labels = data
    
        # zero the parameter gradients
        optimizer.zero_grad()
        predicted, softmaxed = model(inputs)
        for z in range(len(inputs)):
            loss += criterion(predicted[z], torch.max(labels[z],1)[1])
        loss.backward()
        optimizer.step()

print('Finished Training')

