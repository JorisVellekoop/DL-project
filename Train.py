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
import argparse

#GENERAL SETTINGS
view = 0
batch_sz = 4
n_epochs = 100
steps_per_epoch = 1000
validation_steps = 100


# parser = argparse.ArgumentParser(description='Train a SegNet model')
# parser.add_argument('--gpu', type=int)
# args = parser.parse_args()
# CUDA = args.gpu is not None
# GPU_ID = args.gpu

data_path = Path("CamVidData")

print('Number of train frames: ' + str(len(os.listdir(data_path/'train'))))
print('Number of train labels: ' + str(len(os.listdir(data_path/'train_labels'))))
print('Number of val frames: ' + str(len(os.listdir(data_path/'val'))))
print('Number of val labels: ' + str(len(os.listdir(data_path/'val_labels'))))
print('Number of test frames: ' + str(len(os.listdir(data_path/'test'))))
print('Number of test labels: ' + str(len(os.listdir(data_path/'test_labels'))))
print('Total frames: ' + str(len(os.listdir(data_path/'train')) + len(os.listdir(data_path/'val')) + len(os.listdir(data_path/'test'))))

classes = pd.read_csv(data_path / 'class_dict.csv', index_col=0)
n_classes = len(classes)

cls2rgb = {cl:list(classes.loc[cl,:]) for cl in classes.index}

img = cv2.imread(str(data_path) + '/train/0001TP_006690.png')
plt.imshow(img)

mask = cv2.imread(str(data_path) + '/train_labels/0001TP_006690_L.png')
mask = cv2.cvtColor((mask).astype(np.uint8), cv2.COLOR_BGR2RGB)

plt.imshow(mask)

def get_class_probability(self):
    values = np.array(list(self.counts.values()))
    p_values = values/np.sum(values)

    return torch.Tensor(p_values)

def adjust_mask(mask, flat=False):
    semantic_map = []
    for colour in list(cls2rgb.values()):        
        equality = np.equal(mask, colour)# 256x256x3 with True or False
        class_map = np.all(equality, axis = -1)# 256x256 If all True, then True, else False
        semantic_map.append(class_map)# List of 256x256 arrays, map of True for a given found color at the pixel, and False otherwise.
    semantic_map = np.stack(semantic_map, axis=-1)# 256x256x32 True only at the found color, and all False otherwise.
    if flat:
      semantic_map = np.reshape(semantic_map, (-1,256*256))

    return np.float32(semantic_map)# convert to numbers

new_mask = adjust_mask(mask)

def load_CAMVID(data_type='train', enc='ohe', shape='normal'):
  img_path = str(data_path) + '/' + data_type + '/'
  labels_path = str(data_path) + '/' + data_type + '_labels/'
  # without adding target_size=(256,256) in load_img we get Out of mem: 421x960x720x32x4bytes is around 34GB!
  x = np.array([np.array(load_img(str(img_path) + file, target_size=(256,256)))*1./255 for file in sorted(os.listdir(img_path))])
  if(enc=='ohe'):
    
    y = np.array([adjust_mask(np.array(load_img(str(labels_path) + file, target_size=(256,256)))) for file in sorted(os.listdir(labels_path))])
  elif(enc=='sparse_cat'):
    y = np.array([adjust_mask(np.array(load_img(str(labels_path) + file, target_size=(256,256)))) for file in sorted(os.listdir(labels_path))])
  if(shape == 'flat'):
    y = np.reshape(y.shape[0], y.shape[1]*y.shape[2])
    y = np.expand_dims(y, axis=-1)
  return x, y

#Load in the data
x_train, y_train = load_CAMVID(data_type='train_small')
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
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.train()

for epoch in range(n_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    loss = 0.0
    for i, data in enumerate(dataloader_train, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
    
        print(inputs.shape)
        print(labels.shape)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        for z in range(len(inputs)):
            print(outputs[z].shape)
            print(labels[z].shape)
            loss += criterion(outputs[z], labels[z])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')