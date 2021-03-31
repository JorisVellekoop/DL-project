# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:46:21 2021

@author: remco
"""
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from Dataset import CamVid
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import Segnet
import torch

batch_size = 2

def make_mask(mask):
    semantic_map = []
    for colour in colours:  
       equality = np.equal(mask, colour[1])
       class_map = np.all(equality, axis = -1)
       semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    
    #print(semantic_map)
    return np.float32(semantic_map)


transform = transforms.Compose(
    [transforms.CenterCrop((360,480)),
     ])

target_transform = transforms.Compose(
    [transforms.CenterCrop((360,480)),
     ])
        
camvid_train = CamVid('CamVidData_2','train_small',transform=transform,target_transform=target_transform)
camvid_test = CamVid('CamVidData_2','test',transform=transform,target_transform=target_transform)


num_train_img = len(camvid_train.images)

train_dataset = np.arange(num_train_img)
dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    
    
    


a,b = camvid_train.__getitem__(0)


colours = []

for i in range(len(camvid_train.classes)):
    if camvid_train.classes[i][2] != 255:
        colours.append((camvid_train.classes[i][0],camvid_train.classes[i][7]))


for i in range(len(camvid_train.classes)):
    if camvid_train.classes[i][2] == 255:
        colours.append((camvid_train.classes[i][0],camvid_train.classes[i][7]))

n_classes = len(colours)


model = Segnet.CNN(3,n_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5*10**(-4))

model.train()


for i, img_num in enumerate(dataloader_train, 0):
    images = torch.zeros(batch_size,3,360,480)
    labels = torch.zeros(batch_size,n_classes,360,480)
    loss = 0.0
    for z in range(len(img_num)):
        image,label = camvid_train.__getitem__(img_num[z])
        mask = make_mask(label)
        image = torch.tensor(np.array(image))
        mask = torch.tensor(np.array(mask))
        images[z] = image.permute(2,0,1)
        labels[z] = mask.permute(2,0,1)
    print(images.shape)
    optimizer.zero_grad()
    predicted, softmaxed = model(images)
    for k in range(len(labels)):
             loss += criterion(predicted[k], torch.max(labels[k],1)[1])
    loss.backward()
    optimizer.step()
        
    
