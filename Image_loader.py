# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 12:03:01 2021

@author: remco
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np
from torchvision import datasets, transforms

data_dir = 'CamVidData'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.ImageFolder(data_dir+"/train",transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=False, num_workers=2)

trainset_y =datasets.ImageFolder(data_dir+"/train_labels",transform=transform)
trainloader_y = torch.utils.data.DataLoader(trainset_y, batch_size=4,
                                          shuffle=False, num_workers=2)

testset = datasets.ImageFolder(data_dir+"/test",transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('sky', 'building', 'pole', 'road', 'pavement', 'tree',
'sign', 'fence', 'car', 'pedestrian', 'bicycle')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

dataiter2 = iter(trainloader_y)
imagesy, labelsy = dataiter2.next()

print(images.shape)
print(imagesy.shape)

# show images
imshow(torchvision.utils.make_grid(images))
imshow(torchvision.utils.make_grid(imagesy))
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))