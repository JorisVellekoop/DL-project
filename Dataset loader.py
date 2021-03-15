#%matplotlib inline
import matplotlib.pyplot as plt
import torch
import numpy
from torchvision import datasets, transforms
#import helper
import Segnet
import random
import numpy as np

data_dir = 'CamVidData'

transform = transforms.Compose([transforms.ToTensor()])

X_train = datasets.ImageFolder(data_dir+"/train",transform=transform)
X_val = datasets.ImageFolder(data_dir+"/val",transform=transform)
X_test = datasets.ImageFolder(data_dir+"/test",transform=transform)

X_train_label = datasets.ImageFolder(data_dir+"/train_labels",transform=transform)
X_val_label = datasets.ImageFolder(data_dir+"/val_labels",transform=transform)
X_test_label = datasets.ImageFolder(data_dir+"/test_labels",transform=transform)

dataloader_train = torch.utils.data.DataLoader(X_train, batch_size=12, shuffle=False)
dataloader_val = torch.utils.data.DataLoader(X_val, batch_size=12, shuffle=False)
dataloader_test = torch.utils.data.DataLoader(X_test, batch_size=12, shuffle=False)

dataloader_train_label = torch.utils.data.DataLoader(X_train_label, batch_size=12, shuffle=False)
dataloader_val_label = torch.utils.data.DataLoader(X_val_label, batch_size=12, shuffle=False)
dataloader_test_label = torch.utils.data.DataLoader(X_test_label, batch_size=12, shuffle=False)

net = Segnet.CNN(2,2)


images, labels = next(iter(dataloader_train))
images_labels, _ = next(iter(dataloader_train_label))

random_image = random.randint(0, len(images))
plt.imshow(torch.transpose(images[4].T,0,1))
plt.title(f"Training example #{random_image}")
plt.axis('off')
plt.show()

plt.imshow(torch.transpose(images_labels[4].T,0,1))
plt.title(f"Training example #{random_image}")
plt.axis('off')
plt.show()


for i in range(images_labels.shape[0]):
    image = np.transpose(images_labels[i],(1,2,0))
    for ix in range(image.shape[0]):
        for iy in range(image.shape[1]):
            rgb = image[ix,iy,:]
            if torch.allclose(rgb,torch.tensor([0.5020,0.0000,0.0000]),atol=0.001):
                print("Building")
            print(image[ix,iy,:])
    

print(images_labels.shape[0])
print(np.transpose(images_labels[4],(1,2,0)))