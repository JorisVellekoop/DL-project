# %matplotlib inline
import matplotlib.pyplot as plt
import torch
import numpy
from torchvision import datasets, transforms
#import helper
import random

data_dir = 'CamVidData'

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

X_train = datasets.ImageFolder(data_dir+"/train", transform=transform)
X_val = datasets.ImageFolder(data_dir+"/val", transform=transform)
X_test = datasets.ImageFolder(data_dir+"/test", transform=transform)

dataloader_train = torch.utils.data.DataLoader(X_train, batch_size=32, shuffle=False)
dataloader_val = torch.utils.data.DataLoader(X_val, batch_size=32, shuffle=False)
dataloader_test = torch.utils.data.DataLoader(X_test, batch_size=32, shuffle=False)

images, labels = next(iter(dataloader_train))
#helper.imshow(images[0])

random_image = random.randint(0, len(images))
plt.imshow(torch.transpose(images[10].T,0,1))
plt.title(f"Training example #{random_image}")
plt.axis('off')
plt.show()
