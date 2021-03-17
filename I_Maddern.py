import torchvision
import torch
import Color
import matplotlib.pyplot as plt

data_dir = 'CamVidData'

train_dataset = torchvision.datasets.ImageFolder(root=data_dir+'/train', transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

images, labels = next(iter(train_loader))

image = images[0]
a = 0.48
I_maddern = 0.5 + torch.log(image[1]) - a*torch.log(image[2]) - (1-a) * torch.log(image[0])

hue, sat, _ = Color.RGB2HSV(image)

plt.imshow(torch.transpose(I_maddern.T,0,1), cmap='gray')
plt.show()
plt.imshow(torch.transpose(hue.T,0,1), cmap='gray', vmin=0, vmax=360)
plt.show()
plt.imshow(torch.transpose(sat.T,0,1), cmap='gray')
plt.show()

import cv2 as cv

image = image.permute(1, 2, 0).numpy()

cv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)

plt.imshow(cv_image[:, :, 0], cmap='gray')
plt.show()
plt.imshow(cv_image[:,:, 1], cmap='gray')
plt.show()

