import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np

data_dir = 'CamVidData'

train_dataset = torchvision.datasets.ImageFolder(root=data_dir+'/train', transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

images, labels = next(iter(train_loader))

# import cv2 as cv
# image = image.permute(1, 2, 0).numpy()
# cv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)


def RGB2HSV(image):
    max = torch.max(image, dim=0)
    min = torch.min(image, dim=0)
    range = max.values-min.values

    hue, sat = torch.zeros(image[0].shape), torch.zeros(image[0].shape)

    idx = (max.indices == 0) & (range != 0)
    hue[idx] = (image[1][idx] - image[2][idx])/(range[idx]) % 6
    idx = (max.indices == 1) & (range != 0)
    hue[idx] = 2 + (image[2][idx] - image[0][idx])/(range[idx])
    idx = (max.indices == 2) & (range != 0)
    hue[idx] = 4 + (image[0][idx] - image[1][idx])/(range[idx])
    # If range = 0, hue = 0

    idx = max.values != 0
    sat[idx] = range[idx]/max.values[idx]

    return 60*hue, sat, max.values


def Transfrom_Maddern(image, a=48, HS=False):
    I_maddern = 0.5 + torch.log(image[1]) - a * torch.log(image[2]) - (1 - a) * torch.log(image[0])

    if HS:
        hue, sat, _ = RGB2HSV(image)
        I_maddern = torch.dstack((I_maddern, hue, sat))

    return I_maddern


def Transform_Alvarez(image, a=5000, theta=0.6545, HS=False):
    I_alvarez = np.cos(theta)*(a*((image[0]/image[2])**1/a-1)) + np.sin(theta)*(a*((image[1]/image[2])**1/a-1))

    if HS:
        hue, sat, _ = RGB2HSV(image)
        I_alvarez = torch.dstack((I_alvarez, hue, sat))

    return I_alvarez



image_maddern = Transfrom_Maddern(images[0])
image_alvarez = Transform_Alvarez(images[0])

plt.imshow(image_maddern, cmap='gray')
plt.show()
plt.imshow(image_alvarez, cmap='gray')
plt.show()


