import torch

def RGB2HSV(image):
    max = torch.max(image, dim=0)
    min = torch.min(image, dim=0)
    range = max.values-min.values

    # print(range[range==0])

    # argmax = torch.argmax(image, dim=0)

    hue, sat = torch.zeros(image[0].shape), torch.zeros(image[0].shape)

    idx = max.indices == 0
    # print(image[1][idx] - image[2][idx])
    hue[idx] += (image[1][idx] - image[2][idx])/(range[idx]) % 6
    idx = max.indices == 1
    hue[idx] += 2 + (image[2][idx] - image[0][idx])/(range[idx])
    idx = max.indices == 2
    hue[idx] += 4 + (image[0][idx] - image[1][idx])/(range[idx])

    sat = range/max.values

    return 60*hue, sat, max