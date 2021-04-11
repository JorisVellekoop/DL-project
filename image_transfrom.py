import torchvision
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
from scipy.stats import entropy

data_dir = 'CamVidData'

train_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

images, labels = next(iter(train_loader))

# import cv2 as cv
# image = image.permute(1, 2, 0).numpy()
# cv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)


def LocalContrastNorm(image, radius=9):
    """
    image: torch.Tensor , .shape => (1,channels,height,width)

    radius: Gaussian filter size (int), odd
    """
    if radius % 2 == 0:
        radius += 1

    def get_gaussian_filter(kernel_shape):
        x = np.zeros(kernel_shape, dtype='float64')

        def gauss(x, y, sigma=2.0):
            Z = 2 * np.pi * sigma ** 2
            return 1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

        mid = np.floor(kernel_shape[-1] / 2.)
        for kernel_idx in range(0, kernel_shape[1]):
            for i in range(0, kernel_shape[2]):
                for j in range(0, kernel_shape[3]):
                    x[0, kernel_idx, i, j] = gauss(i - mid, j - mid)

        return x / np.sum(x)

    n, c, h, w = image.shape[0], image.shape[1], image.shape[2], image.shape[3]

    gaussian_filter = torch.Tensor(get_gaussian_filter((1, c, radius, radius)))
    filtered_out = torch.nn.functional.conv2d(image, gaussian_filter, padding=radius - 1)
    mid = int(np.floor(gaussian_filter.shape[2] / 2.))
    ### Subtractive Normalization
    centered_image = image - filtered_out[:, :, mid:-mid, mid:-mid]

    ## Variance Calc
    sum_sqr_image = torch.nn.functional.conv2d(centered_image.pow(2), gaussian_filter, padding=radius - 1)
    s_deviation = sum_sqr_image[:, :, mid:-mid, mid:-mid].sqrt()
    per_img_mean = s_deviation.mean()

    ## Divisive Normalization
    divisor = np.maximum(per_img_mean.numpy(), s_deviation.numpy())
    divisor = np.maximum(divisor, 1e-4)
    new_image = centered_image / torch.Tensor(divisor)
    return new_image


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
    # if max = 0, sat = 0

    return 60*hue, sat, max.values


def get_invariant_angle(image):
    """
    image: numpy.array , .shape => (height,width, channels)

    """
    img = cv2.GaussianBlur(np.float32(image), (5,5), 0)

    r, g, b = cv2.split(img)

    im_mean = gmean(img, axis=2)

    mean_r = np.ma.divide(1.*r, im_mean)
    mean_g = np.ma.divide(1.*g, im_mean)
    mean_b = np.ma.divide(1.*b, im_mean)

    log_r = np.ma.log(mean_r)
    log_g = np.ma.log(mean_g)
    log_b = np.ma.log(mean_b)

    rho = cv2.merge((log_r, log_g, log_b))

    u = 1./np.sqrt(3)*np.array([[1,1,1]]).T
    I = np.eye(3)

    tol = 1e-15

    P_u_norm = I - u.dot(u.T)
    U_, s, V_ = np.linalg.svd(P_u_norm, full_matrices = False)

    s[ np.where( s <= tol ) ] = 0.

    U = np.dot(np.eye(3)*np.sqrt(s), V_)
    U = U[ ~np.all( U == 0, axis = 1) ].T

    U = U[::-1,:]
    U[:,1] *= -1.

    chi = rho.dot(U)

    e = np.array([[np.cos(np.radians(np.linspace(1, 180, 180))),
                   np.sin(np.radians(np.linspace(1, 180, 180)))]])

    gs = chi.dot(e)

    prob = np.array([np.histogram(gs[...,i], bins='scott', density=True)[0]
                      for i in range(np.size(gs, axis=3))])

    eta = np.array([entropy(p, base=2) for p in prob])

    theta_min = np.argmin(eta)

    return theta_min

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

plt.imshow(images[0].permute((1,2,0)))
plt.show()

angle = get_invariant_angle(images[0].permute(1,2,0))
image1 = Transform_Alvarez(images[0], theta=angle)
plt.imshow(image1, cmap='gray')
plt.show()



# image_test = LocalContrastNorm(images)
# ret = image_test[0].numpy().transpose((1,2,0))
# scaled_ret = (ret - ret.min())/(ret.max() - ret.min())
# image_maddern = Transfrom_Maddern(images[0])
# # image_alvarez = Transform_Alvarez(images[0])
#
# plt.imshow(scaled_ret)
# plt.show()
#
# plt.imshow(images[0].permute((1,2,0)))
# plt.show()
#
# plt.imshow(image_maddern, cmap='gray')
# plt.show()
# # plt.imshow(image_alvarez, cmap='gray')
# # plt.show()


