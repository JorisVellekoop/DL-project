#%matplotlib inline
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
#import helper
import Segnet

data_dir = '../CamVidData'

transform = transforms.Compose([transforms.ToTensor()])

X_train = datasets.ImageFolder(data_dir+"/train",transform=transform)
X_val = datasets.ImageFolder(data_dir+"/val",transform=transform)
X_test = datasets.ImageFolder(data_dir+"/test",transform=transform)
print(X_train.shape)

X_train_label = datasets.ImageFolder(data_dir+"/train_labels",transform=transform)
X_val_label = datasets.ImageFolder(data_dir+"/val_labels",transform=transform)
X_test_label = datasets.ImageFolder(data_dir+"/test_labels",transform=transform)

dataloader_train = torch.utils.data.DataLoader(X_train, batch_size=12, shuffle=False)
dataloader_val = torch.utils.data.DataLoader(X_val, batch_size=12, shuffle=False)
dataloader_test = torch.utils.data.DataLoader(X_test, batch_size=12, shuffle=False)

dataloader_train_label = torch.utils.data.DataLoader(X_train_label, batch_size=12, shuffle=False)
dataloader_val_label = torch.utils.data.DataLoader(X_val_label, batch_size=12, shuffle=False)
dataloader_test_label = torch.utils.data.DataLoader(X_test_label, batch_size=12, shuffle=False)

net = Segnet.CNN(2, 2)

label_colours = torch.tensor([
    [0.5020,0.5020,0.5020], #Sky
    [0.5020,0.0000,0.0000], #Building
    [0.7529,0.7529,0.5020], #pole
    [0.5020,0.2510,0.5020], #Road
    [0.0000,0.0000,0.5020], #Pavement
    [0.5020,0.5020,0.0000], #Tree
    [0.7140,0.7140,0.5020], #Sign
    [0.2510,0.2510,0.5020], #Fence
    [0.2510,0.0000,0.5020], #Car
    [0.2510,0.2510,0.0000], #Pedestrian
    [0.0000,0.5020,0.7530], #Bicycle
    ])

label_names =  ('sky', 'building', 'pole', 'road', 'pavement', 'tree',
'sign', 'fence', 'car', 'pedestrian', 'bicycle')

label_one_hot = torch.tensor([[1,0,0,0,0,0,0,0,0,0,0], #Sky
                                [0,1,0,0,0,0,0,0,0,0,0], #Building
                                [0,0,1,0,0,0,0,0,0,0,0], #pole
                                [0,0,0,1,0,0,0,0,0,0,0], #Road
                                [0,0,0,0,1,0,0,0,0,0,0], #Pavement
                                [0,0,0,0,0,1,0,0,0,0,0], #Tree
                                [0,0,0,0,0,0,1,0,0,0,0], #Sign
                                [0,0,0,0,0,0,0,1,0,0,0], #Fence
                                [0,0,0,0,0,0,0,0,1,0,0], #Car
                                [0,0,0,0,0,0,0,0,0,1,0], #Pedestrian
                                [0,0,0,0,0,0,0,0,0,0,1] #Bicycle
                                ])


                              
images, labels = next(iter(dataloader_train))
images_labels, _ = next(iter(dataloader_train_label))

image_number = 0
plt.imshow(torch.transpose(images[image_number].T,0,1))
plt.title(f"Training example #{image_number}")
plt.axis('off')
plt.show()

plt.imshow(torch.transpose(images_labels[image_number].T,0,1))
plt.title(f"Training example #{image_number}")
plt.axis('off')
plt.show()

label_per_pixel = torch.zeros(images_labels.shape[0],images_labels.shape[2],images_labels.shape[3],len(label_names))
print(images.shape)

# for i in range(images_labels.shape[0]):
#     image = np.transpose(images_labels[i],(1,2,0))
#     for ix in range(image.shape[0]):
#         for iy in range(image.shape[1]):
#             rgb = image[ix,iy,:]
#             for z in range(len(label_names)):
#                 if torch.allclose(rgb,label_colours[z],atol=0.001):
#                     label_per_pixel[i,ix,iy,:] = label_one_hot[z]
#                     print(label_names[z])
#             #print(image[ix,iy,:])
# print(label_per_pixel)    

# print(images_labels.shape[0])
# print(np.transpose(images_labels[4],(1,2,0)))