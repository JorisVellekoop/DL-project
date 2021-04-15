# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:55:26 2021

@author: remco
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(CNN, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.VGG16 = models.vgg16(pretrained=True)
        
        
        ###Encoder
        #First layer
        self.encoder_conv_11    = nn.Conv2d(input_channels,64,kernel_size=3,padding=1)
        self.encoder_bn_11      = nn.BatchNorm2d(64)
        self.encoder_conv_12    = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.encoder_bn_12      = nn.BatchNorm2d(64)
        
        #Second layer
        self.encoder_conv_21    = nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.encoder_bn_21      = nn.BatchNorm2d(128)
        self.encoder_conv_22    = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.encoder_bn_22      = nn.BatchNorm2d(128) 
        
        #Third layer
        self.encoder_conv_31     = nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.encoder_bn_31      = nn.BatchNorm2d(256)
        self.encoder_conv_32    = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.encoder_bn_32      = nn.BatchNorm2d(256) 
        self.encoder_conv_33    = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.encoder_bn_33      = nn.BatchNorm2d(256) 
        
        #Fourth layer
        self.encoder_conv_41    = nn.Conv2d(256,512,kernel_size=3,padding=1)
        self.encoder_bn_41      = nn.BatchNorm2d(512)
        self.encoder_conv_42    = nn.Conv2d(512,5126,kernel_size=3,padding=1)
        self.encoder_bn_42      = nn.BatchNorm2d(512) 
        self.encoder_conv_43    = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.encoder_bn_43      = nn.BatchNorm2d(512) 
        
        #Fift layer
        self.encoder_conv_51    = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.encoder_bn_51      = nn.BatchNorm2d(512)
        self.encoder_conv_52    = nn.Conv2d(512,5126,kernel_size=3,padding=1)
        self.encoder_bn_52      = nn.BatchNorm2d(512) 
        self.encoder_conv_53    = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.encoder_bn_53      = nn.BatchNorm2d(512) 
        
        self.set_encoder_params()
        
        ###Decoder
        #Fift Layer
        self.decoder_conv_53  = nn.ConvTranspose2d(512,512,kernel_size=3,padding=1)
        self.decoder_bn_53      = nn.BatchNorm2d(512)
        self.decoder_conv_52  = nn.ConvTranspose2d(512,512,kernel_size=3,padding=1)
        self.decoder_bn_52      = nn.BatchNorm2d(512)
        self.decoder_conv_51  = nn.ConvTranspose2d(512,512,kernel_size=3,padding=1)
        self.decoder_bn_51      = nn.BatchNorm2d(512)
        
        #Fourd Layer
        self.decoder_conv_43  = nn.ConvTranspose2d(512,512,kernel_size=3,padding=1)
        self.decoder_bn_43      = nn.BatchNorm2d(512)
        self.decoder_conv_42  = nn.ConvTranspose2d(512,512,kernel_size=3,padding=1)
        self.decoder_bn_42      = nn.BatchNorm2d(512)
        self.decoder_conv_41  = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=3,padding=1)
        self.decoder_bn_41      = nn.BatchNorm2d(256)
        
        #Third Layer
        self.decoder_conv_33  = nn.ConvTranspose2d(256,256,kernel_size=3,padding=1)
        self.decoder_bn_33      = nn.BatchNorm2d(256)
        self.decoder_conv_32  = nn.ConvTranspose2d(256,256,kernel_size=3,padding=1)
        self.decoder_bn_32      = nn.BatchNorm2d(256)
        self.decoder_conv_31  = nn.ConvTranspose2d(256,128,kernel_size=3,padding=1)
        self.decoder_bn_31      = nn.BatchNorm2d(128)
        
        #Second Layer
        self.decoder_conv_22  = nn.ConvTranspose2d(128,128,kernel_size=3,padding=1)
        self.decoder_bn_22      = nn.BatchNorm2d(128)
        self.decoder_conv_21  = nn.ConvTranspose2d(128,64,kernel_size=3,padding=1)
        self.decoder_bn_21      = nn.BatchNorm2d(64)
        
        #First Layer
        self.decoder_conv_12  = nn.ConvTranspose2d(64,64,kernel_size=3,padding=1)
        self.decoder_bn_12      = nn.BatchNorm2d(64)
        self.decoder_conv_11  = nn.ConvTranspose2d(64,output_channels,kernel_size=3,padding=1)
                
        
        
    def set_encoder_params(self):
        #First layer
        self.encoder_conv_11.weight.data = self.VGG16.features[0].weight.data
        self.encoder_conv_11.bias.data = self.VGG16.features[0].bias.data
        
        self.encoder_conv_12.weight.data = self.VGG16.features[2].weight.data
        self.encoder_conv_12.bias.data = self.VGG16.features[2].bias.data
        
        #Second Layer
        self.encoder_conv_21.weight.data = self.VGG16.features[5].weight.data
        self.encoder_conv_21.bias.data = self.VGG16.features[5].bias.data
        
        self.encoder_conv_22.weight.data = self.VGG16.features[7].weight.data
        self.encoder_conv_22.bias.data = self.VGG16.features[7].bias.data
                
        #Third layer
        self.encoder_conv_31.weight.data = self.VGG16.features[10].weight.data
        self.encoder_conv_31.bias.data = self.VGG16.features[10].bias.data
        
        self.encoder_conv_32.weight.data = self.VGG16.features[12].weight.data
        self.encoder_conv_32.bias.data = self.VGG16.features[12].bias.data
        
        self.encoder_conv_33.weight.data = self.VGG16.features[14].weight.data
        self.encoder_conv_33.bias.data = self.VGG16.features[14].bias.data
        
        #Fourth Layer
        self.encoder_conv_41.weight.data = self.VGG16.features[17].weight.data
        self.encoder_conv_41.bias.data = self.VGG16.features[17].bias.data
        
        self.encoder_conv_42.weight.data = self.VGG16.features[19].weight.data
        self.encoder_conv_42.bias.data = self.VGG16.features[19].bias.data
        
        self.encoder_conv_43.weight.data = self.VGG16.features[21].weight.data
        self.encoder_conv_43.bias.data = self.VGG16.features[21].bias.data
        
        #Fift layer
        self.encoder_conv_51.weight.data = self.VGG16.features[24].weight.data
        self.encoder_conv_51.bias.data = self.VGG16.features[24].bias.data
        
        self.encoder_conv_52.weight.data = self.VGG16.features[26].weight.data
        self.encoder_conv_52.bias.data = self.VGG16.features[26].bias.data
        
        self.encoder_conv_53.weight.data = self.VGG16.features[28].weight.data
        self.encoder_conv_53.bias.data = self.VGG16.features[28].bias.data
        
    def forward(self,input_image):
        #Encoder
        #First Layer
        x = F.relu(self.encoder_bn_11(self.encoder_conv_11(input_image)))
        x = F.relu(self.encoder_bn_12(self.encoder_conv_12(x)))
        x, idx1 = F.max_pool2d(x,kernel_size=2,stride=2,return_indices=True)
        
        #Second Layer
        x = F.relu(self.encoder_bn_21(self.encoder_conv_21(x)))
        x = F.relu(self.encoder_bn_22(self.encoder_conv_22(x)))
        x, idx2 = F.max_pool2d(x,kernel_size=2,stride=2,return_indices=True)
        
        #Third Layer
        x = F.relu(self.encoder_bn_31(self.encoder_conv_31(x)))
        x = F.relu(self.encoder_bn_32(self.encoder_conv_32(x)))
        x = F.relu(self.encoder_bn_33(self.encoder_conv_33(x)))
        x, idx3 = F.max_pool2d(x,kernel_size=2,stride=2,return_indices=True)
        
        #Fourth Layer
        x = F.relu(self.encoder_bn_41(self.encoder_conv_41(x)))
        x = F.relu(self.encoder_bn_42(self.encoder_conv_42(x)))
        x = F.relu(self.encoder_bn_43(self.encoder_conv_43(x)))
        x, idx4 = F.max_pool2d(x,kernel_size=2,stride=2,return_indices=True)
        
        #Fifth layer
        x = F.relu(self.encoder_bn_51(self.encoder_conv_51(x)))
        x = F.relu(self.encoder_bn_52(self.encoder_conv_52(x)))
        x = F.relu(self.encoder_bn_53(self.encoder_conv_53(x)))
        x, idx5 = F.max_pool2d(x,kernel_size=2,stride=2,return_indices=True)
        
        #Decoder
        #Fifth Layer
        x = F.max_unpool2d(x, idx5, kernel_size=2,stride=2)
        x = F.relu(self.decoder_bn_53(self.decoder_conv_53(x)))
        x = F.relu(self.decoder_bn_52(self.decoder_conv_52(x)))
        x = F.relu(self.decoder_bn_51(self.decoder_conv_51(x)))
        
        #Fourth Layer
        x = F.max_unpool2d(x, idx4, kernel_size=2,stride=2)
        x = F.relu(self.decoder_bn_43(self.decoder_conv_43(x)))
        x = F.relu(self.decoder_bn_42(self.decoder_conv_42(x)))
        x = F.relu(self.decoder_bn_41(self.decoder_conv_41(x)))
        
        #Third Layer
        x = F.max_unpool2d(x, idx3, kernel_size=2,stride=2)
        x = F.relu(self.decoder_bn_33(self.decoder_conv_33(x)))
        x = F.relu(self.decoder_bn_32(self.decoder_conv_32(x)))
        x = F.relu(self.decoder_bn_31(self.decoder_conv_31(x)))
                   
        #Second Layer
        x = F.max_unpool2d(x, idx2, kernel_size=2,stride=2)
        x = F.relu(self.decoder_bn_22(self.decoder_conv_22(x)))
        x = F.relu(self.decoder_bn_21(self.decoder_conv_21(x)))
        
        #First Layer
        x = F.max_unpool2d(x, idx1, kernel_size=2,stride=2)
        x = F.relu(self.decoder_bn_12(self.decoder_conv_12(x)))
        x = self.decoder_conv_11(x)
        
        x_softmax = F.softmax(x,dim=1)
        return x, x_softmax


