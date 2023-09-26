from layers import *

import torch
import torch.nn as nn

from typing import Literal

class CustomConvLayer(nn.Module):
    def __init__(self, size:Literal['small', 'medium', 'big'], start_channel:int=4, channel_expand_constant:int=4, 
                 dtype=torch.float32, device="cuda:0" if torch.cuda.is_available() else "cpu") -> None:
        super().__init__()
        
        self.size = size
        self.dtype = dtype
        self.device = device
        
        self.start_channel = start_channel
        self.channel_expand_constant = channel_expand_constant
        
        # input.shape = (3, 224, 224)
        if size == 'small': # 4 layers
            self.seq_0 = nn.Sequential(nn.Conv2d(in_channels=3, 
                                                 out_channels=self.start_channel, 
                                                 kernel_size=7, stride=2, padding=3),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**0),
                                       nn.ReLU(),)
            # output.shape = (self.start_channel*self.channel_expand_constant**0, 112, 112)
            self.seq_1 = nn.Sequential(nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**0, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**1, 
                                                 kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**1),
                                       nn.ReLU(),)
            # output.shape = (self.start_channel*self.channel_expand_constant**1, 56, 56)
            self.seq_2 = nn.Sequential(nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**1, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**2, 
                                                 kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**2),
                                       nn.ReLU(),)
            # output.shape = (self.start_channel*self.channel_expand_constant**2, 28, 28)
            self.seq_3 = nn.Sequential(nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**2, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**3, 
                                                 kernel_size=13, stride=1),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**3),
                                       nn.ReLU(),)
            # output.shape = (self.start_channel*self.channel_expand_constant**3, 16, 16)
        elif size == 'medium': # 8 layers
            self.seq_0 = nn.Sequential(nn.Conv2d(in_channels=3, 
                                                 out_channels=self.start_channel, 
                                                 kernel_size=7, stride=1, padding=3),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**0),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**0, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**0, 
                                                 kernel_size=7, stride=2, padding=3),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**0),
                                       nn.ReLU(),)
            # output.shape = (self.start_channel*self.channel_expand_constant**0, 112, 112)
            self.seq_1 = nn.Sequential(nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**0, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**1, 
                                                 kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**1),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**1, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**1, 
                                                 kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**1),
                                       nn.ReLU(),)
            # output.shape = (self.start_channel*self.channel_expand_constant**1, 56, 56)
            self.seq_2 = nn.Sequential(nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**1, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**2, 
                                                 kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**2),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**2, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**2, 
                                                 kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**2),
                                       nn.ReLU(),)
            # output.shape = (self.start_channel*self.channel_expand_constant**2, 28, 28)
            self.seq_3 = nn.Sequential(nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**2, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**3, 
                                                 kernel_size=13, stride=1, padding=6),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**3),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**3, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**3, 
                                                 kernel_size=13, stride=1),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**3),
                                       nn.ReLU(),)
            # output.shape = (self.start_channel*self.channel_expand_constant**3, 16, 16)     
        elif size == 'big': # 12 layers
            self.seq_0 = nn.Sequential(nn.Conv2d(in_channels=3, 
                                                 out_channels=self.start_channel, 
                                                 kernel_size=7, stride=1, padding=3),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**0),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**0, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**0, 
                                                 kernel_size=7, stride=1, padding=3),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**0),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**0, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**0, 
                                                 kernel_size=7, stride=2, padding=3),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**0),
                                       nn.ReLU(),)
            # output.shape = (self.start_channel*self.channel_expand_constant**0, 112, 112)
            self.seq_1 = nn.Sequential(nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**0, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**1, 
                                                 kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**1),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**1, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**1, 
                                                 kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**1),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**1, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**1, 
                                                 kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**1),
                                       nn.ReLU(),)
            # output.shape = (self.start_channel*self.channel_expand_constant**1, 56, 56)
            self.seq_2 = nn.Sequential(nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**1, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**2, 
                                                 kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**2),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**2, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**2, 
                                                 kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**2),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**2, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**2, 
                                                 kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**2),
                                       nn.ReLU(),)
            # output.shape = (self.start_channel*self.channel_expand_constant**2, 28, 28)
            self.seq_3 = nn.Sequential(nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**2, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**3, 
                                                 kernel_size=13, stride=1, padding=6),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**3),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**3, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**3, 
                                                 kernel_size=13, stride=1, padding=6),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**3),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=self.start_channel*self.channel_expand_constant**3, 
                                                 out_channels=self.start_channel*self.channel_expand_constant**3, 
                                                 kernel_size=13, stride=1,),
                                       nn.BatchNorm2d(self.start_channel*self.channel_expand_constant**3),
                                       nn.ReLU(),)
            # output.shape = (self.start_channel*self.channel_expand_constant**3, 16, 16)
        else:
            raise ValueError('Undefined size for ConvLayers.')

        self.output_channels = self.start_channel*self.channel_expand_constant**3
        
        self.to(device)
        
    def forward(self, x):
        # x.shape = (batch_size, 3, 224, 224)
        x = self.seq_0(x)
        x = self.seq_1(x)
        x = self.seq_2(x)
        x = self.seq_3(x)
        # x.shape = (bs, self.start_channel*self.channel_expand_constant**3, 16, 16)
        x = nn.AvgPool2d(kernel_size=2, stride=2)(x)
        # x.shape = (bs, self.start_channel*self.channel_expand_constant**3, 8, 8)
        return x
    
class CustomCapsuleLayer(nn.Module):
    def __init__(self, size:Literal['small', 'medium', 'big'], expansion_constant:Literal[2,4,8],
                 shrinkage_constant:Literal[2,4,8], number_of_train_classes=100, input_channel = 256,
                 dtype=torch.float32, device="cuda:0" if torch.cuda.is_available() else "cpu") -> None:
        super().__init__()
        
        self.size = size
        self.expansion_constant = expansion_constant
        self.shrinkage_constant = shrinkage_constant
        self.dtype = dtype
        self.device = device
        self.number_of_train_classes = number_of_train_classes
        self.input_channel = input_channel
        
        # input.shape = (bs, input_channel, 8, 8)
        self.conv_to_cap_layer_0 = ConvToCapLayer(n_in_capsules_unique=8*8,
                                                  n_in_capsules_sharing_per_unique=int(input_channel/8), 
                                                  n_in_features=8, 
                                                  device=device, dtype=dtype)
        # output.shape = (bs, 8*8, input_channel/8, 8, 1) # (bs, unique, share per unique, feature, 1)
        
        if (self.size == 'small') or (self.size == 'medium') or (self.size == 'big'):
            cap_1 = CapsuleLayer(n_out_capsules=int(self.conv_to_cap_layer_0.n_in_capsules/self.shrinkage_constant), # self.conv_to_cap_layer_0.n_in_capsules/shrinkage_constant
                                n_in_capsules_unique=self.conv_to_cap_layer_0.n_in_capsules_unique, # self.conv_to_cap_layer_0.n_in_capsules_unique
                                n_in_capsules_sharing_per_unique=self.conv_to_cap_layer_0.n_in_capsules_sharing_per_unique, # self.conv_to_cap_layer_0.n_in_capsules_sharing_per_unique
                                n_out_features=self.conv_to_cap_layer_0.n_in_features*self.expansion_constant,
                                n_in_features=self.conv_to_cap_layer_0.n_in_features, 
                                device=device, dtype=dtype)
            # output.shape = (self.conv_to_cap_layer_0.n_in_capsules/shrinkage_constant, self.conv_to_cap_layer_0.n_in_features*self.expansion_constant)
        if (self.size == 'medium') or (self.size == 'big'):
            cap_2 = CapsuleLayer(n_out_capsules=int(cap_1.n_out_capsules/self.shrinkage_constant), # self.conv_to_cap_layer_0.n_in_capsules/shrinkage_constant^2
                                n_in_capsules_unique=cap_1.n_out_capsules, # self.conv_to_cap_layer_0.n_in_capsules/shrinkage_constant
                                n_in_capsules_sharing_per_unique=1, 
                                n_out_features=self.expansion_constant*cap_1.n_out_features, # 8*expansion_constant^2
                                n_in_features=cap_1.n_out_features, # self.conv_to_cap_layer_0.n_in_features*expansion_constant
                                device=device, dtype=dtype)
            # output.shape = (self.conv_to_cap_layer_0.n_in_capsules/shrinkage_constant^2, self.conv_to_cap_layer_0.n_in_features*expansion_constant^2)
        if (self.size == 'big'):
            cap_3 = CapsuleLayer(n_out_capsules=int(cap_2.n_out_capsules/self.shrinkage_constant), # self.conv_to_cap_layer_0.n_in_capsules/shrinkage_constant^3
                                n_in_capsules_unique=cap_2.n_out_capsules, # self.conv_to_cap_layer_0.n_in_capsules/shrinkage_constant^2
                                n_in_capsules_sharing_per_unique=1, 
                                n_out_features=self.expansion_constant*cap_2.n_out_features, # self.conv_to_cap_layer_0.n_in_features*expansion_constant^3
                                n_in_features=cap_2.n_out_features, # self.conv_to_cap_layer_0.n_in_features*expansion_constant^2
                                device=device, dtype=dtype)
            # output.shape = (self.conv_to_cap_layer_0.n_in_capsules/shrinkage_constant^3, self.conv_to_cap_layer_0.n_in_features*expansion_constant^3)
        
        if size == 'small':
            self.sequence_of_capsule_layers_1 = nn.Sequential(cap_1)
            self.face_capsules_2 = CapsuleLayer(n_out_capsules=number_of_train_classes,
                                                n_in_capsules_unique=cap_1.n_out_capsules,
                                                n_in_capsules_sharing_per_unique=1,
                                                n_out_features=self.expansion_constant*cap_1.n_out_features,
                                                n_in_features=cap_1.n_out_features,
                                                device=device, dtype=dtype)
            self.parameter_count = cap_1.weights.shape.numel()
            self.parameter_count += self.face_capsules_2.weights.shape.numel()
            # condition: self.conv_to_cap_layer_0.n_in_capsules / shrinkage_constant >= 1
            # output.shape = (number_of_train_classes, 8*expansion_constant^2)
        elif size == 'medium':
            self.sequence_of_capsule_layers_1 = nn.Sequential(cap_1, cap_2)
            self.face_capsules_2 = CapsuleLayer(n_out_capsules=number_of_train_classes,
                                                n_in_capsules_unique=cap_2.n_out_capsules,
                                                n_in_capsules_sharing_per_unique=1,
                                                n_out_features=self.expansion_constant*cap_2.n_out_features,
                                                n_in_features=cap_2.n_out_features,
                                                device=device, dtype=dtype)
            self.parameter_count = cap_1.weights.shape.numel() + cap_2.weights.shape.numel()
            self.parameter_count += self.face_capsules_2.weights.shape.numel()
            # condition: self.conv_to_cap_layer_0.n_in_capsules / shrinkage_constant^2 >= 1
            # output.shape = (number_of_train_classes, 8*expansion_constant^3)
        elif size == 'big':
            self.sequence_of_capsule_layers_1 = nn.Sequential(cap_1, cap_2, cap_3)
            self.face_capsules_2 = CapsuleLayer(n_out_capsules=number_of_train_classes,
                                                n_in_capsules_unique=cap_3.n_out_capsules,
                                                n_in_capsules_sharing_per_unique=1,
                                                n_out_features=self.expansion_constant*cap_3.n_out_features,
                                                n_in_features=cap_3.n_out_features,
                                                device=device, dtype=dtype)
            self.parameter_count = cap_1.weights.shape.numel() + cap_2.weights.shape.numel() * cap_3.weights.shape.numel()
            self.parameter_count += self.face_capsules_2.weights.shape.numel()
            # condition: self.conv_to_cap_layer_0.n_in_capsules / shrinkage_constant^3 >= 1
            # output.shape = (number_of_train_classes, 8*expansion_constant^4)
        self.to(device)
    
    def forward(self, x):
        # 'small' condition: x.shape = 8*input_channel / shrinkage_constant >= 1
        # 'medium' condition: x.shape = 8*input_channel / shrinkage_constant^2 >= 1
        # 'big' condition: x.shape = 8*input_channel / shrinkage_constant^3 >= 1
        
        # x.shape = (bs, input_channel, 8, 8)
        x = self.conv_to_cap_layer_0(x)
        # x.shape = (bs, 64, input_channel/8, 8, 1)
        x = self.sequence_of_capsule_layers_1(x)
        # if size == 'small': x.shape = (8*input_channel/shrinkage_constant, self.conv_to_cap_layer_0.n_in_features*self.expansion_constant)
        # if size == 'medium': x.shape = (8*input_channel/shrinkage_constant^2, self.conv_to_cap_layer_0.n_in_features*self.expansion_constant^2)
        # if size == 'big': x.shape = (8*input_channel/shrinkage_constant^3, self.conv_to_cap_layer_0.n_in_features*self.expansion_constant^3)
        x = self.face_capsules_2(x)
        # if size == 'small': x.shape = (bs, number_of_train_classes, 8*expansion_constant^2)
        # if size == 'medium': x.shape = (bs, number_of_train_classes, 8*expansion_constant^3)
        # if size == 'big': x.shape = (bs, number_of_train_classes, 8*expansion_constant^4)
        return x
