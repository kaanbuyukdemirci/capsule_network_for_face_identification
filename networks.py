import torch
import torch.nn as nn
from layers import *
from custom_layers import *

# NOTE:
# - ReconstructionNetwork, CapsuleNetwork are specific, they are designed for the given face dataset.
# - DNE: Does Not Exist
# - Masking does expand the computational graph, it is not detached.
# - n = batch size

class ReconstructionNetwork(nn.Module):
    def __init__(self, number_of_classes, device='cpu', dtype=torch.float32) -> None:
        super().__init__()
        self.number_of_classes = number_of_classes
        self.device = device
        self.dtype = dtype
        
        self.fc1 = nn.Linear(in_features=32, out_features=3*224*224)
        
        self.to(device)
    
    def forward(self, x):
        x = self.find_max(x)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x.view(-1, 3, 224, 224)
    
    def find_max(self, x):
        x = x.view(x.shape[0], self.number_of_classes, -1)
        magnitudes = torch.norm(x, dim=2)
        max_indexes = torch.argmax(magnitudes, dim=1)
        return x[torch.arange(x.shape[0]), max_indexes, :]

class CapsuleNetwork(nn.Module):
    def __init__(self, threshold=0.5, alpha=0.0005, lamb=0.5, m_minus=0.1, m_plus=0.9, number_of_classes=100, dtype=torch.float32, device='cpu') -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device
        
        # in shape: (n, 3, 224, 224)
        self.conv_1 = CustomConvLayer(size='small', start_channel=2, channel_expand_constant=2, dtype=dtype, device=device)
        # out shape: (n, start_channel*channel_expand_constant**3, 8, 8)
        
        # in shape: (n, input_channel, 8, 8)
        self.caps_2 = CustomCapsuleLayer(size='small', expansion_constant=2, shrinkage_constant=4, number_of_train_classes=number_of_classes, 
                                           input_channel=16, dtype=dtype, device=device)
        # out shape:
        # if size == 'small': (n, number_of_train_classes, 8*expansion_constant^2)
        # if size == 'medium': (n, number_of_train_classes, 8*expansion_constant^3)
        # if size == 'big': (n, number_of_train_classes, 8*expansion_constant^4)
        
        # in shape: (n, number_of_train_classes, dimension_of_capsule)
        self.mask_3 = MaskLayer(n_classes=number_of_classes, threshold=threshold, flatten=True)
        # out shape: (n, number_of_train_classes*dimension_of_capsule)
        
        # in shape: (n, number_of_train_classes*dimension_of_capsule)
        self.rec_net_4 = ReconstructionNetwork(number_of_classes, device, dtype)
        # out shape: (n, 3, 224, 224)
        
        # in shape_1 input_data: (n, 3, 224, 224)
        # in shape_2 labels: (n, number_of_train_classes)
        # in shape_3 reconstructions: (n, 3, 224, 224)
        # in shape_4 capsule_predictions: (n, number_of_train_classes, dimension_of_capsule)
        self.cost_5 = CapsuleNetworkCostLayer(alpha, lamb, m_minus, m_plus, device, dtype)
        # out shape: (n, number_of_train_classes)
        
        self.to(self.device)
    
    def forward(self, x, y=None):
        x = self.conv_1(x)
        x = self.caps_2(x)
        reconstructions = self.mask_3(x, y)
        reconstructions = self.rec_net_4(reconstructions)
        return x, reconstructions

    def cost(self, input_data, labels, reconstructions, capsule_predictions):
        return self.cost_5(input_data, labels, reconstructions, capsule_predictions)

class Evaluator(object):
    def __init__(self) -> None:
        pass

    def accuracy(self, capsule_network_outputs, labels):
        """ round the capsule sizes to 0 or 1. For a sample to be considered correct, 
        it should rounded prediction should match to the labels.
        """
        # capsule_network_outputs shape: (n, n_capsules, n_capsule_features)
        # labels shape: (n or DNE, n_classes)
        # n_classes = n_capsules
        
        labels = labels.view(-1, labels.shape[-1])
        # labels shape: (n, n_classes)
        
        capsule_network_outputs = torch.norm(capsule_network_outputs, dim=2)
        # capsule_network_outputs shape: (n, n_classes)
        
        accuracy = torch.round(labels-capsule_network_outputs)
        # accuracy shape: (n, n_classes)
        
        accuracy = torch.any(accuracy, dim=1)
        # accuracy shape: (n)
        
        accuracy = 1 - torch.sum(accuracy)/accuracy.shape[0]
        # accuracy shape: (1,)
        
        return accuracy.item()
    
    def one_digit_accuracy(self, capsule_network_outputs, labels):
        """ instead of rounding every capsule, it is now assumed that there is only 1 digit, 
        so we return the onehot representation of the capsule with highest norm. 
        """
        # capsule_network_outputs shape: (n, n_capsules, n_capsule_features)
        # labels shape: (n or DNE, n_classes)
        # n_classes = n_capsules
        
        labels = labels.view(-1, labels.shape[-1])
        # labels shape: (n, n_classes)
        
        capsule_network_outputs = torch.norm(capsule_network_outputs, dim=2)
        # capsule_network_outputs shape: (n, n_classes)
        
        capsule_network_outputs = torch.argmax(capsule_network_outputs, dim=1)
        # capsule_network_outputs shape: (n)
        
        labels = torch.argmax(labels, dim=1)
        # labels shape: (n)
        
        return torch.sum(labels==capsule_network_outputs).item()/labels.shape[0]
