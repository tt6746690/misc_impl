import torch
import torch.nn as nn


class MnistCNN(nn.Module):
    
    def __init__(self, n_channels, n_classes, n_filters):
        super(MnistCNN, self).__init__()
        
        def block(in_channels, out_channels,
                  stride=2, padding=1,
                  nonlinearity=nn.ReLU()):
            return [nn.Conv2d(in_channels, out_channels,
                        kernel_size=4, stride=stride, padding=padding),
                    nonlinearity]
            
        self.conv_blocks = nn.Sequential(
            *block(n_channels,  n_filters),
            *block(n_filters,   2*n_filters),
            *block(2*n_filters, 4*n_filters))
        
        self.fc = nn.Linear(4*n_filters*16, n_classes)
        
    def forward(self, x):
        h = self.conv_blocks(x)
        h = torch.flatten(h, start_dim=1)
        s = self.fc(h)
        return s

