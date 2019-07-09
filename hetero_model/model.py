import os
from os.path import join

import torch
import torch.nn as nn
import torch.distributions as dist

from base_model.model import VGG

class VGG_hetero(nn.Module):
    def __init__(self, parent_dir, device):
        super(VGG_hetero, self).__init__()
        self.device = device
        
        state = torch.load(join(parent_dir,'ckpt.pth'))
        
        # get the pretrained VGG19 network
        self.model = VGG()
        self.model.load_state_dict(state['model'])
    
        # get the feature layers
        self.features_conv = self.model.features_conv
        self.features_linear = self.model.features_linear
    
        # delete self.model variable
        del self.model
    
        # change the output layer
        # first half for `mean`, second half for `rho`
        # where 'sigma' = log(1+exp('rho'))
        self.classifier = nn.Linear(4096, 40)
    
        # stochastic noise
        self.normal = dist.Normal(torch.zeros(20), torch.ones(20))
    
    def forward(self, x):
        x = self.features_conv(x)
        x = x.view((x.size(0), -1))
        x = self.features_linear(x)
        x = self.classifier(x)
        
        mean, rho = x[:,:20], x[:,20:]
        sample = self.normal.sample().to(self.device)
        x = mean + torch.log(1+torch.exp(rho)) * sample
        
        return x
