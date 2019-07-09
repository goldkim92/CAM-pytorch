import os
from os.path import join
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import torch
import torch.nn as nn
import torch.distributions as dist

from hetero_model.model import VGG_hetero


class VGG_for_CAM(nn.Module):
    def __init__(self, parent_dir, device):
        super(VGG_for_CAM, self).__init__()
        self.device = device
        
        state = torch.load(join(parent_dir,'ckpt.pth'))
        
        # get the pretrained VGG19 network
        load_model_dir = join('..','base_model','runs','ckpt')
        self.model = VGG_hetero(load_model_dir, self.device)
        self.model.load_state_dict(state['model'])
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.model.features_conv[:-1]
        
        # get the max pool of the features stem
        self.max_pool = self.model.features_conv[-1]
        
        # get the classifier of the vgg19
        self.features_linear = self.model.features_linear
        self.classifier = self.model.classifier

        # stochastic noise
        self.normal = dist.Normal(torch.zeros(20), torch.ones(20))
        
        # delete self.model variable
        del self.model
        
        # placeholder for the gradients and feature_conv
        self.gradients = None
        self.features = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        self.features = self.features_conv(x)
        
        # register the hook
        h = self.features.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(self.features)
        x = x.view((x.size(0), -1))
        x = self.features_linear(x)
        x = self.classifier(x)

        mean, rho = x[:,:20], x[:,20:]
        sample = self.normal.sample().to(self.device)
        x = mean + torch.log(1+torch.exp(rho)) * sample

        return x, mean, rho
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self):
        return self.features

