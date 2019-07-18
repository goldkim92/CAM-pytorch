import torch
import torch.nn as nn
from torchvision.models import vgg19

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        
        # get the pretrained VGG19 network
        self.vgg = vgg19(pretrained=True)
        
        # get the feature layers
        self.features_conv = self.vgg.features
        self.features_linear = self.vgg.classifier[:-1]
        
        # delete self.vgg variable
        del self.vgg
        
        # change the output layer
        self.classifier = nn.Linear(4096, 20)
        
    def forward(self, x):
        x = self.features_conv(x)
        x = x.view((x.size(0), -1))
        x = self.features_linear(x)
        x = self.classifier(x)
        return x
