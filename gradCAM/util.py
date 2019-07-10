import numpy as np
from PIL import Image
import torchvision as tv

import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from base_model.dataloader import CLASSES


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

    
def torch2pil(input):
    '''
    Args:
        input (torch.cuda.tensor): normalized image of size (1, C, H, W)
    Returns:
        img (PIL.Image): size (W,H)
    '''
    unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    img = input.cpu().squeeze(0)
    img = unnorm(img)
    img = tv.transforms.ToPILImage()(img)
    return img


def torch2classes(target):
    '''
    Args:
        target (torch.cuda.tensor): target of size (1, C)
    Returns:
        true_classes
    '''
    class_idxs = target[0].nonzero().squeeze(0).cpu()
    true_classes = [CLASSES[i] for i in class_idxs]
    return true_classes
    
    
def cam2heatmap(cam):
    '''
    Args:
        cam (torch.tensor): activation map with size (14,14)
    Returns:
        heatmap (PIL.Image): heatmap with size (224,224)
    '''
    cam /= cam.max()
    heatmap = Image.fromarray(np.array(cam*255).astype(np.uint8))
    heatmap = heatmap.resize((224,224), resample=Image.BILINEAR)
    return heatmap
