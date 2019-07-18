import os
from os.path import join
import numpy as np
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import torch
import torchvision as tv

import base_model.dataloader as dataloader
import model
import util

CLASSES = dataloader.CLASSES


class CAM(object):
    def __init__(self):
        self.batch_size = 1
        self.parent_dir = join('..','hetero_model','runs','ckpt')

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load data & build model
        self.load_dataset()
        self.build_model()


    def load_dataset(self):
        t_input = tv.transforms.Compose([
            tv.transforms.Resize((224,224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                    std=(0.229, 0.224, 0.225)),
        ])
        self.test_dataset = dataloader.FolderDataset('test', t_input)


    def build_model(self):
        self.model = model.VGG_for_CAM(self.parent_dir, self.device)
        self.model = self.model.to(self.device)
        self.model.eval()


    def get_item(self, index):
        input, target = self.test_dataset[index]
        input, target = input.unsqueeze(0), target.unsqueeze(0)
        input, target = input.to(self.device), target.to(self.device)
        return input, target


    def torch2pil(self, input):
        unnorm = util.UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
        img = input.cpu().squeeze(0)
        img = unnorm(img)
        img = tv.transforms.ToPILImage()(img)
        return img
        

    def torch2classes(self, target):
        class_idxs = target[0].nonzero().squeeze(0).cpu()
        classes = [CLASSES[i] for i in class_idxs]
        return classes


    def activation(self, input, top_idx):
        # get the most likely prediction of the model
        score, mean, rho = self.model(input)
        topk = score.topk(5)
        topk_probs = topk[0].squeeze(0)
        topk_idxs = topk[1].squeeze(0)
        topk_clss = [CLASSES[i] for i in topk_idxs]

        # get the gradient of the output with respect to the parameters of the model
        score[:, topk_idxs[top_idx]].backward(retain_graph=True)

        # pull the gradients out of the model
        gradients = self.model.get_activations_gradient()
        gradients = gradients.cpu().detach().squeeze(0) # size = [512,14,14]

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[1, 2], keepdim=True)
        pooled_gradients = pooled_gradients.cpu().detach().squeeze(0) # size = [512,1,1]

        # get the activations of the last convolutional layer
        activations = self.model.get_activations()
        activations = activations.cpu().detach().squeeze(0) # size = [512,14,14]

        # weight the channels by corresponding gradient
        grad_cam = activations * pooled_gradients
        grad_cam = grad_cam.mean(dim=0)
        grad_cam = torch.max(grad_cam, torch.tensor(0.))
        grad_cam /= grad_cam.max()

        return grad_cam, topk_probs, topk_clss


    def get_values(self, index, top_idx):
        input, target = self.get_item(index)
        img = self.torch2pil(input)
        classes = self.torch2classes(target)
        
        grad_cam, topk_probs, topk_clss = self.activation(input, top_idx)
        heatmap = Image.fromarray(np.array(grad_cam*255).astype(np.uint8))
        heatmap = heatmap.resize(img.size, resample=Image.BILINEAR)

        prob = topk_probs[top_idx]
        cls = topk_clss[top_idx]
        answer = cls in classes
        return img, heatmap, prob, cls, answer 
        
        


