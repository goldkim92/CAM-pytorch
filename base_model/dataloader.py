import os
from os.path import join
from collections import defaultdict
from PIL import Image

import torch
import torchvision as tv
from torch.utils.data import Dataset, DataLoader

ROOT = join('/','data2','VOC2007')

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
           'bus', 'car', 'cat', 'chair', 'cow', 
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

CLASSES_TO_IDX = {CLASSES[i]:i for i in range(len(CLASSES))}

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', 
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_labels(obj='aeroplane', phase='train'):
    ''' 
    Returns: 
        dictionary: ex) {'000012.jpg':6}  
    '''
    root = join(ROOT, phase)
    if phase == 'train':
        path = join(root,'ImageSets','Main',obj+'_trainval.txt')
    elif phase == 'test':
        path = join(root,'ImageSets','Main',obj+'_test.txt')
    else:
        raise Exception('phase should be in ["train","test"]')

    labels = {}
    with open(path) as file:
        for line in file:
            row = line.split('\n')[0].split()
            if int(row[1])==1:
                labels[row[0]+'.jpg'] = CLASSES_TO_IDX[obj]
            
    return labels


def make_dataset(phase='train'):
    '''
    Return:
        list of tuples: ex) [('000012.jpg': [2,5]), (), ...]
    '''
    dataset = defaultdict(list)
    for dic in [make_labels(obj,phase) for obj in CLASSES]:
        for k,v in dic.items():
            dataset[k].append(v)
    
    return list(dataset.items())


def idx2logit(indices):
    logit = torch.zeros(len(CLASSES))
    logit[indices] = 1
    return logit


class FolderDataset(Dataset):
    def __init__(self, phase, t_input):
        self.phase = phase
        self.root = join(ROOT, self.phase)
        self.parent_dir = join(self.root, 'JPEGImages')
        self.t_input = t_input
        
        self.dataset = make_dataset(phase)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        path = join(self.parent_dir, data[0])
        
        input = Image.open(path).convert('RGB')
        input = self.t_input(input)
        
        target = idx2logit(data[1])
        
        return input, target
    
    def __len__(self):
        return len(self.dataset)


def voc_loader(bs=32):
    train_t_input = tv.transforms.Compose([
        tv.transforms.Resize((256,256)),
        tv.transforms.RandomHorizontalFlip(0.5),
        tv.transforms.RandomCrop((224,224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
    ])

    test_t_input = tv.transforms.Compose([
        tv.transforms.Resize((224,224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
    ])

    voc_train_dataset = FolderDataset('train', train_t_input)
    voc_test_dataset = FolderDataset('test', test_t_input)
    
    train_loader = DataLoader(voc_train_dataset, batch_size=bs, 
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(voc_test_dataset, batch_size=bs, 
                             shuffle=False, num_workers=2)
    
    return train_loader, test_loader


