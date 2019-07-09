import os
from os.path import join
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import dataloader 
import model

CLASSES = dataloader.CLASSES


class CLS_Trainer(object):
    def __init__(self, args):
        super(CLS_Trainer, self).__init__()
        self.epochs = args.epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.phase = args.phase
        
        self.lowest_loss = np.inf
        
        # directory for saving model 
        self.parent_dir = join('runs')
        self.ckpt_dir = join(self.parent_dir, 'ckpt')
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load data & buil model
        self.load_dataset()
        self.build_model()

        
    def load_dataset(self):
        self.train_loader, self.test_loader = dataloader.voc_loader(self.batch_size)

        
    def build_model(self):
#         if self.phase == 'continue_train':
#             self.model = util.load_model(self.parent_dir, self.valid_fold)
#         else:
#             self.model = model.Attention()
        self.model = model.VGG()
        self.model = self.model.to(self.device)
        
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.lr, 
                                    betas=(0.9, 0.999), 
                                    weight_decay=1e-5)
        
        
    def train(self, epoch):
        self.model.train()
        for idx, (inputs, targets) in tqdm(enumerate(self.train_loader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

        print(f'===> Train Loss: {loss.cpu().detach():.4f}')


    def test(self, epoch):
        index_accuracy = 0.
        label_accuracy = 0.
        loss = 0.

        self.model.eval()
        for idx, (inputs, targets) in enumerate(self.test_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss += self.criterion(outputs, targets).detach()

            correct = outputs.ge(0).type(torch.FloatTensor).eq(targets.cpu()).type(torch.float32)
            index_accuracy += correct.mean(1).sum().item()
            label_accuracy += (correct.sum(1) == len(CLASSES)).sum().item()

        loss /= len(self.test_loader.dataset)
        index_accuracy /= len(self.test_loader.dataset)
        label_accuracy /= len(self.test_loader.dataset)

        print(f'===> Test Loss: {loss.cpu():.4f}, Index Accuracy: {index_accuracy:.4f}, Label Accuracy: {label_accuracy:.4f}')

        # saving model criteria : lowest test_loss 
        self.save_model(loss, label_accuracy, epoch)


    def save_model(self, loss, label_accuracy, epoch):
        if loss <= self.lowest_loss:
            print('===> Saving the model....')
            state = {
                'model': self.model.state_dict(),
                'accuracy': label_accuracy, 
                'epoch': epoch
            }
            torch.save(state, join(self.ckpt_dir, 'ckpt.pth'))
            self.lowest_loss = loss

