#!/usr/bin/env python
from __future__ import print_function, division

import torch, shutil
import torch.nn as nn
import  torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
#from sklearn.metrics import roc_auc_score
from mymodel import Model
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from plotval import *
import matplotlib.pyplot as plt
import pynvml

class EyeData(Dataset):
    def __init__(self, root_dir, list_file, transform=None):
        with open(os.path.join(root_dir, list_file)) as fo:
            self.p_infos = fo.readlines()
            self.num_sample = len(self.p_infos)
        self.transform = transform

    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        p_info = self.p_infos[idx]
        p_info = p_info.split('    ')
        img = Image.open(p_info[0])
        #label = np.array([int(p_info[1])])
        label = int(p_info[1])
        if self.transform:
            img = self.transform(img)
        sample = {'image':img, 'label':label}
        return sample

def val_model(model, criterion):
    model.eval()
    i=0
    for sample in val_loader:
        inputs = sample['image']
        labels = sample['label']
        inputs = inputs.to(device)
        labels = labels.to(device, dtype=torch.int64)
        # zero the parameter gradients
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            if i==0:
                labels_val = labels
                preds_val = preds
            else:
                labels_val = torch.cat((labels_val, labels))
                preds_val = torch.cat((preds_val, preds))
            i+=1
    labels_val = labels_val.detach().cpu().numpy()
    preds_val = preds_val.detach().cpu().numpy()
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(labels_val[:,np.newaxis], preds_val[:,np.newaxis], classes=class_names, savefile='val_nonormal.jpg',
                          title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    plot_confusion_matrix(labels_val[:,np.newaxis], preds_val[:,np.newaxis], classes=class_names, savefile='val_normal.jpg', normalize=True,
                          title='Normalized confusion matrix')

if __name__ == '__main__':
    data_dir = '../fsl/dataset'
    best_weights_path = 'mybest.pt'
    trsm = transforms.Compose([
        transforms.Resize(342),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    traindata = EyeData(data_dir, 'train.txt', transform = trsm)
    train_loader = DataLoader(traindata, batch_size=8, shuffle=True, num_workers=4)
    valdata = EyeData(data_dir, 'test.txt', transform = trsm)
    val_loader = DataLoader(valdata, batch_size=8, shuffle=True, num_workers=4)
    
    train_datasize = len(traindata)
    val_datasize = len(valdata)
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    #model = Model(pretrained_weights_path, device)
    
    # define model
    model = models.inception_v3(pretrained=True)
    # modify auxlogit layer
    num_aux_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_aux_ftrs, 13)
    # modify fc layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 13)
    # load 6-classes weights
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model = model.to(device)
    # training configuration
    criterion = nn.CrossEntropyLoss().cuda(device=device)
    val_model(model, criterion)
