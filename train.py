# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
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
import matplotlib.pyplot as plt
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(3)

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

best_acc = 0.0
def val_model(model, criterion, optimizer):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for sample in val_loader:
        inputs = sample['image']
        labels = sample['label']
        inputs = inputs.to(device)
        labels = labels.to(device, dtype=torch.int64)
        # zero the parameter gradients
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / val_datasize
    epoch_acc = running_corrects.double() / val_datasize
    val_losses.append(epoch_loss)
    val_acces.append(epoch_acc)
    print('Val Loss: {:.4f}'.format(epoch_loss))
    print('Val Acc: {:.4f}'.format(epoch_acc))
    global best_acc
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        print('saving model with best_acc: {}'.format(best_acc))
        torch.save(model.state_dict(), best_weights_path)
        
def train_model(model, criterion, optimizer, scheduler, num_epochs=200):
    scheduler.step()
    model.train()   
    running_loss = 0.0
    running_corrects = 0
    for sample in train_loader:
        inputs = sample['image']
        labels = sample['label']
        inputs = inputs.to(device)
        labels = labels.to(device, dtype=torch.int64)

        # zero the parameter gradients
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)            
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / train_datasize
    epoch_acc = running_corrects.double() / train_datasize
    losses.append(epoch_loss)
    acces.append(epoch_acc)
    print('Train Loss: {:.4f}'.format(epoch_loss)) 
    print('Train Acc: {:.4f}'.format(epoch_acc))

if __name__ == '__main__':
    data_dir = '../fsl/dataset'
    best_weights_path = 'c13-fundus-weights.best.pt'
    pretrained_weights_path = '../inceptionv3_NEWFUNDUS_1006.pt'
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
    model.AuxLogits.fc = nn.Linear(num_aux_ftrs, 6)
    # modify fc layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)
    # load 6-classes weights
    model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
    
    # add my fc layer
    model.AuxLogits.fc = nn.Linear(num_aux_ftrs, len(class_names))
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.aux_logits = False
    #model = nn.DataParallel(model, device_ids=[0,1])
    
    model = model.to(device)
    # training configuration
    criterion = nn.CrossEntropyLoss().cuda(device=device)
    # Observe that all parameters are being optimized
    #optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    losses = []
    acces = []
    val_losses = []
    val_acces = []
    
    # train model
    num_epochs = 30
    for epoch in range(num_epochs):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_model(model, criterion, optimizer, exp_lr_scheduler)
        val_model(model, criterion, optimizer)
        end = time.time()
        print('time cost for this epoch: {}'.format(end-start))
    
    plt.subplot(121)
    plt.plot(range(num_epochs), losses, 'g', label='train loss')
    plt.plot(range(num_epochs), val_losses, 'r', label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.subplot(122)
    plt.plot(range(num_epochs), acces, 'g', label='train acc')
    plt.plot(range(num_epochs), val_acces, 'r', label='val acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.savefig('result.jpg')
    plt.show()

# %%

# %%
