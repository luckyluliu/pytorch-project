# -*- coding: utf-8 -*-
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
import torch
import torch.nn as nn
import torchvision.models as models
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

class Model(nn.Module):
    def __init__(self, pretrained_weights_path, device):
        super(Model, self).__init__()
        self.model = models.inception_v3()
        self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, 6)
        self.model.fc = nn.Linear(self.model.fc.in_features, 6)
        self.model.load_state_dict(
            torch.load(pretrained_weights_path, map_location=device))
        del self.model._modules['AuxLogits'] #删除AuxLogits模块
        #self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, self.args.n_classes) #将模型AuxLogits模块的fc输出通道数改成我们需要的分类数
        #print(self.model) #打印模型结构
        #print(self.model._modules.keys())  #可以打印出模型的所有模块名称
        self.features1 = nn.Sequential(*list(self.model.children())[:7]) #去掉最后一层fc层，这句也可以写成# del self.model._modules['fc']
        self.features2 = nn.Sequential(*list(self.model.children())[7:-1])
        self.last_node_num = 2048
        del self.model    
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  #全局池化
        self.classifier = nn.Sequential(nn.Linear(self.last_node_num, 13), nn.LogSoftmax(dim=1))
        
    def forward(self, x):
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print('1:', meminfo.used)
        x = self.features1(x)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print('2:', meminfo.used)
        x = self.features2(x)
        x = self.avg_pool(x)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print('3:', meminfo.used)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.classifier(x)
        return x
