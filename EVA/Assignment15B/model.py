#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 00:23:12 2020

@author: sanket
"""


import torch.nn as nn
import torch.nn.functional as F
import torch


# New Architecture Upsampling and Downsampling in Depth Arch. (Date = 23/05/20)(IOU Mask = 0.80, IOU Depth = 0.55):-

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Input Block 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) 
        # will Concate the inputs
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) 
        #self.pool1 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        #self.pool2 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        #Mask Branch
        self.convblock1m = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False, groups=128),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.convblock2m = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.convblock3m = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(3, 3), padding=1, bias=False)
        )
        #Depth Branch
        self.convblock1d = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False, groups=128),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.convblock2d = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False, groups=128),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.convblock3d = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.convblock4d = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, bias=False, groups=256),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )        
        self.convblock5d = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.convblock6d = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.convblock7d = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.convblock8d = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1, bias=False, groups=128),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.convblock9d = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(3, 3), padding=1, bias=False)
        )

    def forward(self, sample):
        # For analyzing summary
        # f1_ = sample#['f1'].cuda()
        # f2_ = sample#['f2'].cuda()
        #For training the model
        f1_ = sample['f1'].cuda()
        f2_ = sample['f2'].cuda()

        #Input Block
        f1 = self.convblock1(f1_) #32
        f2 = self.convblock1(f2_) #32

        #Concatenating the input 
        f = torch.cat([f1, f2], dim=1) #64

       # f = self.pool1(f)

        f = self.convblock3(f)

        #f = self.pool2(f)

        f = self.convblock4(f)

        # Mask Branch

        f_mask = self.convblock3m(self.convblock2m(self.convblock1m(f)))

        # Depth Branch

        f_depth = self.convblock9d(self.convblock8d(self.convblock7d(self.convblock6d(self.convblock5d(self.convblock4d(self.convblock3d(self.convblock2d(self.convblock1d(f))))))))
)
        return f_mask, f_depth