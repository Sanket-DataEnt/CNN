#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 23:54:06 2020

@author: sanket
"""


import torch
import torchvision
from augmentation import Augmentation
SEED = 1

### Defining dataset
from pathlib import Path
from PIL import Image


class DSet():

  def __init__(self, data_root, train_transform=True):
    augumentation_obj = Augmentation()
    self.depth_transform = augumentation_obj.getDepthTransform()
    self.bg_transform = augumentation_obj.getBgTransform()
    self.OverlayedImages_transform = augumentation_obj.getOverlayedImages()
    self.OverlayedMasks_transform = augumentation_obj.getOverlayedMasks()
    f1 = Path(data_root+'Background/')
    self.f1_files = list(sorted(f1.glob('*.jpg')))

    self.f2_files_ = []
    for i in range(0,40):
      f2 = Path(data_root+'OverlayedImages/'+str(i))
      f2_files = list(sorted(f2.glob('*.jpg')))
      self.f2_files_.extend(f2_files)

    self.f3_files_ = []
    for i in range(0,40):
      f3 = Path(data_root+'OverlayedMasks/'+str(i))
      f3_files = list(sorted(f3.glob('*.jpg')))
      self.f3_files_.extend(f3_files)

    self.f4_files_ = []
    for i in range(0,40):
      f4 = Path(data_root+'DepthImage/'+str(i))
      f4_files = list(sorted(f4.glob('*.jpg')))
      self.f4_files_.extend(f4_files)

    self.train_transform = train_transform
    self.name = {}
    for i in self.f1_files:
      num = str(i).split('.')[0].split('_')[1]
      self.name[num] = i
  
  #Extracting the number of the background_foreground, so that the same can be called for background
  def get_f1_image(self, index):
    a = str(self.f2_files_[index]).split('_')[-3][1:]
    return self.name[a]

  def __len__(self):
    return len(list(self.f2_files_))

  def __getitem__(self, index):
    f1_image_ = self.get_f1_image(index) #calling the background based on the index of the bg_fg
    f1_image = Image.open(f1_image_)
    f1_image = f1_image.convert(mode='RGB')
    f2_image = Image.open(self.f2_files_[index])
    f2_image = f2_image.convert(mode='RGB')
    f3_image = Image.open(self.f3_files_[index])
    f3_image = f3_image.convert(mode='L')
    f4_image = Image.open(self.f4_files_[index])
    f4_image = f4_image.convert(mode='L')
    

    if self.train_transform:
      f1_image = self.bg_transform(f1_image)
      f2_image = self.OverlayedImages_transform(f2_image)
      f3_image = self.OverlayedMasks_transform(f3_image)
      f4_image = self.depth_transform(f4_image)

    return {'f1': f1_image, 'f2' : f2_image, 'f3' : f3_image, 'f4' : f4_image} 