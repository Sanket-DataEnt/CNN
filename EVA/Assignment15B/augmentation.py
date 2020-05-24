#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 23:43:31 2020

@author: sanket
"""


import torchvision.transforms as transforms

class Augmentation:
  
  def __init__(self):
      self.depth_transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),])
      self.bg_transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),])
      self.OverlayedImages_transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),])
      self.OverlayedMasks_transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),])
    
  def getDepthTransform(self):
      return self.depth_transform
    
  def getBgTransform(self):
      return self.bg_transform
  
  def getOverlayedImages(self):
      return self.OverlayedImages_transform
  
  def getOverlayedMasks(self):
      return self.OverlayedMasks_transform