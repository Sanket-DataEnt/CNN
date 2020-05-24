#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 00:37:05 2020

@author: sanket
"""


from tqdm import tqdm
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision
from pathlib import Path


PATH = './saved_models1/'


# Train to print Ground Truth Image
import time
import numpy as np


def calculate_iou(target, prediction, thresh):
        intersection = np.logical_and(np.greater(target,thresh), np.greater(prediction,thresh))
        union = np.logical_or(np.greater(target,thresh), np.greater(prediction,thresh))
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

def show(tensors):
    
    grid_tensor = torchvision.utils.make_grid(tensors)
    grid_image = grid_tensor.permute(1,2,0)
    plt.figure(figsize=(20,20))
    plt.imshow(grid_image)
    plt.show()
    plt.close()
  
def save_plot(tensors, name):
    grid_tensor = torchvision.utils.make_grid(tensors)
    grid_image = grid_tensor.permute(1,2,0)
    plt.figure(figsize=(20,20))
    plt.imshow(grid_image)
    plt.savefig(name, bbox_inches = 'tight')
    plt.close()
  
def train(model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    #pbar = tqdm(train_loader)
    # log_interval = 300
    # globaliter = 0
    for batch_idx, data, in enumerate(train_loader):
      # globaliter += 1
      # d_s_time = time.time()
      data["f1"] = data["f1"].to(device)
      data["f2"] = data["f2"].to(device)
      data["f3"] = data["f3"].to(device)
      data["f4"] = data["f4"].to(device)
      # d_e_time = time.time()
      
    
    
      optimizer.zero_grad()
      output_mask, output_depth = model(data)
    
      # x = torch.cat([data["f1"],data["f2"]], dim=1)
      # f3_out, f4_out = model(x)
      l1 = criterion(output_mask, data["f3"])
      l2 = criterion(output_depth, data["f4"])
      loss = l1 + 2*l2
      #pbar.set_description(desc = f'loss={loss.item()} l1={l1.item()} l2={l2.item()}')
      loss.backward()
      optimizer.step()
    
      #Saving the images
      if batch_idx % 1000 == 0:
        #Overlayed
        sample = data["f2"][0:8,:,:,:]
        save_plot(sample.detach().cpu(), f"plots/{epoch}_Overlayed_{batch_idx}.jpg")
        #Mask
        sample = data["f3"][0:8,:,:,:]
        save_plot(sample.detach().cpu(), f"plots/{epoch}_ActualMask_{batch_idx}.jpg")
        #Predicted Mask
        output_ = output_mask[0:8,:,:,:]
        save_plot(output_.detach().cpu(), f"plots/{epoch}_PredMask_{batch_idx}.jpg")
        #Depth
        sample = data["f4"][0:8,:,:,:]
        save_plot(sample.detach().cpu(), f"plots/{epoch}_ActualDepth_{batch_idx}.jpg")
        #Predicted Depth
        output = output_depth[0:8,:,:,:]
        save_plot(output.detach().cpu(), f"plots/{epoch}+PredDepth_{batch_idx}.jpg")
    
      #printing batch after every 10 epochs
      if batch_idx % 10 == 0:
        print("Batch ID: ", batch_idx)
    
      if batch_idx % 1000 == 0:
        torch.save(model.state_dict(),PATH/f"{epoch}_{batch_idx}_{loss.item()}.pth")
    
      if batch_idx % 100 == 0:
        start_time = time.time()
        #print('start_time:', start_time)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx/ len(train_loader), loss.item()))
        print("Loss Mask: ", l1.item())
        print("Loss Depth: ", l2.item())
    
        iou_depth = calculate_iou(output_depth.detach().cpu().numpy(), data['f4'].detach().cpu().numpy(), 0.5)
        iou_mask = calculate_iou(output_mask.detach().cpu().numpy(), data['f3'].detach().cpu().numpy(), 0.5)
        print('IOU Depth: ', iou_depth)
        print('IOU Mask: ', iou_mask)
        print('Batch ID:', batch_idx)
        end_time = time.time()
        print('time took for 100 batches:', end_time-start_time)
    
    
      if batch_idx % 1000 == 0:
        print("Ground Truth of Depth:")
        show(data['f4'][0:8,:,:,:].detach().cpu())
        show(output_depth[0:8,:,:,:].detach().cpu())
        print("Ground Truth of Mask:")
        show(data['f3'][0:8,:,:,:].detach().cpu())
        show(output_mask[0:8,:,:,:].detach().cpu())
    
      # if batch_idx % log_interval == 0:
      #   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
      #             epoch, batch_idx * len(data), len(train_loader.dataset),
      #             100. * batch_idx / len(train_loader), loss.item()))
      #   iou = calculate_iou(output.detach().cpu().numpy(), data['f4'].detach().cpu().numpy())
      #   print('IOU: ', iou)
      #   print('Batch ID:', batch_idx)
    
        # with train_summary_writer.as_default():
        #     summary.scalar('loss', loss.item(), step=globaliter)
        
import tensorflow as tf
import time
epoch = 1
def test(model, criterion, device, test_loader,optimizer, epoch):
    model.eval()
    loss1 = []
    pbar = tqdm(iter(test_loader), dynamic_ncols=True)
    for batch_idx, data in enumerate(pbar):
        if(batch_idx==0):
            t0 = time.time()
        data["f1"] = data["f1"].to(device)
        data["f2"] = data["f2"].to(device)
        data["f3"] = data["f3"].to(device)
        data["f4"] = data["f4"].to(device)
        optimizer.zero_grad()
        output_mask, output_depth = model(data)
        loss1 = criterion(output_mask, data["f3"])
        loss2 = criterion(output_depth, data["f4"])
        loss = loss1 + 2*loss2
        if(batch_idx != 0 and batch_idx%100==0):
            torch.cuda.empty_cache()
            iou3 = calculate_iou(output_mask.detach().cpu().numpy(), data["f3"].detach().cpu().numpy(),0.5)
            iou4 = calculate_iou(output_depth.detach().cpu().numpy(), data["f4"].detach().cpu().numpy(),0.5)
            #       with train_summary_writer.as_default(): 
                #           tf.summary.scalar('iouf3', iou3, step=batch_idx)
                #           tf.summary.scalar('iouf4', iou4, step=batch_idx)
                #           tf.summary.scalar('lossf3', loss1.item(), step=batch_idx)
                #           tf.summary.scalar('lossf4', loss2.item(), step=batch_idx)
            t3 = time.time()
            print(t3-t0 )
            t0=t3
            print("Batch:" + str(batch_idx), " Epoch:"+str(epoch), " lOSSf3="+str(loss1.item()), " lOSSf4="+str(loss2.item()), 'iouf3', iou3, 'iouf4', iou4)
            for param_group in optimizer.param_groups:
                print("lr= ",param_group['lr'])
                if(batch_idx%100==0):
                    sample0 = data["f2"]
                    sample1 = data["f3"]
                    sample2 = data["f4"]
                    show(sample0)
                    show(sample1)
                    show(sample2)
                    show(output_mask)
                    show(output_depth)
