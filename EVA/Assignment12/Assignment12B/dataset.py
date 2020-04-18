import os
import torch
import torchvision
from trainalbumentation import TrainAlbumentation
from testalbumentation import TestAlbumentation



SEED = 1

class Data():

  def __init__(self):
    self.train_album = TrainAlbumentation()
    self.test_album = TestAlbumentation()

  def getTrainDataSet(self):
    datadir = os.getcwd()+'/MergeData/Train'
    # datadir = os.getcwd()+'/tiny-imagenet-200/train'
    dataset = torchvision.datasets.ImageFolder(root=datadir, transform=self.train_album)
    num_train = len(dataset)
    print("Train Data Size : ", num_train)
    return dataset

  def getTestDataSet(self):
    datadir = os.getcwd()+'/MergeData/Val'
    # datadir = os.getcwd()+'/tiny-imagenet-200/val'
    dataset = torchvision.datasets.ImageFolder(root=datadir, transform=self.test_album)
    num_train = len(dataset)
    print("Train Data Size : ", num_train)
    return dataset

  def get_tiny_imagenet_train_dataset(train_transforms, train_image_data, train_image_labels):
    from tinyimagenetdataset import TinyImagenetDataset
    return TinyImagenetDataset(image_data=train_image_data, image_labels=train_image_labels, transform=train_transforms)
     

  def get_tiny_imagenet_test_dataset(test_transforms, test_image_data, test_image_labels):
    from tinyimagenetdataset import TinyImagenetDataset
    return TinyImagenetDataset(image_data=test_image_data, image_labels=test_image_labels, transform=test_transforms)
    

  def getDataLoader(self, dataset, batches):
    # checking CUDA
    self.cuda = torch.cuda.is_available()
    # For reproducibility
    torch.manual_seed(SEED)
    if self.cuda:
      torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size = batches, num_workers=4, pin_memory=True) if self.cuda else dict(shuffle=True, batch_size=64)

    # train dataloader
    self.dataset_loader = torch.utils.data.DataLoader(dataset, **dataloader_args)

    return self.dataset_loader

  def getGradCamDataLoader(self, dataset):
  # checking CUDA
    self.cuda = torch.cuda.is_available()
    # For reproducibility
    torch.manual_seed(SEED)
    if self.cuda:
      torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=1, num_workers=4, pin_memory=True) if self.cuda else dict(shuffle=True, batch_size=1)

    # train dataloader
    self.dataset_loader = torch.utils.data.DataLoader(dataset, **dataloader_args)

    return self.dataset_loader

