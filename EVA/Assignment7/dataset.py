import torch
import torchvision
from augmentation import Augmentation
SEED = 1

class Data:

  def __init__(self):
    augumentation_obj1 = Augmentation()
    self.transform = augumentation_obj1.getTransform()

  def getDataSet(self, train=False):
    self.dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=self.transform)
    return self.dataset

  def getDataLoader(self, dataset):
    # checking CUDA
    self.cuda = torch.cuda.is_available()
    # For reproducibility
    torch.manual_seed(SEED)
    if self.cuda:
      torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if self.cuda else dict(shuffle=True, batch_size=64)

    # train dataloader
    self.dataset_loader = torch.utils.data.DataLoader(dataset, **dataloader_args)

    return self.dataset_loader

