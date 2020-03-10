
import torchvision.transforms as transforms

class Augmentation:
  
  def __init__(self):
    self.transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

  def getTransform(self):
    return self.transform