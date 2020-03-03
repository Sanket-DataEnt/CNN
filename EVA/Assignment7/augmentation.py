
import torchvision.transforms as transforms

class Augmentation:
  
  def __init__(self):
    self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  def getTransform(self):
    return self.transform