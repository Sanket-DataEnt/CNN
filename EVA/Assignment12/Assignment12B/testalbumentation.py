from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize
from albumentations.pytorch import ToTensor
import numpy as np

class TestAlbumentation():

  def __init__(self):
    self.test_transform = Compose([
      Normalize(
        mean=[0.4802, 0.4481, 0.3975],
        std=[0.2302, 0.2265, 0.2262],
        ),
        ToTensor()
        ])

  def __call__(self,img):
    img = np.array(img)
    img = self.test_transform(image = img)['image']
    return img