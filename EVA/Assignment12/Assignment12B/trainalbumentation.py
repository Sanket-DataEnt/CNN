from albumentations import Compose, Normalize, HorizontalFlip, RandomCrop, PadIfNeeded, Cutout #,CoarseDropout
from albumentations.pytorch import ToTensor
import numpy as np

class TrainAlbumentation():

  def __init__(self):
    self.train_transform = Compose([
      HorizontalFlip(),
      Normalize(
        mean = [0.4802, 0.4481, 0.3975],
        std=[0.2302, 0.2265, 0.2262]
        ,),
        Cutout(num_holes=2, max_h_size=8, max_w_size=8, always_apply=False, p=0.5),
        # CoarseDropout(max_holes=1, max_height=8, max_width=8,
        # fill_value=[0.485, 0.456, 0.406], p=0.5),
        ToTensor()])

  
  def __call__(self,img):
    img = np.array(img)
    img = self.train_transform(image = img)['image']
    return img