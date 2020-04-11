from albumentations import Compose, Normalize, HorizontalFlip, RandomCrop, PadIfNeeded, Cutout #,CoarseDropout
from albumentations.pytorch import ToTensor
import numpy as np

class TrainAlbumentation():

  def __init__(self):
    self.train_transform = Compose([
      PadIfNeeded(min_height = 40, min_width = 40,border_mode=4, value=None, p=1),
      RandomCrop(32,32),
      HorizontalFlip(),
      Normalize(
        mean = [0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        ,),
        Cutout(num_holes=1, max_h_size=8, max_w_size=8, always_apply=False, p=0.5),
        # CoarseDropout(max_holes=1, max_height=8, max_width=8,
        # fill_value=[0.485, 0.456, 0.406], p=0.5),
        ToTensor()])

  # def __init__(self):
  #   self.train_transform = Compose([

  #     HorizontalFlip(),
  #     Normalize(
  #       mean=[0.485, 0.456, 0.406],
  #       std=[0.229, 0.224, 0.225]
  #       ,),
  #       CoarseDropout(max_holes=1, max_height=6, max_width=6,
  #       fill_value=[0.485, 0.456, 0.406], p=0.5),
  #       ToTensor()])
    # self.test_transform = Compose([Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),),ToTensor()])

  def __call__(self,img):
    img = np.array(img)
    img = self.train_transform(image = img)['image']
    return img