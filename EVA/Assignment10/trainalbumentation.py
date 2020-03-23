from albumentations import Compose, Normalize, HorizontalFlip, CoarseDropout
from albumentations.pytorch import ToTensor
import numpy as np

class TrainAlbumentation():

  def __init__(self):
    self.train_transform = Compose([

      HorizontalFlip(),
      Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        ,),
        CoarseDropout(max_holes=1, max_height=6, max_width=6,
        fill_value=[0.485, 0.456, 0.406], p=0.5),
        ToTensor()])
    # self.test_transform = Compose([Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),),ToTensor()])

  def __call__(self,img):
    img = np.array(img)
    img = self.train_transform(image = img)['image']
    return img