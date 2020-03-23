import matplotlib.pyplot as plt
import numpy as np
from lr_finder import LRFinder 
import torch



# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def lr_finder(model, optimizer, criterion, trainloader):
  lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
  lr_finder.range_test(trainloader, end_lr=100, num_iter=100, step_mode="exp")
  lr_finder.plot() #to plot the loss vs Learning Rate curve
  lr_finder.reset() # to reset the lr_finder

def OverallAcc(testloader, model):
  # dataiter = iter(testloader)
  # images, labels = dataiter.next()
  correct = 0
  total = 0

  with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()

  print('Accuracy of the network on the 10000 test images: %d %%' % (
      100 * correct / total))
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))
  with torch.no_grad():
      for data in test_loader:
          images, labels = data
          images = images.cuda()
          labels = labels.cuda()
          outputs = model(images)
          _, predicted = torch.max(outputs, 1)
          c = (predicted == labels).squeeze()
          for i in range(4):
              label = labels[i]
              class_correct[label] += c[i].item()
              class_total[label] += 1


  for i in range(10):
      print('Accuracy of %5s : %2d %%' % (
          classes[i], 100 * class_correct[i] / class_total[i]))
