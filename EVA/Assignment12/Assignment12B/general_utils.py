import matplotlib.pyplot as plt
import numpy as np
# from lr_finder import LRFinder 
import torch
from os import path
from tqdm import tqdm
import requests
from zipfile import ZipFile
import os.path
from os import path



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

# Function to plot Accuracy and Loss
def plot_acc_loss(train_acc, test_acc, trainloss_, testloss_):
  fig, axs = plt.subplots(2,2,figsize=(10,10))
  axs[0,0].plot(train_acc)
  axs[0,0].set_title("Training Accuracy")
  axs[0,0].set_xlabel("Batch")
  axs[0,0].set_ylabel("Accuracy")
  axs[0,1].plot(test_acc) 
  axs[0,1].set_title("Test Accuracy")
  axs[0,1].set_xlabel("Batch")
  axs[0,1].set_ylabel("Accuracy")
  axs[1,0].plot(trainloss_)
  axs[1,0].set_title("Training Loss")
  axs[1,0].set_xlabel("Batch")
  axs[1,0].set_ylabel("Loss")
  axs[1,1].plot(testloss_) 
  axs[1,1].set_title("Test Loss")
  axs[1,1].set_xlabel("Batch")
  axs[1,1].set_ylabel("Loss")

def download_file(folder_path, url):

    # get the file name
    file_name = url.split("/")[-1]
    folder_path = folder_path + "/" + file_name

    if path.exists(folder_path):
        print('File: {} already downloaded.'.format(file_name))
        return folder_path

    # read 1024 bytes every time
    buffer_size = 1024
    # download the body of response by chunk, not immediately
    response = requests.get(url, stream=True)

    # get the total file size
    file_size = int(response.headers.get("Content-Length", 0))

    # progress bar, changing the unit to bytes instead of iteration (default by tqdm)
    progress = tqdm(response.iter_content(buffer_size), f"Downloading {folder_path}", total=file_size, unit="B",
                    unit_scale=True, unit_divisor=1024)
    with open(folder_path, "wb") as f:
        for data in progress:
            # write data read to the file
            f.write(data)
            # update the progress bar manually
            progress.update(len(data))

    return folder_path

def extract_zip_file(file_path, extract_path):

    file_name = file_path.split("/")[-1]
    file_name = file_name.split(".")[0]
    output_folder = extract_path + '/' + file_name

    if path.exists(output_folder):
        print('File: {} already extracted.'.format(file_name))
        return output_folder

    print('Extracting file from {} to {}'.format(file_path, extract_path))
    with ZipFile(file_path, 'r') as zipObj:
        # Extract all the contents of zip file in different directory
        zipObj.extractall(extract_path)

    print('File extraction completed.')
    return output_folder
