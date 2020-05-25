## Problem Statement :
Given an image with foreground objects and background image, predict the depth map as well as a mask for the foreground object.

### Data Description

There are four kinds of images in our dataset (fg, bg, fg_bg, masks, depth) :

1. Foreground Image (fg) :- Different Man, Woman, kids, group of person(for background transparency we have taken png images)
![FG](https://user-images.githubusercontent.com/25937235/82763067-6d6a9680-9e22-11ea-9325-d7175bfb90ef.jpg)

2. Background Image (bg) :- We restricted background to library images(for restricting size of image we have taken jpg images)
![BG](https://user-images.githubusercontent.com/25937235/82763062-604da780-9e22-11ea-8668-b92505f0276d.jpg)

3. Foreground+Background Image (fg_bg) :- Background superposed over Foreground (for restricting size of images we have taken jpg images)
![FG_BG](https://user-images.githubusercontent.com/25937235/82783841-f3b3c680-9e7c-11ea-91a7-89f0a06d9522.jpg)

4. Masks :- masks extracted from fg images(we have taken grayscale images)(.jpg)
![Mask](https://user-images.githubusercontent.com/25937235/82763069-70fe1d80-9e22-11ea-9871-393b0c494b55.jpg)

5. Depth Image :- We have extracted depth images from fg_bg using nyu model(for restricing size of images we have taken grayscale images extracted from colormap. One reason for taking grayscale image is to restrict memory usage. If we are getting all the information in 1 channel why to take three channels.)(.jpg)
![Depth](https://user-images.githubusercontent.com/25937235/82763438-67c28000-9e25-11ea-869c-6124601f3597.jpg)

Total images of each kind : 

  - fg :- 200(flip + no flip)

  - bg :- 100

  - fg_bg :- 400000

  - masks :- 400000

  - depth :- 400000

So, Total we have 12,001,00 images for this project.

### Input and Output

  - Input -> Background, Background+Foreground
  - Output -> Masks and Depths

### Data Loading

To load this huge amount of data is also a challenge if you do not have access to proper resources. As, I was working on Google Colab as it offers free GPU, following are the ideas which I applied to load the dataset : -
  - Loading data from Image Folders -> Initially, I thought this is the best idea, I read the 12,001,00 images into Google       Colab directly through folders on Google drive, where I stored the images. It took more then 40 minutes to load the data through these folders. Then I tried to train, it took around 5-6 hours in training just one epoch. So, now I got, definitely this is not the good idea. So, thought to explore second idea.
  - Unzipping directly into Colab memory -> Now, I initially zipped my all the images which were stored in the folders on the drive to zipped folders. It took around 3 days to Zip the complete 12 lakh images on Google Colab. One thing I learned during this, patience is the key to success. I used the following code to Zip the images:
  
  %%time
  
from zipfile import ZipFile

count = 0

with ZipFile('DepthImages_Sanket.zip', 'a') as myzip:

  for r,d, files in os.walk('/content/drive/My Drive/Assignment1/Output/DepthImage/15'):
  
    for f in files:
    
      x = r + '/' + f
      
      count = count + 1
      
      if(count%500==0):
      
        print(count, end=',')
        
      myzip.write(x, x.split('/')[-3] + '/' + x.split('/')[-2] + '/' + x.split('/')[-1])
      
Finally, after getting the zipped image, I directly unzipped the images into the internal memory of Colab instead of unzipping in the drive. This took around 10 minutes to read all the images. Some 6 corrupt images came, which were removed. So, to train my model I have total (399998x3 + 100) images. This was definitely very fast as of now, however to know what was its effect on training, whether it has decreased the training time for 1 epoch or not, remember patience is the key to success. :) Let's move ahead.

### Structure :-

To make my code more understandable, I tried to make my code in the modular format i,e it is something like a package. Just call the function from the file, if you don't want to go in the granularity of the code. Let's dive more into the modular structure of the code.

  - Augmentation -> This file contains the transformation strategy for the dataset. Since, I have more than enough data size, so I just resized the image to 64x64 and do not applied other transformations. Initially, my image were of the size 160x160 but to keep balance between constraint of
memory and training time in Colab, I choose to have image size of 64x64 with batch size of 32. The code for this is available in augmentation.py 

  - Dataset -> As name suggest, this class focusses only on the creation of the dataset. Actually, I have 40 folders each of FG+BG, Mask and Depth with 10,000 images each and BG has only 100 images. So, this class apply the tricky code which convert the unzipped images into RGB and Grayscale, so that they can be trained by the model. Furthermore, this class also contains the algorithm through which if you have the information of Depth or Mask we can directly search its Background image. If this excites you go and explore dataset.py. 
  
  - model -> This class contains the model which I have used to train the data. Before finalizing the model, I have tried various models some were actually giving very good results but because they were having approx 30Million to 40 Million parameters so, it was very difficult for me to finish training of the model on time without facing memory error in Colab. So, I explored and decided to use Upsampling and Downsampling model for Depth and Simple model architecture for Mask as I found detecting Mask was not very tedious job for model. The model architecture which I have used contains around total 3,631,616 parameters. I know you are now more interested in exploring the result and training time but have some patience. Let's first check the model architecture by exploring model.py file, which will help you in understanding the heart of this project and how it is working to predict mask and depth images in a single architecture.
  
  - training -> This file is very crucial, it explains how I am training the model. Initially, I was very happy by seeing the visuals of the images but later on when I tried different model architecture and loss functions it was very difficult for me to accept or reject by just checking the visulas.So, I want a mathematical function which should explain by numbers how good is my predicted image. Therefore, I choose Intersection Over Union (IOU) to test my results. If IOU for a depth or mask is close to 1 then predicted images are very good however, if it is not then we should have to focus on imrovising architecture or/and loss function. Apart from these, it also contains the functions for plotting and saving the images. To explore and learn more you can go through training.py file.  
  
### Different Loss Functions : - 

I have tried various Loss functions for this data, these are :

    1. SSIM Loss : -
      - It gave me following results 
      - Time took for 100 batches of size 32 = 55 seconds
      - Time took to complete 1 epoch = 88 minutes
      - Mask Loss = 0.10
      - Depth Loss = 0.23
      - IOU for Mask = 0.70
      - IOU for Depth Image = 0.18
      - Tried this loss for 5 epochs but result was not imroving much. Max Mask IOU = 0.76 and Max Depth IOU = 0.39
      
     2. L1 Smooth Loss :- 
      - It gave me following results
      - Time took to complete 100 batches of size 32 = 83 seconds
      - Time took to complete 1 epoch = 130 minutes
      - Mask Loss = 0.004
      - Depth Loss = 0.0130
      - IOU for Mask = 0.777
      - IOU for depth Image = 0.528
      - Although loss value is reduced very much but the respective improvement is not visible in the predicted iamges and IOUs.
      - Below is the Image got while training the model.
        - Description of the Image :-
          - 1st Image Grid = Ground Truth of Depth Image.
          - 2nd Image Grid = Predicted Depth Image.
          - 3rd Image Grid = Ground Truth of Mask.
          - 4th Image Grid = Predicted Mask Image.
      
<img width="1150" alt="L1SmoothLoss" src="https://user-images.githubusercontent.com/25937235/82765053-17511f80-9e31-11ea-9549-7bddd7cdcfad.png">

      3. L1 Loss : -
        - It gave me satisfactory results in both train and test.
        - I ran total 10 epochs.
        - Mask Loss = 0.002
        - Depth Loss = 0.09
        - IOU for Mask = 0.88
        - IOU for Depth = 0.61
        - Below is the train image which I got after running 1 epoch and 500 batches.
        
          - Description of the Image :- 
            - 1st Image Grid = Ground Truth of Foreground Image overlayed over Background Image.
            - 2nd Image Grid = Ground Truth of Mask.
            - 3rd Image Grid = Ground Truth of Depth Image.
            - 4th Image Grid = Predicted Mask Image.
            - 5th Image grid = Predicted Depth Image.
            
![1_500](https://user-images.githubusercontent.com/25937235/82765230-7499a080-9e32-11ea-9570-e9a099960614.jpg)

        - Below is the train image which I got after running 10 epochs and 2000 batches.
        
         - Description of the Image :- 
            - 1st Image Grid = Ground Truth of Foreground Image overlayed over Background Image.
            - 2nd Image Grid = Ground Truth of Mask.
            - 3rd Image Grid = Ground Truth of Depth Image.
            - 4th Image Grid = Predicted Mask Image.
            - 5th Image grid = Predicted Depth Image.
![12_8500](https://user-images.githubusercontent.com/25937235/82765232-76fbfa80-9e32-11ea-9cd6-88a1b5b28db6.jpg)

        - Below is the test image which I got after running 2 epochs and 500 batches
        
          - Description of the test Image :- 
            - 1st Image Grid = Ground Truth of Foreground Image overlayed over Background Image.
            - 2nd Image Grid = Ground Truth of Mask.
            - 3rd Image Grid = Ground Truth of Depth Image.
            - 4th Image Grid = Predicted Mask Image.
            - 5th Image grid = Predicted Depth Image.
![_test_12_500](https://user-images.githubusercontent.com/25937235/82765234-7c594500-9e32-11ea-84e5-402daf70623a.jpg)

        - Below is the test image which I got after running 10 epochs and 1500 batches
        
           - Description of the test Image :- 
            - 1st Image Grid = Ground Truth of Foreground Image overlayed over Background Image.
            - 2nd Image Grid = Ground Truth of Mask.
            - 3rd Image Grid = Ground Truth of Depth Image.
            - 4th Image Grid = Predicted Mask Image.
            - 5th Image grid = Predicted Depth Image.
![13_2000](https://user-images.githubusercontent.com/25937235/82765235-7f543580-9e32-11ea-9afc-0f82fe7bfdce.jpg)
 
