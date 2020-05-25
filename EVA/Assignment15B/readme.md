## Problem Statement :
Given an image with foreground objects and background image, predict the depth map as well as a mask for the foreground object.

### Data Description

There are four kinds of images in our dataset (fg, bg, fg_bg, masks, depth) :

1. Foreground Image (fg) :- Different Man, woman, kids, group of person(for background transparency only png images were selected)
![FG](https://user-images.githubusercontent.com/25937235/82763067-6d6a9680-9e22-11ea-9325-d7175bfb90ef.jpg)

2. Background Image (bg) :- Restricted background to library images(for restricting size of image jpg format was selected.)
![BG](https://user-images.githubusercontent.com/25937235/82763062-604da780-9e22-11ea-8668-b92505f0276d.jpg)

3. Foreground+Background Image (fg_bg) :- Foreground image superimposed over background image (for restricting size of images jpg format was selected)
![FG_BG](https://user-images.githubusercontent.com/25937235/82783841-f3b3c680-9e7c-11ea-91a7-89f0a06d9522.jpg)

4. Masks :- Masks images were extracted from foreground images(only grayscale images were selected)(.jpg)
![Mask](https://user-images.githubusercontent.com/25937235/82763069-70fe1d80-9e22-11ea-9871-393b0c494b55.jpg)

5. Depth Image :- Extracted depth images from fg_bg image using nyu model(for restricing size of images, grayscale images extracted from colormap. Grayscale image also helps to restrict memory usage. If it is possible to get all the information in one channel why to take three channels.)(.jpg)
![Depth](https://user-images.githubusercontent.com/25937235/82763438-67c28000-9e25-11ea-869c-6124601f3597.jpg)

Total images of each kind : 

  - fg :- 200(flip + no flip)

  - bg :- 100

  - fg_bg :- 400000

  - masks :- 400000

  - depth :- 400000

Total images =  12,001,00

### Input and Output

  - Input -> Background, Background+Foreground
  - Output -> Masks and Depths

### Data Loading

Loading data in Gbs (gigabytes) is a very big challenge to run the model on Google Colab. Following steps were taken to succesfully load the dataset :- 

  Method 1. Loading data from image folders :
  
             - Dataset including 1,200,100 images were available on the google drive inside 120 folders (40 folders each for Foreground+Background, Masks and Depths) with 10,000 images each and a background folder with 100 images.  
             - It took around 40-45 minutes to read these images from 121 folders.
             - It also took around 5-6 hours to train just 1 epoch.
             
  Method 2. Unzipping image folders directly into colab memory :
  
             - Since, above method was not optimized so tried this method.
             - First of all, zip the 121 image folders. It is one time tedious task as it will take around 3 days to zip 1,200,100 images in 121 folders.
             - Now, unzip the above zipped folders into the internal memory of Colab instead of unzipping in the drive. This operation took around 10 minutes.
             - Following code is used to zip the images:
             
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

### Details of files used :-

To make code more simple to understand, several modules were created, which has their own unique operations. Following are the modules used in this project :-

    1. FinalAssignment : This is the main file which calls other modules to predict depth and mask images by giving background and background+foreground images.
    
    2. Augmentation : 
       - This file contains the transformation strategy for the dataset.
       - Since data was more than enough to train the model,hence except resizing the images from 160x160 to 64x64 no other transformations were necessarily required. Due to memory constraint, resizing of images is important.
       - Details of the above module can be found in [augmentation] (augmentation.py)
       
    3. Dataset :
      - As name suggest, this class focusses only on the creation of the dataset.
      - It contains the algorithms which convert the unzipped images into RGB and Grayscale, so that they can be trained by the model.
      - It also have the algorithm which can find the original background image given the Depth or the Mask image.
      - Details of the above code is available on dataset.py.

  
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
 
