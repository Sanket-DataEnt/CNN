Group Members: Gunjan Deotale, Abhijit Mali, Sanket Maheshwari, Sanjeev Raichur

Create this dataset and share a link to GDrive (publicly available to anyone) in this readme file. https://drive.google.com/drive/folders/1MST5DUffe3h9Q4B-x7tpNxXl4q4_E8ah

Add your dataset statistics:

1. Kinds of images (fg, bg, fg_bg, masks, depth) : 

    fg :- Different Man, Woman, kids, group of person(for background transparency we have taken png images) 

    bg :- We restricted background to library images(for restricting size of image we have taken jpg images) 

    fg_bg :- bg superposed over fg (for restricting size of images we have taken jpg images) 

    masks :- masks extracted from fg images(we have taken grayscale images)(.jpg) 

    depth :- We have extracted depth images from fg_bg using nyu model(for restricing size of images we have taken grayscale images extracted from colormap)(.jpg)

2. Total images of each kind :
    fg :- 200(flip + no flip) 
    
    bg :- 100 
    
    fg_bg :- 392783 
    
    masks :- 392469 
    
    depth :- 394673

3. The total size of the dataset :-

    9182546 Output/ 
    
    6446124 Output/OverlayedImages/ 
    
    1119541 Output/OverlayedMasks/ 
    
    1616796 Output/DepthImage/

 4. Mean/STD values for your fg_bg, masks and depth images

    fg_bg :- (BGR format) Mean: - [0.3234962448835791, 0.3776562499540454, 0.4548452917585805] stdDev: - [0.22465676724491895, 0.2299902629415973, 0.23860387182601098]

    masks :- (BGR format) Mean: - [0.07863663756127236, 0.07863663756127236, 0.07863663756127236] stdDev: - [0.2541994994472449, 0.2541994994472449, 0.2541994994472449]

    depth :- (BGR format) Mean: - [0.2943823440611593, 0.2943823440611593, 0.2943823440611593] stdDev: - [0.15619204938398595, 0.15619204938398595, 0.15619204938398595]

5. Visualisation of the dataset which I have used : 

    a. Background Images :
    
    ![BG](https://user-images.githubusercontent.com/25937235/81506014-33bd6a00-9311-11ea-9e63-f21a9968edf5.jpg)
    
    b. Foreground Images :
    
    ![FG](https://user-images.githubusercontent.com/25937235/81506017-39b34b00-9311-11ea-9a33-e33fc8597dc2.jpg)
    
    c. Foreground Masks :
    
    ![Mask](https://user-images.githubusercontent.com/25937235/81506025-4637a380-9311-11ea-83b2-82b3aceef001.jpg)
    
    d. Foreground + Background :
    
    ![FG_BG](https://user-images.githubusercontent.com/25937235/81506021-446de000-9311-11ea-909b-a7eb3d765fca.jpg)
    
    e. Foreground + Background + Mask :
    
    ![FG_BG_Mask](https://user-images.githubusercontent.com/25937235/81506026-4899fd80-9311-11ea-9987-d7c56969e72b.jpg)
    
    
Explain how you created your dataset

1. How were fg created with transparency :- 

    We mainly downloaded images from internet without background, for some images we extracted foreground by using background removal technique in PowerPoint as shown in lecture.

2. How were masks created for fgs :-

    We figure out that mask images are nothing but alpha channels of images. So we extracted masks using following code :
    
        image = cv2.imread("Foregroundimg.png", cv2.IMREAD_UNCHANGED) 
        imagealpha = image[:,:,3] 
        cv2.imwrite("ForegroundMask.jpg", imagealpha)

3. How did you overlay the fg over bg and created 20 variants

    - First of all background images were resized to 160x160.
    - All foreground images were resized to 80(max side) and other side was reshaped as per aspect ratio.
    - images were randomly placed by choosing starting x,y randomly on background, but also making sure that foreground image doesnot go out of background image.
    - Code for generation of data is mentioned in DataGeneration.py
    
4. How did you create your depth images?

    Although we used mostly same code as given in assignment for depthimage, we need to modify code to save images.
    We have modified code in utils.py and test.py to save images directly to drive

5. How full data was created?

    Creating full dataset singlehandedly was quite taxing, so we subdivided data in 4 people, each one created 100000 fg_bg, 100000 masks, 100000 depth images
    At the end we merged all folders data in one drive by sharing of folders with each other.
