## Problems Solved :
1. Write a code that draws this curve (without the arrows). In submission, you'll upload your drawn curve and code for that

2. Write a code which

    1. uses this new ResNet Architecture for Cifar10:
    
        PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        
        Layer1 -
        
            X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
            
            R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
            
            Add(X, R1)
            
        Layer 2 -
        
            Conv 3x3 [256k]
            
            MaxPooling2D
            
            BN
            
            ReLU
            
        Layer 3 -
        
            X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
            
            R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
            
            Add(X, R2)
            
        MaxPooling with Kernel Size 4
        
        FC Layer 
        
        SoftMax
        
    3. Uses One Cycle Policy such that:
    
        Total Epochs = 24
        
        Max at Epoch = 5
        
        LRMIN = FIND
        
        LRMAX = FIND
        
        NO Annihilation
        
    4. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
    
    5. Batch size = 512
    
    6. Target Accuracy: 90%
    
    
  
  ## Accuracy Achieved : 
     90.25% in 24th Epoch. Used CyclicLR.
     
  
  ## File Details : 
  Created 9 functions and called them in 11_final.ipynb file. Details of the function are:-

    trainalbumentation.py : It performs following transformation on train dataset: Horizontalflip, Normalize, Cutout and ToTensor.

    testalbumentation.py : It performs following transformation on test dataset: Normalize and ToTensor.

    dataset.py : It will help in downloading CIFAR10 dataset, check wehether GPU is available and put datasets on dataloaders.

    general_utils.py : It includes general functions which are directly called in the mail file like, imshow(to visualize image), lr_finder(to find the best LR), among others.

    lr_finer.py : It includes the code to find the best Learning Rate.

    model.py : It includes the model architecture along with RESNET versions.

    train_.py : It includes training function.

    test_.py : It includes test function.

    init.py: To initialise

     
