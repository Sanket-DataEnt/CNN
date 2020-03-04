### Problems Solved :

    1. Update the code such that it uses GPU
    2. change the architecture to C1C2C3C40 (basically 3 MPs)
    3. total RF must be more than 44
    4. one of the layers must use Depthwise Separable Convolution
    5. one of the layers must use Dilated Convolution
    6. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
    7. achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M. 


### Files Details:

Created 5 functions and called them in final.ipynb file. Details of the function are:-

1. augmentation.py : It will do all types of augmentation.

2. dataset.py : It will help in downloading CIFAR10 dataset, check wehether GPU is available and put datasets on dataloaders.

3. model.py : It includes the model architecture.

4. training.py : It includes training and test functions

5. init.py: To initialise

Maximum Receptive field achieved in this model is 66. Achieved 85% overall accuracy at 41st epoch.
