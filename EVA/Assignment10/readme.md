### Problems Solved :


    1. Pick your last code
    2. Make sure  to Add CutOut to your code. It should come from your transformations (albumentations)
    3. Use this repo: https://github.com/davidtvs/pytorch-lr-finder (Links to an external site.) 
        Move LR Finder code to your modules
        Implement LR Finder (for SGD, not for ADAM)
        Implement ReduceLROnPlatea: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau (Links to an external site.)
    4. Find best LR to train your model
    5. Use SDG with Momentum
    6. Train for 50 Epochs. 
    7. Show Training and Test Accuracy curves
    8. Target 88% Accuracy.
    9. Run GradCAM on the any 25 misclassified images. Make sure you mention what is the prediction and what was the ground truth label.
    10. Submit

### Accuracy Achieved :
Achieved accuracy of 89.38% in 48th epoch.

### File Details :

Created 10 functions and called them in 10_final.ipynb file. Details of the function are:-

trainalbumentation.py : It performs following transformation on train dataset: Horizontalflip, Normalize, Cutout and ToTensor.

testalbumentation.py : It performs following transformation on test dataset: Normalize and ToTensor.

dataset.py : It will help in downloading CIFAR10 dataset, check wehether GPU is available and put datasets on dataloaders.

general_utils.py : It includes general functions which are directly called in the mail file like, imshow(to visualize image), lr_finder(to find the best LR), among others.

gradcam.py : It is used to apply GradCAM and heatmap to test data.

lr_finer.py : It includes the code to find the best Learning Rate.

model.py : It includes the model architecture along with RESNET versions.

train_.py : It includes training function.

test_.py : It includes test function.

init.py: To initialise


### Logs of CIFAR10 dataset using RESNET18 :

  0%|          | 0/391 [00:00<?, ?it/s]

lr=  0.1
EPOCH : 0

Epoch= 0 Loss=1.4264709949493408 Batch_id=390 Accuracy=53.75: 100%|██████████| 391/391 [00:39<00:00,  9.88it/s]
100%|██████████| 79/79 [00:02<00:00, 30.68it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 1.5774063 Test Accuracy= 43.58
lr=  0.1
EPOCH : 1

Epoch= 1 Loss=1.0528624057769775 Batch_id=390 Accuracy=60.00: 100%|██████████| 391/391 [00:39<00:00,  9.89it/s]
100%|██████████| 79/79 [00:02<00:00, 31.10it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 1.0876403 Test Accuracy= 58.31
lr=  0.1
EPOCH : 2

Epoch= 2 Loss=1.138086199760437 Batch_id=390 Accuracy=63.75: 100%|██████████| 391/391 [00:39<00:00,  9.84it/s]
100%|██████████| 79/79 [00:02<00:00, 31.12it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 1.0374198 Test Accuracy= 66.54
lr=  0.1
EPOCH : 3

Epoch= 3 Loss=0.7424899935722351 Batch_id=390 Accuracy=83.75: 100%|██████████| 391/391 [00:39<00:00,  9.97it/s]
100%|██████████| 79/79 [00:02<00:00, 30.44it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.5522871 Test Accuracy= 71.72
lr=  0.1
EPOCH : 4

Epoch= 4 Loss=0.753738522529602 Batch_id=390 Accuracy=78.75: 100%|██████████| 391/391 [00:39<00:00,  9.93it/s]
100%|██████████| 79/79 [00:02<00:00, 30.90it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.7038654 Test Accuracy= 76.25
lr=  0.1
EPOCH : 5

Epoch= 5 Loss=0.48429131507873535 Batch_id=390 Accuracy=87.50: 100%|██████████| 391/391 [00:39<00:00,  9.95it/s]
100%|██████████| 79/79 [00:02<00:00, 31.22it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.3625278 Test Accuracy= 79.2
lr=  0.1
EPOCH : 6

Epoch= 6 Loss=0.5812996029853821 Batch_id=390 Accuracy=88.75: 100%|██████████| 391/391 [00:38<00:00, 10.05it/s]
100%|██████████| 79/79 [00:02<00:00, 30.96it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.8644341 Test Accuracy= 80.78
lr=  0.1
EPOCH : 7

Epoch= 7 Loss=0.6354566812515259 Batch_id=390 Accuracy=83.75: 100%|██████████| 391/391 [00:39<00:00,  9.92it/s]
100%|██████████| 79/79 [00:02<00:00, 30.71it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.786577 Test Accuracy= 81.36
lr=  0.1
EPOCH : 8

Epoch= 8 Loss=0.3603810667991638 Batch_id=390 Accuracy=95.00: 100%|██████████| 391/391 [00:39<00:00,  9.89it/s]
100%|██████████| 79/79 [00:02<00:00, 30.55it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.5142551 Test Accuracy= 83.74
lr=  0.1
EPOCH : 9

Epoch= 9 Loss=0.4922218322753906 Batch_id=390 Accuracy=90.00: 100%|██████████| 391/391 [00:39<00:00,  9.88it/s]
100%|██████████| 79/79 [00:02<00:00, 30.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.61703986 Test Accuracy= 82.9
lr=  0.05
EPOCH : 10

Epoch= 10 Loss=0.16548825800418854 Batch_id=390 Accuracy=98.75: 100%|██████████| 391/391 [00:39<00:00,  9.94it/s]
100%|██████████| 79/79 [00:02<00:00, 30.57it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.24635674 Test Accuracy= 86.32
lr=  0.05
EPOCH : 11

Epoch= 11 Loss=0.19377431273460388 Batch_id=390 Accuracy=96.25: 100%|██████████| 391/391 [00:39<00:00,  9.93it/s]
100%|██████████| 79/79 [00:02<00:00, 30.73it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.6791507 Test Accuracy= 86.31
lr=  0.05
EPOCH : 12

Epoch= 12 Loss=0.24602946639060974 Batch_id=390 Accuracy=92.50: 100%|██████████| 391/391 [00:39<00:00,  9.93it/s]
100%|██████████| 79/79 [00:02<00:00, 30.77it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.17308977 Test Accuracy= 85.21
lr=  0.05
EPOCH : 13

Epoch= 13 Loss=0.1393483579158783 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.87it/s]
100%|██████████| 79/79 [00:02<00:00, 30.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 1.0758138 Test Accuracy= 86.53
lr=  0.05
EPOCH : 14

Epoch= 14 Loss=0.23494064807891846 Batch_id=390 Accuracy=98.75: 100%|██████████| 391/391 [00:39<00:00,  9.85it/s]
100%|██████████| 79/79 [00:02<00:00, 30.92it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.21138385 Test Accuracy= 86.02
lr=  0.05
EPOCH : 15

Epoch= 15 Loss=0.22283633053302765 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00, 10.00it/s]
100%|██████████| 79/79 [00:02<00:00, 30.40it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.16017818 Test Accuracy= 85.28
lr=  0.05
EPOCH : 16

Epoch= 16 Loss=0.15499861538410187 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.90it/s]
100%|██████████| 79/79 [00:02<00:00, 30.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.651981 Test Accuracy= 85.23
lr=  0.05
EPOCH : 17

Epoch= 17 Loss=0.05582995340228081 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.92it/s]
100%|██████████| 79/79 [00:02<00:00, 30.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.15569893 Test Accuracy= 85.97
lr=  0.05
EPOCH : 18

Epoch= 18 Loss=0.12893252074718475 Batch_id=390 Accuracy=98.75: 100%|██████████| 391/391 [00:39<00:00,  9.86it/s]
100%|██████████| 79/79 [00:02<00:00, 30.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.6002223 Test Accuracy= 86.02
lr=  0.05
EPOCH : 19

Epoch= 19 Loss=0.03433629125356674 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.87it/s]
100%|██████████| 79/79 [00:02<00:00, 30.28it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.30870312 Test Accuracy= 86.91
lr=  0.05
EPOCH : 20

Epoch= 20 Loss=0.10063917934894562 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.92it/s]
100%|██████████| 79/79 [00:02<00:00, 30.45it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.18502977 Test Accuracy= 86.76
lr=  0.05
EPOCH : 21

Epoch= 21 Loss=0.058951862156391144 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.87it/s]
100%|██████████| 79/79 [00:02<00:00, 30.14it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.6866295 Test Accuracy= 85.95
lr=  0.025
EPOCH : 22

Epoch= 22 Loss=0.05623110011219978 Batch_id=390 Accuracy=98.75: 100%|██████████| 391/391 [00:39<00:00,  9.94it/s]
100%|██████████| 79/79 [00:02<00:00, 30.40it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.5882575 Test Accuracy= 88.18
lr=  0.025
EPOCH : 23

Epoch= 23 Loss=0.010054278187453747 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.89it/s]
100%|██████████| 79/79 [00:02<00:00, 30.73it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 1.1340843 Test Accuracy= 88.33
lr=  0.025
EPOCH : 24

Epoch= 24 Loss=0.02390102669596672 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.96it/s]
100%|██████████| 79/79 [00:02<00:00, 30.56it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 1.1749216 Test Accuracy= 88.22
lr=  0.025
EPOCH : 25

Epoch= 25 Loss=0.025589603930711746 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.87it/s]
100%|██████████| 79/79 [00:02<00:00, 30.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.22688136 Test Accuracy= 88.3
lr=  0.0125
EPOCH : 26

Epoch= 26 Loss=0.0031192898750305176 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.88it/s]
100%|██████████| 79/79 [00:02<00:00, 30.99it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.21568039 Test Accuracy= 88.84
lr=  0.0125
EPOCH : 27

Epoch= 27 Loss=0.0018336891662329435 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.86it/s]
100%|██████████| 79/79 [00:02<00:00, 30.76it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 1.192367 Test Accuracy= 88.99
lr=  0.0125
EPOCH : 28

Epoch= 28 Loss=0.0069408295676112175 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.92it/s]
100%|██████████| 79/79 [00:02<00:00, 30.95it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.009840727 Test Accuracy= 88.99
lr=  0.0125
EPOCH : 29

Epoch= 29 Loss=0.014517098665237427 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.98it/s]
100%|██████████| 79/79 [00:02<00:00, 30.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.61438733 Test Accuracy= 89.07
lr=  0.0125
EPOCH : 30

Epoch= 30 Loss=0.04137825965881348 Batch_id=390 Accuracy=98.75: 100%|██████████| 391/391 [00:39<00:00,  9.87it/s]
100%|██████████| 79/79 [00:02<00:00, 30.26it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.5583463 Test Accuracy= 89.01
lr=  0.0125
EPOCH : 31

Epoch= 31 Loss=0.0019211411708965898 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.91it/s]
100%|██████████| 79/79 [00:02<00:00, 30.50it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.11353257 Test Accuracy= 88.81
lr=  0.0125
EPOCH : 32

Epoch= 32 Loss=0.0015236735343933105 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.86it/s]
100%|██████████| 79/79 [00:02<00:00, 30.88it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.16779912 Test Accuracy= 88.9
lr=  0.00625
EPOCH : 33

Epoch= 33 Loss=0.0032729923259466887 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:38<00:00, 10.06it/s]
100%|██████████| 79/79 [00:02<00:00, 30.41it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 1.3890114 Test Accuracy= 89.0
lr=  0.00625
EPOCH : 34

Epoch= 34 Loss=0.002587008522823453 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.88it/s]
100%|██████████| 79/79 [00:02<00:00, 30.40it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.09648889 Test Accuracy= 89.07
lr=  0.00625
EPOCH : 35

Epoch= 35 Loss=0.0013035893207415938 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.90it/s]
100%|██████████| 79/79 [00:02<00:00, 30.82it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.678726 Test Accuracy= 89.16
lr=  0.00625
EPOCH : 36

Epoch= 36 Loss=0.002081936690956354 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.96it/s]
100%|██████████| 79/79 [00:02<00:00, 31.08it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.17095876 Test Accuracy= 89.17
lr=  0.003125
EPOCH : 37

Epoch= 37 Loss=0.0038290799129754305 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.91it/s]
100%|██████████| 79/79 [00:02<00:00, 30.00it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.09537488 Test Accuracy= 89.2
lr=  0.003125
EPOCH : 38

Epoch= 38 Loss=0.0007678151014260948 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.89it/s]
100%|██████████| 79/79 [00:02<00:00, 30.65it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.5315529 Test Accuracy= 89.13
lr=  0.003125
EPOCH : 39

Epoch= 39 Loss=0.00532571692019701 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.94it/s]
100%|██████████| 79/79 [00:02<00:00, 30.61it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.2153798 Test Accuracy= 89.24
lr=  0.003125
EPOCH : 40

Epoch= 40 Loss=0.005643778946250677 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.92it/s]
100%|██████████| 79/79 [00:02<00:00, 30.27it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.47306406 Test Accuracy= 89.14
lr=  0.0015625
EPOCH : 41

Epoch= 41 Loss=0.001472044037654996 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.98it/s]
100%|██████████| 79/79 [00:02<00:00, 30.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.2904172 Test Accuracy= 89.14
lr=  0.0015625
EPOCH : 42

Epoch= 42 Loss=0.05284352973103523 Batch_id=390 Accuracy=98.75: 100%|██████████| 391/391 [00:39<00:00,  9.96it/s]
100%|██████████| 79/79 [00:02<00:00, 30.15it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.11764896 Test Accuracy= 89.23
lr=  0.0015625
EPOCH : 43

Epoch= 43 Loss=0.0009248256683349609 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.82it/s]
100%|██████████| 79/79 [00:02<00:00, 30.87it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 1.0648241 Test Accuracy= 89.2
lr=  0.0015625
EPOCH : 44

Epoch= 44 Loss=0.001385462237522006 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.90it/s]
100%|██████████| 79/79 [00:02<00:00, 30.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.5989394 Test Accuracy= 89.37
lr=  0.00078125
EPOCH : 45

Epoch= 45 Loss=0.0013061284553259611 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:38<00:00, 10.10it/s]
100%|██████████| 79/79 [00:02<00:00, 30.29it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.0027000308 Test Accuracy= 89.25
lr=  0.00078125
EPOCH : 46

Epoch= 46 Loss=0.0002259731263620779 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.85it/s]
100%|██████████| 79/79 [00:02<00:00, 31.02it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.28483367 Test Accuracy= 89.27
lr=  0.00078125
EPOCH : 47

Epoch= 47 Loss=0.000986850238405168 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.97it/s]
100%|██████████| 79/79 [00:02<00:00, 30.43it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.23042843 Test Accuracy= 89.27
lr=  0.00078125
EPOCH : 48

Epoch= 48 Loss=0.0004206061421427876 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.87it/s]
100%|██████████| 79/79 [00:02<00:00, 30.31it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.40459588 Test Accuracy= 89.38
lr=  0.00078125
EPOCH : 49

Epoch= 49 Loss=0.001270854496397078 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [00:39<00:00,  9.89it/s]
100%|██████████| 79/79 [00:02<00:00, 30.94it/s]

Test Loss= 0.6632105 Test Accuracy= 89.24
Finished Training
