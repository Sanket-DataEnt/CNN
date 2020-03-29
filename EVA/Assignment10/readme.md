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
Achieved accuracy of 90.88% in 31th epoch.

### File Details :

Created 10 functions and called them in 10_final.ipynb file. Details of the function are:-

1. trainalbumentation.py : It performs following transformation on train dataset: Horizontalflip, Normalize, Cutout and ToTensor.

2. testalbumentation.py : It performs following transformation on test dataset: Normalize and ToTensor.

3. dataset.py : It will help in downloading CIFAR10 dataset, check wehether GPU is available and put datasets on dataloaders.

4. general_utils.py : It includes general functions which are directly called in the mail file like, imshow(to visualize image), lr_finder(to find the best LR), among others.

5. gradcam.py : It is used to apply GradCAM and heatmap to test data.

6. lr_finer.py : It includes the code to find the best Learning Rate.

7. model.py : It includes the model architecture along with RESNET versions.

8. train_.py : It includes training function.

9. test_.py : It includes test function.

10. init.py: To initialise


### Logs of CIFAR10 dataset using RESNET18 :

   0%|          | 0/391 [00:00<?, ?it/s]

lr=  0.1
EPOCH : 0

Epoch= 0 Loss=1.5123251676559448 Batch_id=390 Accuracy=45.00: 100%|██████████| 391/391 [01:14<00:00,  5.26it/s]
100%|██████████| 79/79 [00:03<00:00, 19.83it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 1.6498334 Test Accuracy= 43.29
lr=  0.1
EPOCH : 1

Epoch= 1 Loss=1.2597752809524536 Batch_id=390 Accuracy=62.50: 100%|██████████| 391/391 [01:17<00:00,  5.07it/s]
100%|██████████| 79/79 [00:03<00:00, 19.76it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 1.1307361 Test Accuracy= 58.38
lr=  0.1
EPOCH : 2

Epoch= 2 Loss=1.1895872354507446 Batch_id=390 Accuracy=62.50: 100%|██████████| 391/391 [01:16<00:00,  5.09it/s]
100%|██████████| 79/79 [00:04<00:00, 19.25it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.68496805 Test Accuracy= 67.05
lr=  0.1
EPOCH : 3

Epoch= 3 Loss=0.7186205983161926 Batch_id=390 Accuracy=78.75: 100%|██████████| 391/391 [01:16<00:00,  5.09it/s]
100%|██████████| 79/79 [00:04<00:00, 19.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.48704386 Test Accuracy= 74.0
lr=  0.1
EPOCH : 4

Epoch= 4 Loss=0.687415599822998 Batch_id=390 Accuracy=80.00: 100%|██████████| 391/391 [01:17<00:00,  5.07it/s]
100%|██████████| 79/79 [00:04<00:00, 19.55it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 1.1058831 Test Accuracy= 78.58
lr=  0.1
EPOCH : 5

Epoch= 5 Loss=0.5043783783912659 Batch_id=390 Accuracy=91.25: 100%|██████████| 391/391 [01:17<00:00,  5.07it/s]
100%|██████████| 79/79 [00:03<00:00, 20.06it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.6915033 Test Accuracy= 80.57
lr=  0.1
EPOCH : 6

Epoch= 6 Loss=0.5488005876541138 Batch_id=390 Accuracy=88.75: 100%|██████████| 391/391 [01:17<00:00,  5.07it/s]
100%|██████████| 79/79 [00:04<00:00, 19.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.4049392 Test Accuracy= 82.08
lr=  0.1
EPOCH : 7

Epoch= 7 Loss=0.5193403959274292 Batch_id=390 Accuracy=88.75: 100%|██████████| 391/391 [01:17<00:00,  5.07it/s]
100%|██████████| 79/79 [00:04<00:00, 19.56it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.41967934 Test Accuracy= 83.4
lr=  0.1
EPOCH : 8

Epoch= 8 Loss=0.27697452902793884 Batch_id=390 Accuracy=95.00: 100%|██████████| 391/391 [01:17<00:00,  5.07it/s]
100%|██████████| 79/79 [00:04<00:00, 19.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.27373827 Test Accuracy= 84.39
lr=  0.1
EPOCH : 9

Epoch= 9 Loss=0.32973194122314453 Batch_id=390 Accuracy=91.25: 100%|██████████| 391/391 [01:17<00:00,  5.06it/s]
100%|██████████| 79/79 [00:03<00:00, 19.83it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.5875218 Test Accuracy= 83.37
lr=  0.1
EPOCH : 10

Epoch= 10 Loss=0.2828597128391266 Batch_id=390 Accuracy=96.25: 100%|██████████| 391/391 [01:17<00:00,  5.07it/s]
100%|██████████| 79/79 [00:03<00:00, 19.83it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.17574593 Test Accuracy= 86.24
lr=  0.1
EPOCH : 11

Epoch= 11 Loss=0.1646014153957367 Batch_id=390 Accuracy=97.50: 100%|██████████| 391/391 [01:16<00:00,  5.09it/s]
100%|██████████| 79/79 [00:03<00:00, 20.02it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.57582146 Test Accuracy= 85.61
lr=  0.1
EPOCH : 12

Epoch= 12 Loss=0.26126980781555176 Batch_id=390 Accuracy=97.50: 100%|██████████| 391/391 [01:17<00:00,  5.07it/s]
100%|██████████| 79/79 [00:04<00:00, 19.73it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.27803802 Test Accuracy= 85.48
lr=  0.1
EPOCH : 13

Epoch= 13 Loss=0.18847133219242096 Batch_id=390 Accuracy=96.25: 100%|██████████| 391/391 [01:17<00:00,  5.07it/s]
100%|██████████| 79/79 [00:03<00:00, 19.76it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.22922942 Test Accuracy= 86.66
lr=  0.1
EPOCH : 14

Epoch= 14 Loss=0.21946251392364502 Batch_id=390 Accuracy=97.50: 100%|██████████| 391/391 [01:16<00:00,  5.13it/s]
100%|██████████| 79/79 [00:03<00:00, 19.83it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.71463203 Test Accuracy= 84.94
lr=  0.05
EPOCH : 15

Epoch= 15 Loss=0.05956057459115982 Batch_id=390 Accuracy=98.75: 100%|██████████| 391/391 [01:16<00:00,  5.09it/s]
100%|██████████| 79/79 [00:04<00:00, 19.65it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.432046 Test Accuracy= 89.43
lr=  0.05
EPOCH : 16

Epoch= 16 Loss=0.0780767872929573 Batch_id=390 Accuracy=98.75: 100%|██████████| 391/391 [01:17<00:00,  5.08it/s]
100%|██████████| 79/79 [00:04<00:00, 19.68it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.26869252 Test Accuracy= 88.93
lr=  0.05
EPOCH : 17

Epoch= 17 Loss=0.09490491449832916 Batch_id=390 Accuracy=98.75: 100%|██████████| 391/391 [01:17<00:00,  5.04it/s]
100%|██████████| 79/79 [00:04<00:00, 19.66it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.6207756 Test Accuracy= 89.28
lr=  0.05
EPOCH : 18

Epoch= 18 Loss=0.052327901124954224 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:16<00:00,  5.09it/s]
100%|██████████| 79/79 [00:04<00:00, 19.40it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.44001862 Test Accuracy= 89.39
lr=  0.025
EPOCH : 19

Epoch= 19 Loss=0.05902609974145889 Batch_id=390 Accuracy=98.75: 100%|██████████| 391/391 [01:17<00:00,  5.03it/s]
100%|██████████| 79/79 [00:04<00:00, 19.69it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.45499298 Test Accuracy= 90.06
lr=  0.025
EPOCH : 20

Epoch= 20 Loss=0.02836495079100132 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:17<00:00,  5.05it/s]
100%|██████████| 79/79 [00:04<00:00, 19.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.14469838 Test Accuracy= 90.31
lr=  0.025
EPOCH : 21

Epoch= 21 Loss=0.02139303646981716 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:17<00:00,  5.03it/s]
100%|██████████| 79/79 [00:04<00:00, 19.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.27680984 Test Accuracy= 90.35
lr=  0.025
EPOCH : 22

Epoch= 22 Loss=0.019117027521133423 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:16<00:00,  5.11it/s]
100%|██████████| 79/79 [00:04<00:00, 19.50it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.86188716 Test Accuracy= 89.88
lr=  0.025
EPOCH : 23

Epoch= 23 Loss=0.006897866725921631 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:17<00:00,  5.07it/s]
100%|██████████| 79/79 [00:04<00:00, 19.38it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.7032445 Test Accuracy= 90.27
lr=  0.025
EPOCH : 24

Epoch= 24 Loss=0.004496121313422918 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:16<00:00,  5.09it/s]
100%|██████████| 79/79 [00:04<00:00, 19.33it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.31070104 Test Accuracy= 90.35
lr=  0.0125
EPOCH : 25

Epoch= 25 Loss=0.013602388091385365 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:17<00:00,  5.07it/s]
100%|██████████| 79/79 [00:04<00:00, 19.35it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.3593832 Test Accuracy= 90.6
lr=  0.0125
EPOCH : 26

Epoch= 26 Loss=0.03450096771121025 Batch_id=390 Accuracy=98.75: 100%|██████████| 391/391 [01:17<00:00,  5.03it/s]
100%|██████████| 79/79 [00:04<00:00, 19.69it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.18752003 Test Accuracy= 90.73
lr=  0.0125
EPOCH : 27

Epoch= 27 Loss=0.030811602249741554 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:16<00:00,  5.11it/s]
100%|██████████| 79/79 [00:04<00:00, 19.31it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.9418331 Test Accuracy= 90.59
lr=  0.0125
EPOCH : 28

Epoch= 28 Loss=0.07312172651290894 Batch_id=390 Accuracy=98.75: 100%|██████████| 391/391 [01:17<00:00,  5.03it/s]
100%|██████████| 79/79 [00:04<00:00, 19.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.3063342 Test Accuracy= 90.7
lr=  0.00625
EPOCH : 29

Epoch= 29 Loss=0.0029149590991437435 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:17<00:00,  5.06it/s]
100%|██████████| 79/79 [00:04<00:00, 19.66it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.05469647 Test Accuracy= 90.74
lr=  0.00625
EPOCH : 30

Epoch= 30 Loss=0.007004410028457642 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:17<00:00,  5.04it/s]
100%|██████████| 79/79 [00:04<00:00, 19.41it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.2910376 Test Accuracy= 90.72
lr=  0.00625
EPOCH : 31

Epoch= 31 Loss=0.001278007053770125 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:17<00:00,  5.03it/s]
100%|██████████| 79/79 [00:03<00:00, 19.77it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.37462485 Test Accuracy= 90.88
lr=  0.00625
EPOCH : 32

Epoch= 32 Loss=0.0035407363902777433 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:17<00:00,  5.02it/s]
100%|██████████| 79/79 [00:04<00:00, 19.50it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.8330813 Test Accuracy= 90.62
lr=  0.00625
EPOCH : 33

Epoch= 33 Loss=0.004569208715111017 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:18<00:00,  5.01it/s]
100%|██████████| 79/79 [00:04<00:00, 19.64it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.04147789 Test Accuracy= 90.85
lr=  0.00625
EPOCH : 34

Epoch= 34 Loss=0.0017037272918969393 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:17<00:00,  5.07it/s]
100%|██████████| 79/79 [00:04<00:00, 19.44it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.11748293 Test Accuracy= 90.73
lr=  0.00625
EPOCH : 35

Epoch= 35 Loss=0.006344503257423639 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:16<00:00,  5.08it/s]
100%|██████████| 79/79 [00:04<00:00, 19.59it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.14275476 Test Accuracy= 90.8
lr=  0.00625
EPOCH : 36

Epoch= 36 Loss=0.002796429442241788 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:16<00:00,  5.11it/s]
100%|██████████| 79/79 [00:04<00:00, 19.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.000839293 Test Accuracy= 90.85
lr=  0.00625
EPOCH : 37

Epoch= 37 Loss=0.004674184136092663 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:17<00:00,  5.03it/s]
100%|██████████| 79/79 [00:04<00:00, 19.33it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.35070598 Test Accuracy= 90.7
lr=  0.00625
EPOCH : 38

Epoch= 38 Loss=0.003927850630134344 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:17<00:00,  5.03it/s]
100%|██████████| 79/79 [00:04<00:00, 19.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.0024410486 Test Accuracy= 90.84
lr=  0.00625
EPOCH : 39

Epoch= 39 Loss=0.008460909128189087 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:17<00:00,  5.07it/s]
100%|██████████| 79/79 [00:04<00:00, 19.45it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.09134573 Test Accuracy= 90.68
lr=  0.00625
EPOCH : 40

Epoch= 40 Loss=0.006428241729736328 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:16<00:00,  5.09it/s]
100%|██████████| 79/79 [00:04<00:00, 19.51it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.8066479 Test Accuracy= 90.73
lr=  0.003125
EPOCH : 41

Epoch= 41 Loss=0.03427452966570854 Batch_id=390 Accuracy=98.75: 100%|██████████| 391/391 [01:17<00:00,  5.05it/s]
100%|██████████| 79/79 [00:04<00:00, 19.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.03370601 Test Accuracy= 90.6
lr=  0.003125
EPOCH : 42

Epoch= 42 Loss=0.0011128544574603438 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:17<00:00,  5.05it/s]
100%|██████████| 79/79 [00:04<00:00, 19.57it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.19729587 Test Accuracy= 90.84
lr=  0.003125
EPOCH : 43

Epoch= 43 Loss=0.0070192874409258366 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:17<00:00,  5.02it/s]
100%|██████████| 79/79 [00:04<00:00, 19.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.0008852482 Test Accuracy= 90.72
lr=  0.003125
EPOCH : 44

Epoch= 44 Loss=0.0025783181190490723 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:17<00:00,  5.05it/s]
100%|██████████| 79/79 [00:04<00:00, 19.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.57738733 Test Accuracy= 90.76
lr=  0.0015625
EPOCH : 45

Epoch= 45 Loss=0.001134073711000383 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:17<00:00,  5.02it/s]
100%|██████████| 79/79 [00:04<00:00, 19.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.14882621 Test Accuracy= 90.64
lr=  0.0015625
EPOCH : 46

Epoch= 46 Loss=0.0026853501331061125 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:16<00:00,  5.12it/s]
100%|██████████| 79/79 [00:03<00:00, 19.96it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.77674013 Test Accuracy= 90.8
lr=  0.0015625
EPOCH : 47

Epoch= 47 Loss=0.003787511494010687 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:17<00:00,  5.04it/s]
100%|██████████| 79/79 [00:04<00:00, 19.33it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.674089 Test Accuracy= 90.59
lr=  0.0015625
EPOCH : 48

Epoch= 48 Loss=0.0007427573436871171 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:17<00:00,  5.08it/s]
100%|██████████| 79/79 [00:04<00:00, 19.58it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test Loss= 0.8298829 Test Accuracy= 90.8
lr=  0.00078125
EPOCH : 49

Epoch= 49 Loss=0.004010808654129505 Batch_id=390 Accuracy=100.00: 100%|██████████| 391/391 [01:16<00:00,  5.10it/s]
100%|██████████| 79/79 [00:03<00:00, 19.76it/s]

Test Loss= 0.47275516 Test Accuracy= 90.84
Finished Training
