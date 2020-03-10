### Problems Solved :
  1. Go through this repository: https://github.com/kuangliu/pytorch-cifar (Links to an external site.)
  2. Extract the ResNet18 model from this repository and add it to your API/repo. 
  3. Use your data loader, model loading, train, and test code to train ResNet18 on Cifar10
  4. Your Target is 85% accuracy. No limit on the number of epochs. Use default ResNet18 code (so params are fixed). 
  
### Accuracy achieved:
Achieved accuracy of 88.3% in 86th epoch. 

### Files Details:

Created 5 functions and called them in final.ipynb file. Details of the function are:-

    augmentation.py : It will do all types of augmentation.

    dataset.py : It will help in downloading CIFAR10 dataset, check wehether GPU is available and put datasets on dataloaders.

    model.py : It includes the model architecture along with RESNET versions.

    training_new.py : It includes training and test functions

    init.py: To initialise



### Logs of CIFAR10 dataset using RESNET18

  0%|          | 0/391 [00:00<?, ?it/s]

lr=  0.001
EPOCH: 0

Loss=1.38945734500885 Batch_id=390 Accuracy=36.10: 100%|██████████| 391/391 [03:14<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0121, Accuracy: 4492/10000 (44.92%)

lr=  0.001
EPOCH: 1

Loss=1.3683644533157349 Batch_id=390 Accuracy=52.98: 100%|██████████| 391/391 [03:14<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0099, Accuracy: 5643/10000 (56.43%)

lr=  0.001
EPOCH: 2

Loss=0.9715396761894226 Batch_id=390 Accuracy=61.44: 100%|██████████| 391/391 [03:14<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0081, Accuracy: 6363/10000 (63.63%)

lr=  0.001
EPOCH: 3

Loss=0.9238306879997253 Batch_id=390 Accuracy=66.89: 100%|██████████| 391/391 [03:15<00:00,  2.49it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0068, Accuracy: 6872/10000 (68.72%)

lr=  0.001
EPOCH: 4

Loss=0.7251920700073242 Batch_id=390 Accuracy=70.72: 100%|██████████| 391/391 [03:18<00:00,  2.49it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0065, Accuracy: 7108/10000 (71.08%)

lr=  0.001
EPOCH: 5

Loss=0.6583478450775146 Batch_id=390 Accuracy=73.29: 100%|██████████| 391/391 [03:17<00:00,  2.50it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0056, Accuracy: 7467/10000 (74.67%)

lr=  0.001
EPOCH: 6

Loss=0.6623149514198303 Batch_id=390 Accuracy=76.06: 100%|██████████| 391/391 [03:16<00:00,  2.49it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0054, Accuracy: 7594/10000 (75.94%)

lr=  0.001
EPOCH: 7

Loss=0.6148439049720764 Batch_id=390 Accuracy=78.34: 100%|██████████| 391/391 [03:16<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0049, Accuracy: 7863/10000 (78.63%)

lr=  0.001
EPOCH: 8

Loss=0.6131184697151184 Batch_id=390 Accuracy=79.98: 100%|██████████| 391/391 [03:14<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0052, Accuracy: 7708/10000 (77.08%)

lr=  0.001
EPOCH: 9

Loss=0.4793870449066162 Batch_id=390 Accuracy=81.46: 100%|██████████| 391/391 [03:15<00:00,  2.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0044, Accuracy: 8097/10000 (80.97%)

lr=  0.001
EPOCH: 10

Loss=0.5642732977867126 Batch_id=390 Accuracy=82.62: 100%|██████████| 391/391 [03:14<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0048, Accuracy: 7954/10000 (79.54%)

lr=  0.001
EPOCH: 11

Loss=0.3612387776374817 Batch_id=390 Accuracy=83.72: 100%|██████████| 391/391 [03:14<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0040, Accuracy: 8267/10000 (82.67%)

lr=  0.001
EPOCH: 12

Loss=0.47900304198265076 Batch_id=390 Accuracy=84.83: 100%|██████████| 391/391 [03:17<00:00,  2.51it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0043, Accuracy: 8195/10000 (81.95%)

lr=  0.001
EPOCH: 13

Loss=0.3347383141517639 Batch_id=390 Accuracy=85.61: 100%|██████████| 391/391 [03:18<00:00,  2.49it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0041, Accuracy: 8268/10000 (82.68%)

lr=  0.001
EPOCH: 14

Loss=0.24627470970153809 Batch_id=390 Accuracy=86.76: 100%|██████████| 391/391 [03:18<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0043, Accuracy: 8216/10000 (82.16%)

lr=  0.001
EPOCH: 15

Loss=0.3578574061393738 Batch_id=390 Accuracy=87.29: 100%|██████████| 391/391 [03:17<00:00,  2.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0040, Accuracy: 8372/10000 (83.72%)

lr=  0.001
EPOCH: 16

Loss=0.2623690962791443 Batch_id=390 Accuracy=88.00: 100%|██████████| 391/391 [03:15<00:00,  2.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0038, Accuracy: 8404/10000 (84.04%)

lr=  0.001
EPOCH: 17

Loss=0.33591628074645996 Batch_id=390 Accuracy=88.52: 100%|██████████| 391/391 [03:14<00:00,  2.51it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0040, Accuracy: 8378/10000 (83.78%)

lr=  0.001
EPOCH: 18

Loss=0.28435462713241577 Batch_id=390 Accuracy=89.49: 100%|██████████| 391/391 [03:14<00:00,  2.56it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0039, Accuracy: 8377/10000 (83.77%)

lr=  0.001
EPOCH: 19

Loss=0.3006454110145569 Batch_id=390 Accuracy=89.71: 100%|██████████| 391/391 [03:14<00:00,  2.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0036, Accuracy: 8519/10000 (85.19%)

lr=  0.001
EPOCH: 20

Loss=0.3151876628398895 Batch_id=390 Accuracy=90.34: 100%|██████████| 391/391 [03:15<00:00,  2.48it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0039, Accuracy: 8459/10000 (84.59%)

lr=  0.001
EPOCH: 21

Loss=0.22972705960273743 Batch_id=390 Accuracy=90.91: 100%|██████████| 391/391 [03:18<00:00,  2.48it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0037, Accuracy: 8528/10000 (85.28%)

lr=  0.001
EPOCH: 22

Loss=0.2846398651599884 Batch_id=390 Accuracy=91.38: 100%|██████████| 391/391 [03:18<00:00,  2.50it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0038, Accuracy: 8533/10000 (85.33%)

lr=  0.001
EPOCH: 23

Loss=0.30655190348625183 Batch_id=390 Accuracy=91.96: 100%|██████████| 391/391 [03:16<00:00,  2.57it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0038, Accuracy: 8517/10000 (85.17%)

lr=  0.001
EPOCH: 24

Loss=0.1836203783750534 Batch_id=390 Accuracy=92.28: 100%|██████████| 391/391 [03:15<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0038, Accuracy: 8553/10000 (85.53%)

lr=  0.001
EPOCH: 25

Loss=0.09235204756259918 Batch_id=390 Accuracy=92.91: 100%|██████████| 391/391 [03:14<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0039, Accuracy: 8526/10000 (85.26%)

lr=  0.001
EPOCH: 26

Loss=0.30905693769454956 Batch_id=390 Accuracy=93.06: 100%|██████████| 391/391 [03:13<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0038, Accuracy: 8568/10000 (85.68%)

lr=  0.001
EPOCH: 27

Loss=0.19437088072299957 Batch_id=390 Accuracy=93.46: 100%|██████████| 391/391 [03:15<00:00,  2.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0036, Accuracy: 8630/10000 (86.30%)

lr=  0.001
EPOCH: 28

Loss=0.1371096819639206 Batch_id=390 Accuracy=94.13: 100%|██████████| 391/391 [03:19<00:00,  2.48it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0037, Accuracy: 8633/10000 (86.33%)

lr=  0.001
EPOCH: 29

Loss=0.15439748764038086 Batch_id=390 Accuracy=94.36: 100%|██████████| 391/391 [03:19<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0036, Accuracy: 8623/10000 (86.23%)

lr=  0.001
EPOCH: 30

Loss=0.12962642312049866 Batch_id=390 Accuracy=94.56: 100%|██████████| 391/391 [03:17<00:00,  2.55it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0037, Accuracy: 8623/10000 (86.23%)

lr=  0.001
EPOCH: 31

Loss=0.1599341183900833 Batch_id=390 Accuracy=95.05: 100%|██████████| 391/391 [03:14<00:00,  2.51it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0036, Accuracy: 8661/10000 (86.61%)

lr=  0.001
EPOCH: 32

Loss=0.08182393014431 Batch_id=390 Accuracy=95.48: 100%|██████████| 391/391 [03:15<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0037, Accuracy: 8676/10000 (86.76%)

lr=  0.001
EPOCH: 33

Loss=0.163604274392128 Batch_id=390 Accuracy=95.51: 100%|██████████| 391/391 [03:14<00:00,  2.51it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0037, Accuracy: 8640/10000 (86.40%)

lr=  0.001
EPOCH: 34

Loss=0.1087997704744339 Batch_id=390 Accuracy=95.90: 100%|██████████| 391/391 [03:19<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0037, Accuracy: 8695/10000 (86.95%)

lr=  0.001
EPOCH: 35

Loss=0.27844586968421936 Batch_id=390 Accuracy=96.06: 100%|██████████| 391/391 [03:21<00:00,  2.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0038, Accuracy: 8683/10000 (86.83%)

lr=  0.001
EPOCH: 36

Loss=0.12486498057842255 Batch_id=390 Accuracy=96.46: 100%|██████████| 391/391 [03:20<00:00,  2.45it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0038, Accuracy: 8670/10000 (86.70%)

lr=  0.001
EPOCH: 37

Loss=0.14843301475048065 Batch_id=390 Accuracy=96.40: 100%|██████████| 391/391 [03:18<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0041, Accuracy: 8606/10000 (86.06%)

lr=  0.001
EPOCH: 38

Loss=0.05948757380247116 Batch_id=390 Accuracy=96.74: 100%|██████████| 391/391 [03:15<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0038, Accuracy: 8728/10000 (87.28%)

lr=  0.001
EPOCH: 39

Loss=0.14601098001003265 Batch_id=390 Accuracy=96.75: 100%|██████████| 391/391 [03:16<00:00,  2.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0041, Accuracy: 8684/10000 (86.84%)

lr=  0.001
EPOCH: 40

Loss=0.10267338901758194 Batch_id=390 Accuracy=97.20: 100%|██████████| 391/391 [03:15<00:00,  2.50it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0042, Accuracy: 8646/10000 (86.46%)

lr=  0.001
EPOCH: 41

Loss=0.05390879511833191 Batch_id=390 Accuracy=97.19: 100%|██████████| 391/391 [03:18<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0039, Accuracy: 8730/10000 (87.30%)

lr=  0.001
EPOCH: 42

Loss=0.050198666751384735 Batch_id=390 Accuracy=97.38: 100%|██████████| 391/391 [03:23<00:00,  2.45it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0039, Accuracy: 8693/10000 (86.93%)

lr=  0.001
EPOCH: 43

Loss=0.14187276363372803 Batch_id=390 Accuracy=97.58: 100%|██████████| 391/391 [03:23<00:00,  2.41it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0038, Accuracy: 8737/10000 (87.37%)

lr=  0.001
EPOCH: 44

Loss=0.06742522865533829 Batch_id=390 Accuracy=97.64: 100%|██████████| 391/391 [03:20<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0041, Accuracy: 8693/10000 (86.93%)

lr=  0.001
EPOCH: 45

Loss=0.04283802583813667 Batch_id=390 Accuracy=97.87: 100%|██████████| 391/391 [03:15<00:00,  2.56it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0043, Accuracy: 8651/10000 (86.51%)

lr=  0.001
EPOCH: 46

Loss=0.018683135509490967 Batch_id=390 Accuracy=97.75: 100%|██████████| 391/391 [03:16<00:00,  2.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0038, Accuracy: 8768/10000 (87.68%)

lr=  0.001
EPOCH: 47

Loss=0.06467752158641815 Batch_id=390 Accuracy=97.99: 100%|██████████| 391/391 [03:15<00:00,  2.50it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0041, Accuracy: 8703/10000 (87.03%)

lr=  0.001
EPOCH: 48

Loss=0.038294486701488495 Batch_id=390 Accuracy=98.20: 100%|██████████| 391/391 [03:19<00:00,  2.45it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0041, Accuracy: 8734/10000 (87.34%)

lr=  0.001
EPOCH: 49

Loss=0.06935688108205795 Batch_id=390 Accuracy=98.02: 100%|██████████| 391/391 [03:23<00:00,  2.41it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0041, Accuracy: 8701/10000 (87.01%)

lr=  0.001
EPOCH: 50

Loss=0.021208738908171654 Batch_id=390 Accuracy=98.23: 100%|██████████| 391/391 [03:21<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0040, Accuracy: 8764/10000 (87.64%)

lr=  0.001
EPOCH: 51

Loss=0.059570468962192535 Batch_id=390 Accuracy=98.20: 100%|██████████| 391/391 [03:20<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0039, Accuracy: 8777/10000 (87.77%)

lr=  0.001
EPOCH: 52

Loss=0.11136598885059357 Batch_id=390 Accuracy=98.36: 100%|██████████| 391/391 [03:16<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0042, Accuracy: 8775/10000 (87.75%)

lr=  0.001
EPOCH: 53

Loss=0.13013973832130432 Batch_id=390 Accuracy=98.54: 100%|██████████| 391/391 [03:16<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0040, Accuracy: 8792/10000 (87.92%)

lr=  0.001
EPOCH: 54

Loss=0.008410257287323475 Batch_id=390 Accuracy=98.54: 100%|██████████| 391/391 [03:16<00:00,  2.49it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0042, Accuracy: 8758/10000 (87.58%)

lr=  0.001
EPOCH: 55

Loss=0.019356150180101395 Batch_id=390 Accuracy=98.56: 100%|██████████| 391/391 [03:18<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0043, Accuracy: 8750/10000 (87.50%)

lr=  0.001
EPOCH: 56

Loss=0.020375484600663185 Batch_id=390 Accuracy=98.63: 100%|██████████| 391/391 [03:22<00:00,  2.44it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0043, Accuracy: 8747/10000 (87.47%)

lr=  0.001
EPOCH: 57

Loss=0.04855557531118393 Batch_id=390 Accuracy=98.69: 100%|██████████| 391/391 [03:21<00:00,  2.45it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0040, Accuracy: 8847/10000 (88.47%)

lr=  0.001
EPOCH: 58

Loss=0.01712799072265625 Batch_id=390 Accuracy=98.88: 100%|██████████| 391/391 [03:20<00:00,  2.49it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0043, Accuracy: 8774/10000 (87.74%)

lr=  0.001
EPOCH: 59

Loss=0.05429081246256828 Batch_id=390 Accuracy=98.88: 100%|██████████| 391/391 [03:15<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0042, Accuracy: 8744/10000 (87.44%)

lr=  0.001
EPOCH: 60

Loss=0.06980063766241074 Batch_id=390 Accuracy=98.89: 100%|██████████| 391/391 [03:16<00:00,  2.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0043, Accuracy: 8759/10000 (87.59%)

lr=  0.001
EPOCH: 61

Loss=0.035055506974458694 Batch_id=390 Accuracy=98.88: 100%|██████████| 391/391 [03:19<00:00,  2.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0041, Accuracy: 8811/10000 (88.11%)

lr=  0.001
EPOCH: 62

Loss=0.06170586496591568 Batch_id=390 Accuracy=99.01: 100%|██████████| 391/391 [03:21<00:00,  2.43it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0043, Accuracy: 8787/10000 (87.87%)

lr=  0.001
EPOCH: 63

Loss=0.010923338122665882 Batch_id=390 Accuracy=98.98: 100%|██████████| 391/391 [03:22<00:00,  2.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0043, Accuracy: 8775/10000 (87.75%)

lr=  0.001
EPOCH: 64

Loss=0.026546180248260498 Batch_id=390 Accuracy=99.12: 100%|██████████| 391/391 [03:21<00:00,  2.51it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0043, Accuracy: 8768/10000 (87.68%)

lr=  0.001
EPOCH: 65

Loss=0.0572361946105957 Batch_id=390 Accuracy=99.07: 100%|██████████| 391/391 [03:15<00:00,  2.51it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0044, Accuracy: 8779/10000 (87.79%)

lr=  0.001
EPOCH: 66

Loss=0.022574832662940025 Batch_id=390 Accuracy=99.06: 100%|██████████| 391/391 [03:16<00:00,  2.50it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0042, Accuracy: 8794/10000 (87.94%)

lr=  0.001
EPOCH: 67

Loss=0.012550020590424538 Batch_id=390 Accuracy=99.09: 100%|██████████| 391/391 [03:16<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0042, Accuracy: 8817/10000 (88.17%)

lr=  0.001
EPOCH: 68

Loss=0.07774049043655396 Batch_id=390 Accuracy=99.13: 100%|██████████| 391/391 [03:16<00:00,  2.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0044, Accuracy: 8804/10000 (88.04%)

lr=  0.001
EPOCH: 69

Loss=0.026986246928572655 Batch_id=390 Accuracy=99.11: 100%|██████████| 391/391 [03:18<00:00,  2.48it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0043, Accuracy: 8789/10000 (87.89%)

lr=  0.001
EPOCH: 70

Loss=0.073288694024086 Batch_id=390 Accuracy=99.23: 100%|██████████| 391/391 [03:21<00:00,  2.45it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0043, Accuracy: 8789/10000 (87.89%)

lr=  0.001
EPOCH: 71

Loss=0.024812400341033936 Batch_id=390 Accuracy=99.23: 100%|██████████| 391/391 [03:20<00:00,  2.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0042, Accuracy: 8801/10000 (88.01%)

lr=  0.001
EPOCH: 72

Loss=0.01029695849865675 Batch_id=390 Accuracy=99.27: 100%|██████████| 391/391 [03:19<00:00,  2.48it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0043, Accuracy: 8788/10000 (87.88%)

lr=  0.001
EPOCH: 73

Loss=0.01531075220555067 Batch_id=390 Accuracy=99.25: 100%|██████████| 391/391 [03:20<00:00,  2.49it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0043, Accuracy: 8839/10000 (88.39%)

lr=  0.001
EPOCH: 74

Loss=0.026914458721876144 Batch_id=390 Accuracy=99.22: 100%|██████████| 391/391 [03:20<00:00,  2.49it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0043, Accuracy: 8826/10000 (88.26%)

lr=  0.001
EPOCH: 75

Loss=0.027647340670228004 Batch_id=390 Accuracy=99.24: 100%|██████████| 391/391 [03:19<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0045, Accuracy: 8818/10000 (88.18%)

lr=  0.001
EPOCH: 76

Loss=0.03567442297935486 Batch_id=390 Accuracy=99.37: 100%|██████████| 391/391 [03:19<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0043, Accuracy: 8823/10000 (88.23%)

lr=  0.001
EPOCH: 77

Loss=0.010736530646681786 Batch_id=390 Accuracy=99.28: 100%|██████████| 391/391 [03:20<00:00,  2.49it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0045, Accuracy: 8818/10000 (88.18%)

lr=  0.001
EPOCH: 78

Loss=0.028607213869690895 Batch_id=390 Accuracy=99.35: 100%|██████████| 391/391 [03:20<00:00,  2.45it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0045, Accuracy: 8796/10000 (87.96%)

lr=  0.001
EPOCH: 79

Loss=0.005456519313156605 Batch_id=390 Accuracy=99.28: 100%|██████████| 391/391 [03:20<00:00,  2.43it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0043, Accuracy: 8829/10000 (88.29%)

lr=  0.001
EPOCH: 80

Loss=0.1496879905462265 Batch_id=390 Accuracy=99.38: 100%|██████████| 391/391 [03:20<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0044, Accuracy: 8816/10000 (88.16%)

lr=  0.001
EPOCH: 81

Loss=0.007991194725036621 Batch_id=390 Accuracy=99.36: 100%|██████████| 391/391 [03:21<00:00,  2.48it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0044, Accuracy: 8789/10000 (87.89%)

lr=  0.001
EPOCH: 82

Loss=0.022987687960267067 Batch_id=390 Accuracy=99.39: 100%|██████████| 391/391 [03:19<00:00,  2.42it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0043, Accuracy: 8827/10000 (88.27%)

lr=  0.001
EPOCH: 83

Loss=0.0036554515827447176 Batch_id=390 Accuracy=99.38: 100%|██████████| 391/391 [03:20<00:00,  2.41it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0044, Accuracy: 8826/10000 (88.26%)

lr=  0.001
EPOCH: 84

Loss=0.027913808822631836 Batch_id=390 Accuracy=99.41: 100%|██████████| 391/391 [03:21<00:00,  2.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0044, Accuracy: 8825/10000 (88.25%)

lr=  0.001
EPOCH: 85

Loss=0.033405303955078125 Batch_id=390 Accuracy=99.47: 100%|██████████| 391/391 [03:20<00:00,  2.43it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0045, Accuracy: 8797/10000 (87.97%)

lr=  0.001
EPOCH: 86

Loss=0.01086760126054287 Batch_id=390 Accuracy=99.46: 100%|██████████| 391/391 [03:20<00:00,  2.50it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.0045, Accuracy: 8830/10000 (88.30%)

lr=  0.001
EPOCH: 87

Loss=0.02284623309969902 Batch_id=191 Accuracy=99.39:  49%|████▉     | 191/391 [01:38<01:43,  1.94it/s]

Buffered data was truncated after reaching the output size limit.

