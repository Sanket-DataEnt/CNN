## Detailed Analysis of 5 Files:


### EVA4S5F11 (First File)

Target : 99.4% test accuracy in less then 15 epochs

Result : Train Acc (98.76%), Test Acc.(99.44%), Params (10,382)

Analysis : underfitting, and training accuracy is not crossing 98% so it can be further pushed. Achieved 99.4% accuracy but with 10.3K 	parameters.

File link : https://github.com/Sanket-DataEnt/CNN/blob/master/EVA/Assignment5/EVA4S5F11.ipynb


### EVA4S5F12 (Second File)

Target : 99.4% test accuracy in less then 15 epochs with less then 10k parameters

Result : Train Acc (98.30%), Test Acc.(99.36%), Params (9,670)

Analysis : Achieved 99.42% accuracy in 6th epoch with 9.6K parameters. However, achieved 99.4% only once.

File link : https://github.com/Sanket-DataEnt/CNN/blob/master/EVA/Assignment5/EVA4S5F12.ipynb


### EVA4S5F13 (Third File)

Target : 99.4% test accuracy in less then 15 epochs with less then 8k parameters

Result : Train Acc (99.23%), Test Acc.(99.33%), Params (7,724)

Analysis : Achieved 99.33% accuracy in 13th epoch with 7.7K parameters. Accuracy might be decreased due to less number of parameters.

File link : https://github.com/Sanket-DataEnt/CNN/blob/master/EVA/Assignment5/EVA4S5F13.ipynb


### EVA4S5F14 (Fourth File)

Target : 99.4% test accuracy in less then 15 epochs with less then 8k parameters

Result : Train Acc (99.05%), Test Acc.(99.42%), Params (7,724)

Analysis : Achieved 99.42% accuracy in 13th epoch with 7.7K parameters. However, achieved 99.42% only once. I changed learning rate from 0.01 to 0.05 and also dropout from 0.01 to 0.05. It helped in getting this accuracy with only 7.7k parameters.

File link : https://github.com/Sanket-DataEnt/CNN/blob/master/EVA/Assignment5/EVA4S5F14.ipynb


### EVA4S5F15 (Fifth File)

Target : 99.4% test accuracy more than once in less then 15 epochs with less then 8k parameters

Result : Train Acc (99.30%), Test Acc.(99.41%), Params (7,724)

Analysis : Achieved 99.41% accuracy in 12th & 13th epoch with 7.7K parameters. Furthermore, the model is not suffering from overfitting. 

File link : https://github.com/Sanket-DataEnt/CNN/blob/master/EVA/Assignment5/EVA4S5F15.ipynb