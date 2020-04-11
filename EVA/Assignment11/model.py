import torch
import torch.nn as nn
import torch.nn.functional as F




class Net (nn.Module):
    def __init__(self):
        super (Net, self).__init__()
        
        #prep layer
        self.prep= self.prep_block(3, 64, 3, 1) #input layer, o/p = 32x32x64
        
        #Layer 1
        self.conv1 = self.conv_block(64, 128, 3, 1, 2) #o/p = 16x16x128
        self.res1 = self.res_block(128, 128, 3, 1)

        #Layer 2
        self.conv2 = self.conv_block(128, 256, 3, 1, 2)  #o/p = 8x8x256

        #Layer 3 
        self.conv3 = self.conv_block(256, 512, 3, 1, 2) #o/p = 4x4x512
        self.res3 = self.res_block(512, 512, 3, 1)

        self.pool1 = nn.MaxPool2d(4) #o/p = 1x1x512
        #self.conv4 = nn.Conv2d(512,10,1)
        self.linear = nn.Linear(512,10) # 512x10
        
    def prep_block(self,inputs, output, kernel, p):
        prep_bloc = nn.Sequential(nn.Conv2d(in_channels=inputs, out_channels=output, kernel_size=(kernel, kernel), padding = p, bias=False),
                                 nn.BatchNorm2d(output),
                                 nn.ReLU())
        return prep_bloc

    def conv_block(self, inputs, output, kernel, p, m):
        conv_bloc = nn.Sequential(nn.Conv2d(in_channels=inputs, out_channels=output, kernel_size=(kernel, kernel), padding = p, bias=False),
                                 nn.MaxPool2d(m),
                                 nn.BatchNorm2d(output),
                                 nn.ReLU())
        return conv_bloc

    def res_block(self, inputs, output, kernel, p):
        res_bloc = nn.Sequential(nn.Conv2d(in_channels=inputs, out_channels=output, kernel_size=(kernel, kernel), padding = p, bias=False),
                                nn.BatchNorm2d(output),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=inputs, out_channels=output, kernel_size=(kernel, kernel), padding = p, bias=False),
                                nn.BatchNorm2d(output),
                                nn.ReLU()
                                )
        return res_bloc

        
    def forward(self, x):
        x = self.prep(x) #i/p
        x = self.conv1(x) 
        r1 = self.res1(x)
        x1 = x + r1 
        x = x + F.relu(x1)
        x = self.conv2(x)
        x = self.conv3(x)
        r3 = self.res3(x)
        x1 = x + r3
        x = x + F.relu(x1)
        x = self.pool1(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        x = F.softmax(x)
        return x