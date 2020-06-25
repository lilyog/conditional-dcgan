import torch
import torch.nn as nn
import torch.nn.functional as F
# Discriminator
class Discriminator(nn.Module):
    def __init__(self, inputSize, hiddenSize, attsize):
        super(Discriminator, self).__init__()
        self.conv1_img = nn.Conv2d(inputSize, hiddenSize//2, 4, 2, 1)
        self.conv1_att = nn.Conv2d(attsize, hiddenSize//2, 4, 2, 1) 
        self.conv2 = nn.Conv2d(hiddenSize, hiddenSize*2, 4, 2, 1)   
        self.conv2_bn = nn.BatchNorm2d(hiddenSize*2)
        self.conv3 = nn.Conv2d(hiddenSize*2, hiddenSize*4, 4, 2, 1)   
        self.conv3_bn = nn.BatchNorm2d(hiddenSize*4)
        self.conv4 = nn.Conv2d(hiddenSize*4, hiddenSize*8, 4, 2, 1) 
        self.conv4_bn = nn.BatchNorm2d(hiddenSize*8)  
        self.conv5 = nn.Conv2d(hiddenSize*8, 1, 4, 1, 0)
        self.sig = nn.Sigmoid()
    

    def forward(self, x, att):
        x = F.leaky_relu(self.conv1_img(x), 0.2)    #b x hidd x 32 x 32
        att = F.leaky_relu(self.conv1_att(att), 0.2)#b x hidd x 32 x 32
        x = torch.cat([x, att], 1)                  #b x hidd*2 x 32 x 32
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = self.sig(self.conv5(x))
        return x