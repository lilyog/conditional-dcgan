import torch
import torch.nn as nn
import torch.nn.functional as F
# Generator
class Generator(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, attsize):
        super(Generator, self).__init__()
        self.deconv1_img = nn.ConvTranspose2d(inputSize, hiddenSize*4, 4, 1, 0)
        self.deconv1_att = nn.ConvTranspose2d(attsize, hiddenSize*4, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(hiddenSize*4)
        self.deconv2 = nn.ConvTranspose2d(hiddenSize*8, hiddenSize*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(hiddenSize*4)
        self.deconv3 = nn.ConvTranspose2d(hiddenSize*4, hiddenSize*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(hiddenSize*2)
        self.deconv4 = nn.ConvTranspose2d(hiddenSize*2, hiddenSize, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(hiddenSize)
        self.deconv5 = nn.ConvTranspose2d(hiddenSize, outputSize, 4, 2, 1)
        self.tan = nn.Tanh()
    def forward(self, x, att):
        x = F.leaky_relu(self.deconv1_bn(self.deconv1_img(x)),0.2)
        y = F.leaky_relu(self.deconv1_bn(self.deconv1_att(att)),0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)),0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)),0.2)
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)),0.2)
        x = self.tan(self.deconv5(x))
        return x