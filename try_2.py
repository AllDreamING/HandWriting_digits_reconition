from PIL import Image
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class LeNet(nn.Module):
    def __init__(self,num_classes=10):
        super(LeNet,self).__init__()
        # 卷积层C1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            # 池化层P1
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # 卷积层C2
        self.conv2 = nn.Sequential(
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            # 池化层P2
            nn.MaxPool2d(2,2)
        )
        # 全连接层FC1
        self.fc1 = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU()
        )
        # 全连接层FC2
        self.fc2 = nn.Sequential(
            nn.Linear(120,84),
            nn.ReLU()
        )
        # 全连接层FC3
        self.fc3 = nn.Linear(84,10)

    # 定义前向传播过程，输入为x（即按照逻辑将各层进行输出）
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 将矩阵转成一维向量
        x = x.view(x.size()[0],-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class ConvNET(nn.Module):

    def __init__(self,num_classes = 10):
        super(ConvNET,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=5,stride=1,padding=2),
            # 批归一化
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc = nn.Linear(7*7*32,num_classes)
    
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0),-1)
        out = self.fc(out)
        return out

def test_mydata():
    im = plt.imread('new6.png')
    images = Image.open('new6.png')
    images = images.resize((28,28))
    images = images.convert('L')

    transform = transforms.ToTensor()
    images = transform(images)
    images = images.resize(1,1,28,28)

    model = LeNet()
    model.load_state_dict(torch.load('./model.ckpt'))
    model.eval()
    outputs = model(images)

    values,indices = outputs.data.max(1)
    plt.title('{}'.format(int(indices[0])))
    plt.imshow(im)
    plt.show()

test_mydata()