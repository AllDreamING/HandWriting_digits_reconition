import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d
from torch.nn.modules.pooling import MaxPool2d
from torch.utils.data import dataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F


# 指定使用GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#  设定超参数
# 训练次数
num_epochs = 5
# 输出类别
num_classes = 10
# 每批大小
batch_size = 100
# 学习率
learning_rate = 0.001

# 数据集的下载及使用
# 训练集的下载
train_dataset = torchvision.datasets.MNIST(root='data/',train='True',transform=transforms.ToTensor(),download='true')
# 测试集的下载
test_dataset = torchvision.datasets.MNIST(root='data/',train='False',transform=transforms.ToTensor())
# 训练集数据的加载
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
# 测试集
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

# 卷积输出的计算:  N = (W - k + 2P)/S + 1
# LeNet
class LeNet(nn.Module):
    def __init__(self,outchannel=10):
        super(LeNet,self).__init__()
        # 卷积层C1
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=2)
        # 池化层P1
        self.p1 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 卷积层C2
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        # 池化层P2
        self.p2 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 全连接层fc1
        self.fc1 = nn.Linear(16*5*5,120)
        # 全连接层fc2
        self.fc2 = nn.Linear(120,84)
        # 全连接层fc3
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.p1(x)
        x = F.relu(self.conv2(x))
        x = self.p2(x)
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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


# Res18
class Resblk(nn.Module):
    def __init__(self,inchannel,outchannel,stride=2):
        super(Resblk,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.conv2 = nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(outchannel)

        self.extra = nn.Sequential()
        if outchannel != inchannel:
            self.extra = nn.Sequential(
                # 用1*1的卷积来保证输出和输入维度相等
                nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=3,padding=0),
            nn.BatchNorm2d(64)
        )
        # 4个残差块
        self.blk1 = Resblk(64,128,stride=2)
        self.blk2 = Resblk(128,256,stride=2)
        self.blk3 = Resblk(256,512,stride=2)
        self.blk4 = Resblk(512,512,stride=2)

        
        self.fc = nn.Linear(512*1*1,10)

    def forward(self,x):
        out = F.relu(self.conv1(x))

        out = self.blk1(out)
        out = self.blk2(out)
        out = self.blk3(out)
        out = self.blk4(out)
        out = F.adaptive_avg_pool2d(out,[1,1])
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out




model = ConvNET(num_classes).to(device)
# 选择交叉熵作为损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# 模型的训练
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 将模型的梯度设置为0
        optimizer.zero_grad()
        # 前向传播求出预测值
        outputs = model(images)
        # 计算损失函数
        loss = criterion(outputs,labels)
        # 反向传播求梯度
        loss.backward()
        # 更新所有参数
        optimizer.step()

        if (i+1)%100 == 0 :
            print('Epoch [{}/{}], Step[{}/{}],Loss:{:.4f}'.format(epoch+1,num_epochs,i+1,total_step,loss.item()))

# 模型的测试
model.eval() 
with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
    print('Test Accuracy of the model on the  10000 test images:    {}%'.format(100*correct/total))

# 将模型保存在本地
torch.save(model.state_dict(),'./model.ckpt')
