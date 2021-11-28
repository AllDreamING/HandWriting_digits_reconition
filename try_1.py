import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.pooling import MaxPool2d
from torch.utils.data import dataset
import torchvision
import torchvision.transforms as transforms

# 指定使用GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#  设定超参数
num_epochs = 5
num_classes = 10
batch_size = 100
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




model = LeNet(num_classes).to(device)
# 交叉熵损失
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# 模型的训练
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
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
