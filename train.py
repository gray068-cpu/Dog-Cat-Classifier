#CIFAR-10 | CNN
#=========================================
#IMPORTS AND SETUP -- إستورات المكتبات 

import torch
import torch.nn as nn
import torch.optim as op
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

#import matplotlib.pyplot as plt

#DATA LOADING & AUGMENTATION
#========================================

transform = {
    #TRAIN
    'train':transforms.Compose([
    transforms.Resize(48),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    #EVAL
    'test':transforms.Compose([
    transforms.Resize(48),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5) ,(0.5,0.5,0.5))
    ])
    }

dataset = {
    #TRAIN
    'train':datasets.CIFAR10(root='./data', train=True, download=True, transform =    transform['train']),
    #EVAL
    'test':datasets.CIFAR10(root='./data', train=False, download=True, transform = transform['test'])
}

dataloader = { 
    'train':DataLoader(dataset['train'], batch_size = 32, shuffle=True, num_workers=2, pin_memory=True),
    'test':DataLoader(dataset['test'], batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#=========================================
#MODEL ARCHITECTURE

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(4*4*128, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 4*4*128)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
net = Net().to(device)
#=========================================
#LOSS & OPTIMIZER

criterion = nn.CrossEntropyLoss()
optimizer = op.AdamW(net.parameters(), lr=0.0003, weight_decay=1e-4)

#=========================================
#TRAINING LOOP

def train(epoch):
    net.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(dataloader['train']):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        inputs = net(images)
        loss = criterion(inputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()  
    print(f'Epoch {epoch} Loss: {running_loss/len(dataloader["train"]):.3f}')
    
#=========================================
# EVALUATION
def test():
    net.eval() 
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')

#=========================================
#MAIN EXECUTION
#=========================================
for epoch in range(20):
    train(epoch)
test()
