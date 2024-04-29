import argparse
import torch
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from repvgg import RepConv, Repvgg ,RepConv_Gaussian,RepConv_dilate
from repvgg import RepConv_Gaussian, Repvgg_Gaussian
from repvgg import RepConv_dilate, Repvgg_dilate

def count_parameters(model):
    """모델의 학습 가능한 파라미터 수를 계산합니다."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Argument parsing
parser = argparse.ArgumentParser(description='Train a RepVGG model on CIFAR-10')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 50)')
parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
args = parser.parse_args()

# 데이터셋 설정
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 데이터셋 로딩
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# 모델 초기화 및 파라미터 수 계산
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Repvgg_Gaussian(num_classes=10).to(device)
print(f'Initial model parameters: {count_parameters(model)}')

# 학습
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def train(model, epochs, optimizer, scheduler):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}"), 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f'Loss after epoch {epoch + 1}: {running_loss / len(trainloader):.3f}')
    print(f'Model parameters after training: {count_parameters(model)}')
    print('Finished Training')

train(model, args.epochs, optimizer, scheduler)

# 재매개변수화 후 테스트 및 파라미터 수 출력
def fuse_model(model):
    for m in model.modules():
        if isinstance(m, RepConv_dilate):
            m.deploy = True
            m.fuse_convs()
    return model

model = fuse_model(model)
print(f'Model parameters after reparameterization: {count_parameters(model)}')

model.eval()

lt = time.localtime(time.time())
save_time = time.strftime("%d%H%M",lt)
torch.save(model.state_dict(), './models/'+save_time+'.pt')
