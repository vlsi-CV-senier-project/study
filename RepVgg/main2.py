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
from repvgg import RepConv_Gaussian, Repvgg_Gaussian

# Argument parsing
parser = argparse.ArgumentParser(description='Train a RepVGG model on CIFAR-10')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 1)')
parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
args = parser.parse_args()

# Dataset transformations
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

# Loading datasets
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

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
        scheduler.step()  # Update the learning rate
        print(f'Loss after epoch {epoch + 1}: {running_loss / len(trainloader):.3f}')
    print('Finished Training')

def fuse_model(model):
    for m in model.modules():
        if isinstance(m, RepConv_Gaussian):
            m.deploy = True
            m.fuse_convs()
    return model

def test(model):
    correct = 0
    total = 0
    inference_time = 0.0  # Total inference time
    with torch.no_grad():
        for data in tqdm(testloader, desc="Evaluating"):
            images, labels = data[0].to(device), data[1].to(device)
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()  # Start time
            outputs = model(images)
            end_time.record()  # End time
            
            torch.cuda.synchronize()  # Waits for everything to finish running
            inference_time += start_time.elapsed_time(end_time)  # Calculates total inference time
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')
        print(f'Average inference time per batch: {inference_time / len(testloader):.3f} ms')

model = Repvgg_Gaussian(num_classes=10).to(device)

# Main script
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # Adjust step_size and gamma as needed

# train
train(model, args.epochs, optimizer, scheduler)

# reparam 후 test
model = fuse_model(model)
model.eval()
test(model)

lt = time.localtime(time.time())
save_time = time.strftime("%d%H%M",lt)
torch.save(model.state_dict(), './models/'+save_time+'.pt')