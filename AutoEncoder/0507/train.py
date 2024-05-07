import torch
import torchvision
from torchvision import transforms
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt

# ViT 모델 불러오기
from ViT_model import *

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--name', default="default_name", type=str, help='name to save')
    args = parser.parse_args()
    return args

def train(model, args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    model.to(DEVICE)
    model.train()
    
    # 데이터 로드 및 변환
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # CIFAR-10 이미지 크기 맞춤
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    valid_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(args.epochs):
        print(f"Epoch: {epoch + 1}")
        train_loss = 0.0
        start_time = time.time()

        for inputs, _ in train_loader:
            inputs = inputs.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Train Loss: {train_loss:.6f}")
        scheduler.step()

        # Validation step
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            for inputs, _ in valid_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                valid_loss += loss.item()

            valid_loss /= len(valid_loader)
            print(f"Validation Loss: {valid_loss:.6f}")

        print(f"Time per epoch (min): {(time.time() - start_time) / 60:.2f}")

    model_size = sum(param.numel() for param in model.parameters())
    print(f'Model size: {model_size:,} parameters')
    torch.save(model.state_dict(), f'./model/{args.name}_model.pth')

def test_vit_image(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder()  # 모델 클래스 이름 확인 필요
    model.load_state_dict(torch.load(f'./model/{args.name}_model.pth'))
    model.eval()
    model.to(DEVICE)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-10 이미지 크기 맞춤
        transforms.ToTensor()
    ])

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    for i, (inputs, _) in enumerate(test_loader):
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)

        input_img = inputs.squeeze().cpu().permute(1, 2, 0)
        plt.imshow(input_img.numpy())
        plt.axis('off')
        plt.savefig(f'./result/{args.name}_in_{i}.png')
        plt.close()

        output_img = outputs.squeeze().cpu().permute(1, 2, 0)
        plt.imshow(output_img.detach().numpy(), cmap='gray')
        plt.axis('off')
        plt.savefig(f'./result/{args.name}_out_{i}.png')
        plt.close()

if __name__ == '__main__':
    args = parse_argument()
    model = Autoencoder()  # 모델 클래스 이름 변경 필요
    train(model, args)
    test_vit_image(args)
