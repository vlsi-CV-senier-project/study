import torch
import torchvision
import time

import torchvision.transforms as transforms
from torch import nn, optim

import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights

import argparse
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Training with Pruning and Quantization')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # CIFAR-100 데이터셋 불러오기와 사전학습된 ResNet50 모델 로드 로직 여기에 포함
    # CIFAR-100 데이터셋 불러오기
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # 사전학습된 ResNet50 모델 로드
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.to(device)

    # 모델의 마지막 fully connected layer를 CIFAR-100 데이터셋에 맞게 수정
    model.fc = torch.nn.Linear(model.fc.in_features, 100)
    model.fc.to(device)

    # 파인튜닝을 위한 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # 학습률 감소 설정

    # 학습 로직
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        loop = tqdm(trainloader, leave=True)
        for i, data in enumerate(loop, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch [{epoch+1}/{args.epochs}]')
            loop.set_postfix(loss=loss.item())
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        scheduler.step()  # 학습률 감소 적용

    # 모델을 평가 모드로 전환
    model.eval()
    # 정확도 평가
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy}%')

    # 모델 크기 계산
    model_size = sum(param.numel() for param in model.parameters())
    print(f'Model size: {model_size} parameters')

    # 추론 속도 평가
    start_time = time.time()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            _ = model(images)
    end_time = time.time()
    total_time = end_time - start_time
    average_inference_time = total_time / len(testset)
    print(f'Average inference time per image: {average_inference_time:.6f} seconds')

if __name__ == '__main__':
    main()
