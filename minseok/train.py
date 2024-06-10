import torch
import torchvision
from torchvision import transforms
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import time

#import your model
from model import vgg_net

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    
    args = parser.parse_args()
    return args



def train(model,args):
    DEVICE = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    print(DEVICE)
    
    model.to(DEVICE)
    model.train()
    #dataload
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
        #use matched norm
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    root_dir = './data'
    trainset = torchvision.datasets.MNIST(root = root_dir, download = True, train = True, transform=transform)
    #validset = torchvision.datasets.MNIST(root = root_dir, download = True, train = False, transform=transform)
    num_train = len(trainset)
    num_valid = len(validset)
    print(f"Train data : {num_train}\tValid data : {num_valid}")
    
    train_loader = DataLoader(trainset, batch_size = 64, shuffle = True)
    valid_loader = DataLoader(validset, batch_size = 64, shuffle = True)

    optimizer = torch.optim.SGD(model.parameters(),lr = args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 학습률 감소 설정


    for epoch in range(args.epochs):
        train_loss = 0.0
        train_acc = 0.0
        step = 0
        start_time = time.time()
        for i, sample in enumerate(train_loader):
            step += 1
            
            inputs, labels = sample
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            train_correct = torch.sum(predicted == labels).item()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += train_correct
        end_time = time.time()
        train_loss = train_loss/step
        train_acc = train_acc/step
        print(f"Train Loss: {train_loss:.6f}\tTrain Acc: {train_acc:.4f}\tTime per epoch(min): {(end_time-start_time)/60}")
        scheduler.step()

        #test
        with torch.no_grad():
            model.eval()
            step = 0
            start_time = time.time()
            for i,sample in enumerate(valid_loader):
                step += 1
                optimizer.zero_grad()
                inputs, labels = sample
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                valid_correct = torch.sum(predicted == labels).item()
                loss = criterion(outputs,labels)

                valid_loss += loss.item()
                valid_acc += valid_correct

            end_time = time.time()
            valid_acc = valid_acc/step
            valid_loss = valid_loss/step
            print(f"Valid Loss : {valid_loss:.6f}\tValid Acc : {valid_acc:.4f}\tTime per epoch(min): {(end_time-start_time)/60}")
            model_size = sum(param.numel() for param in model.parameters())
            print(f'Model size: {model_size:,} parameters')

    torch.save(model.state_dict(), 'models.pth')



# args_dict = vars(args)
#     print(args_dict)

if __name__ == '__main__':
    model = vgg()
    train(model, parse_argument())
    args_dict = vars(args)
    print(args_dict)