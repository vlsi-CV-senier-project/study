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
from PIL import Image
import glob

#import your model
#from cnn.Conv_model import Autoencoder
from ViT_model import *


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--name', default="default_name", type=str, help='name to save')
    #criterion, optimizer
    args = parser.parse_args()
    return args



def train(model,args):
    DEVICE = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    args_dict = vars(args)
    print(args_dict)

    model.to(DEVICE)
    model.train()
    #dataload
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
        #use matched norm
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    root_dir = '../data'
    trainset = torchvision.datasets.OxfordIIITPet(root = root_dir, download = True, transform=transform)
    #validset = torchvision.datasets.OxfordIIITPet(root = root_dir, download = True, transform=transform)
    num_train = len(trainset)
    #num_valid = len(validset)
    print(f"Train data : {num_train}")
    
    train_loader = DataLoader(trainset, batch_size = 64, shuffle = True)
    #valid_loader = DataLoader(validset, batch_size = 64, shuffle = True)

    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 학습률 감소 설정

# 다른 예제의 구현 방식, 이미지 크기 변경 필요
# train_loss = 0

# def train(autoencoder, train_loader):

#     autoencoder.train()
#     global train_loss
#     for step, (x, label) in enumerate(train_loader):
#         y = x.view(-1, 28*28).to(DEVICE)    #원본으로부터 y 설정
#         x = x + torch.normal(0, 0.5, size = x.size())  #노이즈 추가
#         x = x.view(-1, 28*28).to(DEVICE)
#         label = label.to(DEVICE)

#         encoded, decoded = autoencoder(x)

#         loss = criterion(decoded, y)
#         #실제 예측한 값(decoded), 정답(y)간의 Loss 계산
#         train_loss = loss
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()


    for epoch in range(args.epochs):
        print(f"epoch : {epoch}")
        train_loss = 0.0
        step = 0
        start_time = time.time()
        for i, sample in enumerate(train_loader):
            step += 1
            
            inputs, labels = sample
            #y = inputs.view(-1,128*128).to(DEVICE)
            #inputs = inputs + torch.normal(0, 0.5, size = x.size())  #노이즈 추가
            #inputs = inputs.view(-1, 128*128).to(DEVICE)
            
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            encoder, decoder = model(inputs)

            #_, predicted = torch.max(outputs, 1)
            # train_correct = torch.sum(predicted == labels).item() #필요한 경우 linear를 거친 output 필요
            loss = criterion(decoder, inputs)    #output 이미지와 실제 이미지 비교

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            #train_acc += train_correct
        end_time = time.time()
        train_loss = train_loss/step
        #train_acc = train_acc/step
        print(f"Train Loss: {train_loss:.6f}\tTime per epoch(min): {(end_time-start_time)/60}")
        scheduler.step()

        if epoch == args.epochs-1:
            continue

        #test
        continue    #test용 데이터셋이 별도로 존재하지 않음 -> validation이 불필요
        with torch.no_grad():
            model.eval()
            step = 0
            valid_loss = 0
            start_time = time.time()
            for i,sample in enumerate(valid_loader):
                step += 1
                optimizer.zero_grad()
                inputs, labels = sample
                y = inputs.view(-1,128*128).to(DEVICE)
                inputs = inputs.view(-1,128*128).to(DEVICE)
                labels = labels.to(DEVICE)

                encoder, decoder = model(inputs)
                #_, predicted = torch.max(outputs, 1)
                #valid_correct = torch.sum(predicted == labels).item()
                loss = criterion(decoder, y)

                valid_loss += loss.item()
                #valid_acc += valid_correct

            end_time = time.time()
            #valid_acc = valid_acc/step
            valid_loss = valid_loss/step
            print(f"Valid Loss : {valid_loss:.6f}\tTime per epoch(min): {(end_time-start_time)/60}")
            
    model_size = sum(param.numel() for param in model.parameters())
    print(f'Model size: {model_size:,} parameters')
    torch.save(model.state_dict(), f'./model/{args.name}_models.pth')
        
        
def test_image(args):
    DEVICE = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    model = Autoencoder()
    model.load_state_dict(torch.load(f'./model/{args.name}_models.pth'))
    model.eval()
    model.to(DEVICE)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
        #use matched norm
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    root_dir = '../data'
    dataset = torchvision.datasets.OxfordIIITPet(root = root_dir, download = True, transform=transform)
    subset = torch.utils.data.Subset(dataset, [0,200,400,600,800,1000])
    sub_loader = DataLoader(subset, batch_size = 1, shuffle = False)

    i=-1
    for x, _ in sub_loader:
        i = i+1
        #x = x.view(-1, 128*128)
        x = x.to(DEVICE)
        _, output = model(x)
        #img = np.reshape(x.to("cpu").data.numpy(), (-1,128, 128))
        img = np.reshape(x.to("cpu").data.numpy(), (-1,224, 224))
        img = img.transpose(1,2,0)
        plt.imshow(img)  # Use appropriate color map if needed
        plt.savefig(f'./result/{args.name}_in_{i}.png')  # Save the figure as a PNG file
        plt.close()

        output = output.squeeze(0)
        img = np.reshape(output.to("cpu").data.numpy(), (-1,224, 224))
        img = img.transpose(1,2,0)
        plt.imshow(img)  # Use appropriate color map if needed
        plt.savefig(f'./result/{args.name}_out_{i}.png')  # Save the figure as a PNG file
        plt.close()

def test_vit_image(args):
    DEVICE = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    model = Autoencoder()
    model.load_state_dict(torch.load(f'./model/{args.name}_models.pth'))
    model.eval()
    model.to(DEVICE)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
        #use matched norm
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    root_dir = '../data'
    dataset = torchvision.datasets.OxfordIIITPet(root = root_dir, download = True, transform=transform)
    subset = torch.utils.data.Subset(dataset, [0,200,400,600,800,1000])
    sub_loader = DataLoader(subset, batch_size = 1, shuffle = False)

    for i, (x, _) in enumerate(sub_loader):
        x = x.to(DEVICE)
        _, output = model(x)
    
        input_img = x.squeeze().cpu().permute(1, 2, 0)
        plt.imshow(input_img.numpy())
        plt.axis('off')
        plt.savefig(f'./result/{args.name}_in_{i}.png')
        plt.close()

        output_img = output.squeeze().cpu().permute(1, 2, 0)
        plt.imshow(output_img.detach().numpy(), cmap='gray')
        plt.axis('off')
        plt.savefig(f'./result/{args.name}_out_{i}.png')
        plt.close()




#군집화 함수 필요

if __name__ == '__main__':
    model = Autoencoder()
    args = parse_argument()
    train(model, args)
    test_vit_image(args)