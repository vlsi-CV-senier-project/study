import torch
import torchvision
import torch.nn as nn
from torchvision import datasets
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import time
#from ViT_model import ViT
# import model
from AE_T import TransformerAutoEncoder
from AE_CNN_relu import Autoencoder
from AE_CNN_lrelu import Autoencoder_L


from torchvision.transforms import v2
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import torchmetrics
import torchvision.utils as vutils
import numpy as np
# image size
IMG_SIZE = 64

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--name', default="default_name", type=str, help='name to save')
    parser.add_argument('--model_name', default="TransformerAutoEncoder", type=str, help='model to train')
    return parser.parse_args()

args = parse_argument()
print(args)

train_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8),
    # v2.CenterCrop(size=(IMG_SIZE, IMG_SIZE)),
    v2.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE),antialias = True),   #CenterCrop -> RandomResizedCrop
    v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8),
    v2.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE)),        #CenterCrop -> RandomResizedCrop
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# test transforms
# random_tensor = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
# train_transform_result = train_transforms(random_tensor)
# test_transform_result = test_transforms(random_tensor)
# print(train_transform_result.shape)
# print(test_transform_result.shape)


# train_datasets
train_datasets = datasets.CIFAR10(root='../../data', train=True, download=True, transform=train_transforms)
test_datasets = datasets.CIFAR10(root='../../data', train=False, download=True, transform=test_transforms)
# check datasets length
print(len(train_datasets))
print(len(test_datasets))


# dataloaders
train_dataloader = DataLoader(train_datasets, batch_size = args.batch_size, shuffle=True, num_workers=16)
test_dataloader = DataLoader(test_datasets, batch_size = args.batch_size, shuffle=False, num_workers=16)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# 모델 인스턴스화
model_class = globals()[args.model_name]
model = model_class().to(DEVICE)

# 옵티마이저, 스케줄러, 손실 함수 선언
optimizer = optim.Adam(model.parameters(),  lr=args.lr)
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
criterion = nn.MSELoss()

train_losses, test_losses, psnrs, ssims = [], [], [], []


def plot_all_metrics(train_losses, test_losses, psnrs, ssims ):
    plt.figure(figsize=(12, 10))

    # Training Loss 그래프
    plt.subplot(2, 2, 1)  # 2x2 그리드에서 첫 번째 위치
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Testing Loss 그래프
    plt.subplot(2, 2, 2)  # 2x2 그리드에서 두 번째 위치
    plt.plot(test_losses, label='Testing Loss', color= 'red')
    plt.title('Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # PSNR 그래프
    plt.subplot(2, 2, 3)  # 2x2 그리드에서 세 번째 위치
    plt.plot(psnrs, label='PSNR', color='green')
    plt.title('PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.ylim([min(psnrs) - 1, max(psnrs) + 1])  # PSNR 그래프의 y축 범위를 동적으로 조절
    plt.legend()

    # SSIM 그래프
    plt.subplot(2, 2, 4)  # 2x2 그리드에서 네 번째 위치
    plt.plot(ssims, label='SSIM', color='orange')
    plt.title('SSIM')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.ylim([min(ssims) - 0.1, max(ssims) + 0.1])  # SSIM 그래프의 y축 범위를 동적으로 조절
    plt.legend()

    # 모든 그래프를 하나의 이미지로 저장
    plt.tight_layout(pad=3.0)
    plt.savefig(f'./result/{args.name}_all_metrics.png')
    plt.show()


def visualize_reconstructions(model, dataloader, device, num_images=5):
    model.load_state_dict(torch.load(f"./model/{args.name}_best.pth"))
    model.eval()  # 모델을 평가 모드로 설정
    images, _ = next(iter(dataloader))  # 데이터로더에서 첫 배치를 가져옴
    images = images.to(device)
    with torch.no_grad():  # 그라디언트 계산을 비활성화
        outputs = model(images)  # 모델을 통해 이미지를 복원

    # 이미지 선택 및 시각화
    fig, axs = plt.subplots(2, num_images, figsize=(10, 4))
    for i in range(num_images):
        idx = np.random.randint(0, images.size(0))  # 랜덤 인덱스 선택
        # 원본 이미지
        axs[0, i].imshow(np.transpose(vutils.make_grid(images[idx], normalize=True).cpu(), (1, 2, 0)))
        axs[0, i].set_title('Original Image')
        axs[0, i].axis('off')
        # 복원된 이미지
        axs[1, i].imshow(np.transpose(vutils.make_grid(outputs[idx], normalize=True).cpu(), (1, 2, 0)))
        axs[1, i].set_title('Reconstructed Image')
        axs[1, i].axis('off')
    plt.savefig(f'./result/{args.name}_sample.png')
    plt.show()
    

    
def train_fn(model, dataloader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    train_loss = 0.0

    for images, _ in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, images)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        train_loss += loss.item()

    train_loss /= len(dataloader)
    train_losses.append(train_loss)
    print(f"Train Loss: {train_loss:.4f}")

def test_fn(model, dataloader, criterion, device, psnrs, ssims):
    model.eval()
    test_loss = 0.0
    psnr = 0.0
    ssim = 0.0

    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, images)

            test_loss += loss.item()
            psnr += psnr_metric(outputs, images).item()
            ssim += ssim_metric(outputs, images).item()

    test_loss /= len(dataloader)
    psnr /= len(dataloader)
    ssim /= len(dataloader)

    psnrs.append(psnr)
    ssims.append(ssim)
    test_losses.append(test_loss)

    print(f"Test Loss: {test_loss:.4f}, PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
    return test_loss


def train_autoencoder(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, epochs, save_period, psnrs, ssims):
    best_loss = float('inf')
    scaler = GradScaler()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_fn(model, train_dataloader, criterion, optimizer, scheduler, device, scaler)
        
        if val_dataloader is not None:
            val_loss = test_fn(model, val_dataloader, criterion, device, psnrs, ssims)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f"./model/{args.name}_best.pth")


    plot_all_metrics(train_losses, test_losses, psnrs, ssims)
    return model

# 메트릭스 리스트를 인수로 추가
model = train_autoencoder(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, DEVICE, args.epochs, 5, psnrs, ssims)
visualize_reconstructions(model, test_dataloader, DEVICE)