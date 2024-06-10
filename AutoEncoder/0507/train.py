import torch
import torchvision
from torchvision import datasets
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import time
import matplotlib.pyplot as plt
#from ViT_model import ViT
from AE_T import TransformerAutoEncoder
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import MSELoss
from torchvision.transforms import v2
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import torchmetrics


# image size
IMG_SIZE = 64

train_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8),
    v2.CenterCrop(size=(IMG_SIZE, IMG_SIZE)),
    v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8),
    v2.CenterCrop(size=(IMG_SIZE, IMG_SIZE)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# test transforms
random_tensor = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
train_transform_result = train_transforms(random_tensor)
test_transform_result = test_transforms(random_tensor)
print(train_transform_result.shape)
print(test_transform_result.shape)


# train_datasets
train_datasets = datasets.CIFAR10(root='../../data', train=True, download=True, transform=train_transforms)
test_datasets = datasets.CIFAR10(root='../../data', train=False, download=True, transform=test_transforms)
# check datasets length
print(len(train_datasets))
print(len(test_datasets))

BATCH_SIZE = 64

# dataloaders
train_dataloader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
test_dataloader = DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)

# test dataloaders
# for x, y in train_dataloader:
#     print(x.shape, y)
#     break

# for x, y in test_dataloader:
#     print(x.shape, y)
#     break

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 모델 선언
model = TransformerAutoEncoder().to(DEVICE)

# 옵티마이저, 스케줄러, 손실 함수 선언
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
criterion = MSELoss()

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--name', default="default_name", type=str, help='name to save')
    return parser.parse_args()


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
    print(f"Train Loss: {train_loss:.4f}")

def test_fn(model, dataloader, criterion, device):
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

    print(f"Test Loss: {test_loss:.4f}, PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
    return test_loss

def train_autoencoder(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, epochs, save_period):
    best_loss = float('inf')
    scaler = GradScaler()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_fn(model, train_dataloader, criterion, optimizer, scheduler, device, scaler)
        
        if val_dataloader is not None:
            val_loss = test_fn(model, val_dataloader, criterion, device)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f"./model/best_model_epoch_{epoch+1}.pth")
        
        # if (epoch + 1) % save_period == 0:
        #     torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
    return model

model = train_autoencoder(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, DEVICE, epochs=100, save_period=5)