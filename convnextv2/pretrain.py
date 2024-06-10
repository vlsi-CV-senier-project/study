import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
from torch.optim.lr_scheduler import CosineAnnealingLR
from fcmae import convnextv2_tiny
from tqdm import tqdm
import torchmetrics
from torch.cuda.amp import GradScaler

# 하이퍼파라미터 설정
batch_size = 128
num_epochs = 100
lr = 1e-4
mask_ratio = 0.5
img_size = 32

# CIFAR-10 데이터셋 로드
train_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8),
    v2.CenterCrop(size=(img_size, img_size)),
    v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8),
    v2.CenterCrop(size=(img_size, img_size)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# FCMAE 모델 초기화
model = convnextv2_tiny(img_size=img_size, in_chans=3, patch_size=4, mask_ratio=mask_ratio)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 손실함수와 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

def train_fn(model, dataloader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    train_loss = 0.0
    for images, _ in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            loss, _, _ = model(images)  # Ensure model returns loss as the first output

        # Scale the loss and call backward
        scaler.scale(loss).backward()

        # Step the optimizer with the scaler
        scaler.step(optimizer)
        scaler.update()

        # Update the learning rate scheduler
        scheduler.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(dataloader.dataset)
    print(f"Train Loss: {train_loss:.4f}")


def test_fn(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            with torch.cuda.amp.autocast():
                loss, reconstructed_patches, mask = model(images)
            test_loss += loss.item()
    test_loss /= len(dataloader)
    print(f"Test Loss: {test_loss:.4f}")
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
                torch.save(model.state_dict(), f"best_model_epoch_{epoch+1}.pth")
        if (epoch + 1) % save_period == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
    return model

# 학습 루프
model = train_autoencoder(model, trainloader, testloader, criterion, optimizer, scheduler, device, num_epochs, save_period=10)