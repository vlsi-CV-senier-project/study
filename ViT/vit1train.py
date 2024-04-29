import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vit import ViT
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# CIFAR-10 데이터셋을 위한 데이터 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])  # 정규화
])

# CIFAR-10 훈련 데이터셋 및 테스트 데이터셋 로드
train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ViT 모델 초기화
vit_model = ViT(image_size=32, patch_size=4, num_classes=10, dim=512, depth=6, heads=8, mlp_dim=512, dropout = 0.1,emb_dropout = 0.1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
vit_model.to(device)

# 손실 함수 및 최적화 함수 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit_model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=200)



def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct = 0, 0
    
    # tqdm을 이용하여 진행 바 표시
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_correct / len(dataloader.dataset)
    
    return avg_loss, avg_accuracy

def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0

    # tqdm을 이용하여 진행 바 표시
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

    return total_correct / len(dataloader.dataset)

# 훈련 및 평가 실행
for epoch in range(200):  # 10 에포크만큼 훈련
    train_loss, train_accuracy = train(vit_model, train_loader, optimizer, criterion, device)
    test_accuracy = evaluate(vit_model, test_loader, device)
    scheduler.step()
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    #with open('training_results.txt', 'w') as f:
     #   print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}", file=f)
    