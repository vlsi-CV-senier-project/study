import os
import time
import torch
# import visdom
import argparse
import torch.nn as nn
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
# from timm.models.layers import trunc_normal_
from torchvision.datasets.cifar import CIFAR100

# nn.Parameter를 이용해 구현하는 방법
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size + 1) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x):
        B = x.size(0)

        # 입력 이미지를 패치로 변환
        patches = self.projection(x)  # (B, embed_dim, num_patches^(1/2), num_patches^(1/2))
        _, _, H, W = patches.size()

        # 패치 차원을 전치하여 2D 형태로 변환
        patches = patches.permute(0, 2, 3, 1)  # (B, num_patches^(1/2), num_patches^(1/2), embed_dim)

        # 2D 패치를 1D로 펼치기
        patches = patches.view(B, -1, patches.size(-1))  # (B, num_patches, embed_dim)

        # 위치 정보를 인코딩하기 위해 positional embedding 추가
        patches = patches + self.positional_embedding[:, :patches.size(1), :]

        return patches

# # 모델 초기화
# image_size = 224
# patch_size = 16
# in_channels = 3
# embed_dim = 768

# # 입력 이미지 텐서 생성
# x = torch.randn(1, in_channels, image_size, image_size)

# # 패치 임베딩 모듈 생성
# patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)

# # 패치 임베딩 수행
# patches = patch_embedding(x)
# print(patches.shape)  # 출력: torch.Size([1, 196, 256])

# Multilayer Self-Attention Mechanism
class MSA(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dimension should be divisible by number of heads'
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # split queries, keys, values

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# MLP block used in Transformer
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# Encapsulation of the Transformer block
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MSA(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# Vision Transformer Model
class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_chans=3, num_classes=100, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(image_size, patch_size, in_chans, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, nn.GELU, nn.LayerNorm) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

def main():
    # 1. ** argparser **
    parer = argparse.ArgumentParser()
    parer.add_argument('--epoch', type=int, default=50)
    parer.add_argument('--batch_size', type=int, default=128)
    parer.add_argument('--lr', type=float, default=0.001)
    # parer.add_argument('--step_size', type=int, default=390)
    parer.add_argument('--root', type=str, default='../data')
    parer.add_argument('--log_dir', type=str, default='./log_learn')
    parer.add_argument('--name', type=str, default='vit_learnpos_cifar100')
    # parer.add_argument('--rank', type=int, default=0)
    ops = parer.parse_args()

    # 2. ** device **
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    # 3. ** visdom **
    # vis = visdom.Visdom(port=8097)

    # 4. ** dataset / dataloader **
    transform_cifar = tfs.Compose([
        tfs.RandomCrop(32, padding=4),
        tfs.RandomHorizontalFlip(),
        tfs.ToTensor(),
        tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
                      std=(0.2023, 0.1994, 0.2010)),
    ])

    test_transform_cifar = tfs.Compose([tfs.ToTensor(),
                                        tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                      std=(0.2023, 0.1994, 0.2010)),
                                        ])
    train_set = CIFAR100(root=ops.root,
                        train=True,
                        download=True,
                        transform=transform_cifar)

    test_set = CIFAR100(root=ops.root,
                       train=False,
                       download=True,
                       transform=test_transform_cifar)

    train_loader = DataLoader(dataset=train_set,
                              shuffle=True,
                              batch_size=ops.batch_size)

    test_loader = DataLoader(dataset=test_set,
                             shuffle=False,
                             batch_size=ops.batch_size)

    # 5. ** model **
    model = ViT().to(device)

    # 6. ** criterion **
    criterion = nn.CrossEntropyLoss()

    # 7. ** optimizer **
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=ops.lr,
                                 weight_decay=5e-5)

    # 8. ** scheduler **
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ops.epoch, eta_min=1e-5)

    # 9. ** logger **
    os.makedirs(ops.log_dir, exist_ok=True)

    # 10. ** training **
    print("training...")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    for epoch in range(ops.epoch):
        model.train()
        tic = time.time()
        epoch_loss = 0
        for idx, (img, target) in enumerate(train_loader):
            img = img.to(device)
            target = target.to(device)
            output = model(img)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch} Train Loss: {epoch_loss:.5f} LR: {lr:.5f} Time: {time.time() - tic:.2f}s")

        if epoch == 25 or epoch == 49:
            save_path = os.path.join(ops.log_dir, ops.name, 'saves')
            os.makedirs(save_path, exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(save_path, ops.name + f'.{epoch}.pth.tar'))

        # Validation
        model.eval()
        correct = 0
        val_loss = 0
        total = 0
        with torch.no_grad():
            for img, target in test_loader:
                img = img.to(device)
                target = target.to(device)
                output = model(img)
                loss = criterion(output, target)

                output = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                val_loss += loss.item()
                total += target.size(0)

        val_loss /= len(test_loader)
        accuracy = correct / total
        print(f"Epoch: {epoch} Validation Accuracy: {accuracy * 100:.2f}% Loss: {val_loss:.4f}")

        scheduler.step()

if __name__ == '__main__':
    main()