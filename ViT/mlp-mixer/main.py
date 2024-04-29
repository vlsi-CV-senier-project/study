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
from torchvision.datasets.cifar import CIFAR10

class MLP_Block(nn.Module):
    def __init__(self, num_features, expansion, dropout):
        super(MLP_Block, self).__init__()
        num_hidden = int(expansion*num_features)
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.Gelu = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.Gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x 

class Token_Mixer(nn.Module):
    def __init__(self, num_patches, num_channels, expansion, dropout):
        super(Token_Mixer, self).__init__()
        self.layer_norm = nn.LayerNorm(num_channels)
        self.mlp_block = MLP_Block(num_patches, expansion, dropout)
    
    def forward(self, x):
        initial = x
        x = self.layer_norm(x)
        x = torch.transpose(x, 1, 2)
        x = self.mlp_block(x)
        x = torch.transpose(x, 1, 2)
        output = initial + x
        return output

class Channel_Mixer(nn.Module):
    def __init__(self, num_channels, expansion, dropout):
        super(Channel_Mixer, self).__init__()
        self.layer_norm = nn.LayerNorm(num_channels)
        self.mlp_block = MLP_Block(num_channels, expansion, dropout)
    
    def forward(self, x):
        initial = x
        x = self.layer_norm(x)
        x = self.mlp_block(x)
        output = initial + x
        return output

class Mixer_Layer(nn.Module):
    def __init__(self, num_patches, num_channels, expansion_token, expansion_channel,dropout):
        super(Mixer_Layer, self).__init__()

        self.token_mixer = Token_Mixer( num_patches, num_channels, expansion_token, dropout)
        self.channel_mixer = Channel_Mixer(num_channels, expansion_channel, dropout)

    def forward(self, x):
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        return x

class MLP_Mixer(nn.Module):
    def __init__(self, image_shape : tuple, 
                 patch_size: int,
                 num_classes, 
                 num_mixers, 
                 num_features, 
                 expansion_token=4,
                 expansion_channel=0.5, 
                 dropout=0.5):
        
        super(MLP_Mixer, self).__init__()
        
        if len(image_shape)==2:
            in_channel = 1 
        elif len(image_shape):
            in_channel = image_shape[2]

        assert image_shape[0] % patch_size == 0
        num_patches = (image_shape[0]//patch_size)**2

        #this conv layer is only for breaking the image into patches of latent dim size
        self.patch_breaker = nn.Conv2d(in_channel, num_features, kernel_size=patch_size, stride=patch_size)

        layers=[]
        for _ in range(num_mixers):
            layers.append(Mixer_Layer(num_patches, 
                                    num_features,
                                    expansion_token,
                                    expansion_channel, 
                                    dropout))
            
        self.mixer_layers = nn.Sequential(*layers)

        self.final_fc = nn.Linear(num_features , num_classes)

    def forward(self, x):
        patches = self.patch_breaker(x)
        batch_size, num_features, h, w = patches.shape
        patches = patches.permute(0,2,3,1)
        patches = patches.view(batch_size, -1, num_features)

        patches = self.mixer_layers(patches)
                    
        outputs = torch.mean(patches, dim=1)
        outputs = self.final_fc(outputs)

        return outputs

def main():
    # 1. ** argparser **
    parer = argparse.ArgumentParser()
    parer.add_argument('--epoch', type=int, default=100)
    parer.add_argument('--batch_size', type=int, default=128)
    parer.add_argument('--lr', type=float, default=0.001)
    # parer.add_argument('--step_size', type=int, default=390)
    parer.add_argument('--root', type=str, default='../data')
    parer.add_argument('--log_dir', type=str, default='./log_mixer')
    parer.add_argument('--name', type=str, default='mlp_mixer_cifar100')
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
    # train_set = CIFAR100(root=ops.root,
    #                     train=True,
    #                     download=True,
    #                     transform=transform_cifar)

    # test_set = CIFAR100(root=ops.root,
    #                    train=False,
    #                    download=True,
    #                    transform=test_transform_cifar)
    
    train_set = CIFAR10(root=ops.root,
                        train=True,
                        download=True,
                        transform=transform_cifar)

    test_set = CIFAR10(root=ops.root,
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
    # cifar 10 / 100 에 따라 num_classes 수 변경
    model = MLP_Mixer(
        image_shape=(32, 32, 3),  
        patch_size=4,             
        num_classes=10,          
        num_mixers=8,            
        num_features=512,         
        expansion_token=4,
        expansion_channel=0.5,
        dropout=0.5
    ).to(device)


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