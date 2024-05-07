import os
import argparse
import math
import torch
import torchvision
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import v2 as transforms_v2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import *
from utils import setup_seed

# Define transformations using torchvision.transforms.v2
def get_transforms(train=True):
    if train:
        return transforms_v2.Compose([
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.uint8),
            transforms_v2.CenterCrop(size=(224, 224)),
            transforms_v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms_v2.RandomHorizontalFlip(p=0.5),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms_v2.Compose([
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.uint8),
            transforms_v2.CenterCrop(size=(224, 224)),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')

    args = parser.parse_args()
    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)
    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    # Apply the v2 transforms
    train_dataset = OxfordIIITPet(root='data', transform=get_transforms(train=True), download=True)
    dataloader = DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)

    writer = SummaryWriter(os.path.join('logs', 'oxford-pet', 'mae-pretrain'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0  # Initialize step_count
    for e in range(args.total_epoch):
        model.train()
        losses = []
        for img, _ in tqdm(iter(dataloader)):
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            step_count += 1  # Increment step_count correctly
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)

        # Save model
        torch.save(model.state_dict(), args.model_path)
