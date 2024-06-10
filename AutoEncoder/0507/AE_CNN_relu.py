import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),  # 추가된 층
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 기존 층
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),  # 추가된 층
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 기존 층
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),  # 추가된 층
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 기존 층
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1),  # 추가된 층
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),  # 기존 층
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

