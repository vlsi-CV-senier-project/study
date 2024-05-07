import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 예: 입력 채널 3, 출력 채널 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 크기를 반으로 줄임
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        print("Size_____")
        print(x.size())
        encoded = self.encoder(x)
        print(encoded.size())
        decoded = self.decoder(encoded)
        print(decoded.size())
        return encoded, decoded
