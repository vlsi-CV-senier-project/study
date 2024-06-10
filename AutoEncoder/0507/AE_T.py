import torch
import torch.nn as nn
from torchvision.transforms import v2

# Patch Embedding definition adjusted for 64x64 input
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # B, E, P, P
        x = x.flatten(2)  # B, E, N
        x = x.transpose(1, 2)  # B, N, E
        return x

# test
img_size = 64
patch_size = 4
in_chans = 3
embed_dim = 192
x = torch.randn(1, in_chans, img_size, img_size)
# model = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
# out = model(x)
# print(out.shape)

# Adjust TransformerAutoEncoder for 64x64 input
class TransformerAutoEncoder(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=3, embed_dim=192, num_heads=12, num_layers=12, dropout_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, in_chans * patch_size * patch_size)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        memory = self.encoder(x)
        x = self.decoder(x, memory)
        # print("before proj",x.shape)  1*256*192
        x = self.proj(x)
        x = x.reshape(x.shape[0], in_chans, img_size, img_size)   
        # print("after proj",x.shape)   1*3*64*64
        return x

model = TransformerAutoEncoder()
out = model(x)
print(out.shape)