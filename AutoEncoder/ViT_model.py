import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = ViT(image_size= 224, num_classes= 14*14)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 64, 2, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        #y = inputs.view(-1,128*128).to(DEVICE)
        #print("input shape",x.size())
        encoded = self.encoder(x)
        encoded = encoded.view(-1,1,14,14)
        #print("encoded shape",encoded.size())
        decoded = self.decoder(encoded)
        decoded = decoded.view(-1,3,224,224)
        return encoded, decoded
    


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