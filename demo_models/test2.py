# load resnet18 with the pre-trained weights
from torchvision import models
import torch

resnet50_pretrained = models.resnet50(pretrained=True)

print(resnet50_pretrained)
