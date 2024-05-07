import torch
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 데이터셋 불러오기
dataset = OxfordIIITPet(root='data/', download=True, transform=transforms.Compose([
    transforms.Resize((64, 64)),  # 이미지 크기 조정
    transforms.ToTensor()         # 이미지를 텐서로 변환
]))

# 데이터셋에서 이미지와 라벨 추출
images = []
labels = []
for i in range(len(dataset)):
    img, label = dataset[i]
    images.append(img)
    labels.append(label)

# 이미지 텐서를 NumPy 배열로 변환
images_np = torch.stack(images).numpy()
images_np = images_np.reshape(len(images), -1)  # t-SNE를 위해 2D로 변환

# t-SNE 적용
tsne = TSNE(n_components=2, random_state=0)
results = tsne.fit_transform(images_np)

# 결과 시각화
plt.figure(figsize=(12, 10))
scatter = plt.scatter(results[:, 0], results[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('t-SNE visualization of OxfordIIITPet Dataset')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

# Plot을 이미지 파일로 저장
plt.savefig('tsne_visualization.png')
plt.close()  # 현재 피규어를 닫음

print("The plot was saved as 'tsne_visualization.png'.")
