import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# MNIST 데이터셋 로드
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
mnist_loader = torch.utils.data.DataLoader(dataset=mnist_dataset, batch_size=len(mnist_dataset), shuffle=False)

# 데이터를 numpy 배열로 변환
data, labels = next(iter(mnist_loader))
data = data.view(data.size(0), -1).numpy()

# t-SNE 모델 생성
tsne = TSNE(n_components=2, perplexity=30, random_state=42)

# 데이터를 2차원으로 축소
reduced_data = tsne.fit_transform(data)

# 결과 시각화 및 저장
plt.figure(figsize=(12, 10))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for i in range(10):
    mask = labels == i
    plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], s=10, color=colors[i], label=str(i))
plt.legend()
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('t-SNE Visualization of MNIST Dataset')

# 파일로 저장
plt.savefig('MNIST_t-SNE_Visualization.png')
plt.close()  # 사용된 리소스 해제

print("The t-SNE plot has been saved as 'MNIST_t-SNE_Visualization.png'.")
