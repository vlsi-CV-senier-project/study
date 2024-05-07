import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from vit_model import Autoencoder  # Vision Transformer 모델 구조가 정의된 파일을 임포트
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA  # PCA를 위한 임포트 추가

# 데이터 로딩 및 전처리
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 데이터셋 경로 지정
dataset = datasets.ImageFolder(root='../../data/oxford_custom', transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 모델 로드
model = Autoencoder()
model.load_state_dict(torch.load('../model/Vit_models.pth'))
model.eval()

# 특징 추출
features = []
for inputs, _ in data_loader:
    encoded, _ = model(inputs)
    flat_encoded = encoded.view(encoded.size(0), -1).detach().numpy()  # 평탄화하여 numpy 배열로 변환
    features.append(flat_encoded)

# 모든 배치의 모양이 일관되도록 보장
consistent_features = [f for f in features if f.shape[1] == features[0].shape[1]]
if len(consistent_features) != len(features):
    print("Some batches have inconsistent feature sizes and were excluded.")
features = np.concatenate(consistent_features, axis=0)

# 계층적 군집화 수행
linked = linkage(features, method='ward')

# 37개의 군집으로 제한
clusters = fcluster(linked, t=37, criterion='maxclust')

# Dendrogram을 이용한 시각화
plt.figure(figsize=(12, 10))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.savefig('hierarchical_clustering_dendrogram2.png')
plt.show()
plt.close()  # 리소스 해제

# PCA를 이용하여 특징을 2차원으로 축소하고 각 군집을 시각화
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

# 군집화된 데이터의 색상 매핑
plt.figure(figsize=(12, 10))
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, len(np.unique(clusters))))
for i, cluster in enumerate(np.unique(clusters)):
    plt.scatter(features_2d[clusters == cluster, 0], features_2d[clusters == cluster, 1], color=colors[i], label=f'Cluster {cluster}')
plt.legend()
plt.title('Hierarchical Clustering with PCA visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig('h_Vit_pca.png')
plt.show()
plt.close()  # 리소스 해제

print("The PCA plot has been saved as 'hierarchical_clustering_pca.png'.")
