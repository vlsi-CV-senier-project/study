import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from sklearn.cluster import KMeans
#from vit_model import Autoencoder  # Autoencoder 모델 구조가 정의된 파일을 임포트
#from Conv_model import Autoencoder
from MLP_model import Autoencoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 플로팅을 위한 임포트
from sklearn.decomposition import PCA

# 데이터 로딩 및 전처리
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    #transforms.Grayscale(num_output_channels=1),

])

dataset = datasets.OxfordIIITPet(root='../../data', download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 모델 로드
model = Autoencoder()
model.load_state_dict(torch.load('../model/models.pth'))
model.eval()

# 특징 추출
features = []
for inputs, _ in data_loader:
    encoded, _ = model(inputs)
    features.append(encoded.view(encoded.size(0), -1).detach().numpy())
    transforms.Grayscale(num_output_channels=1),
#
features = np.concatenate(features, axis=0)
# # 특징 추출 MLP
# features = []
# for inputs, _ in data_loader:
#     inputs = inputs.view(inputs.size(0), -1)  # 입력 데이터를 평탄화 (128*128*3 = 49152)
#     encoded, _ = model(inputs)
#     features.append(encoded.detach().numpy())
# features = np.concatenate(features, axis=0)

# K-means 군집화 수행
kmeans = KMeans(n_clusters=37, random_state=0)
clusters = kmeans.fit_predict(features)

# PCA를 이용하여 특징을 3차원으로 축소
pca = PCA(n_components=3)
features_3d = pca.fit_transform(features)

# 시각화
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')  # 3D 플롯 생성

colors = plt.get_cmap('viridis')(np.linspace(0, 1, 37))  # 37개의 고유한 색상 생성

for i in range(37):
    ax.scatter(features_3d[clusters == i, 0], features_3d[clusters == i, 1], features_3d[clusters == i, 2], 
               color=colors[i], label=f'Cluster {i}')

ax.set_title('K-means Clustering with 3D PCA Visualization')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
plt.legend()
plt.show()
plt.savefig('kmeans_mlp_3d.png')
plt.close()

print("The 3D K-means plot has been saved as 'kmeans_mlp_3d.png'.")
