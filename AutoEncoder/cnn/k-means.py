# k_means.py
# 사전에 군집의 개수를 설정해야한다!
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from sklearn.cluster import KMeans
#from Conv_model import Autoencoder  # 모델 구조가 정의된 파일을 임포트
#from MLP_model import Autoencoder
from vit_model import Autoencoder

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA  # PCA를 위한 임포트 추가

# 데이터 로딩 및 전처리
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # MLP에서 need
    # transforms.Grayscale(num_output_channels=1),
    
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dataset = datasets.OxfordIIITPet(root = '../../data', download = True, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 모델 로드
model = Autoencoder()

# .pth파일 이름 수정할 것
model.load_state_dict(torch.load('../model/Vit_models.pth'))
model.eval()

# 특징 추출 MLP
# features = []
# for inputs, _ in data_loader:
#     inputs = inputs.view(inputs.size(0), -1)  # 입력 데이터를 평탄화 (128*128*3 = 49152)
#     encoded, _ = model(inputs)
#     features.append(encoded.detach().numpy())
# features = np.concatenate(features, axis=0)

# 특징 추출Conv
features = []
for inputs, _ in data_loader:
    encoded, _ = model(inputs)
    features.append(encoded.view(encoded.size(0), -1).detach().numpy())  # 평탄화하여 numpy 배열로 변환

features = np.concatenate(features, axis=0)

# K-means 군집화 수행
kmeans = KMeans(n_clusters=37, random_state=0)
clusters = kmeans.fit_predict(features)

# PCA를 이용하여 특징을 2차원으로 축소
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

# 시각화
plt.figure(figsize=(12, 10))

# 커스텀 색상 팔레트 생성
cmap = plt.get_cmap('viridis')  # 'viridis' 색상 맵 사용
colors = cmap(np.linspace(0, 1, 37))  # 37개의 고유한 색상을 생성

for i in range(37):  # 군집의 개수만큼 반복
    plt.scatter(features_2d[clusters == i, 0], features_2d[clusters == i, 1], color=colors[i], label=f'Cluster {i}')
plt.legend()
plt.title('K-means Clustering_vir with PCA visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
plt.savefig('kmeans_vit.png')
plt.close()  # 사용된 리소스 해제

print("The k_means plot has been saved as 'ox_pet_k-means_Visualization.png'.")


