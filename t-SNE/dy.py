import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# Define transformations for the images only
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize the images
    transforms.ToTensor(),          # Convert images to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Define a function to load and transform the dataset appropriately
def load_dataset():
    dataset = datasets.OxfordIIITPet(root='./data', split='trainval',
                                     target_types='category',
                                     transform=transform,  # Correctly use 'transform'
                                     download=True)
    return dataset

dataset = load_dataset()

# DataLoader for handling the dataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Function to extract and flatten the data from the DataLoader
def preprocess_data(dataloader):
    data, labels = [], []
    for images, targets in dataloader:
        images = images.view(images.size(0), -1)  # Flatten the images
        data.append(images)
        labels.append(targets)  # Directly append targets assuming they are tensors

    data = torch.cat(data, dim=0)
    labels = torch.cat(labels, dim=0)
    return data.numpy(), labels.numpy()

# Preprocess data
data, labels = preprocess_data(dataloader)

# Function to apply t-SNE and visualize results
def apply_tsne(parameters):
    tsne = TSNE(**parameters)
    result = tsne.fit_transform(data)  # Use the preprocessed and flattened data

    plt.scatter(result[:, 0], result[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar()
    title_str = f"t-SNE: Perplexity={parameters['perplexity']}, LR={parameters.get('learning_rate', 'auto')}"
    plt.title(title_str)
    
    # Save the plot with dynamic filename reflecting the parameters used
    filename = f"tsne_perplexity_{parameters['perplexity']}_LR_{parameters.get('learning_rate', 'auto')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as '{filename}'.")

params_list = [
    # Variation 1: Default settings with slight adjustment in perplexity
    {'n_components': 2, 'perplexity': 20, 'learning_rate': 'auto', 'early_exaggeration': 12, 'init': 'pca'},
    # Variation 2: Higher perplexity and early exaggeration
    {'n_components': 2, 'perplexity': 50, 'learning_rate': 'auto', 'early_exaggeration': 24, 'init': 'pca'},
    # Variation 3: Lower learning rate with random initialization
    {'n_components': 2, 'perplexity': 30, 'learning_rate': 50, 'early_exaggeration': 12, 'init': 'random'},
    # Variation 4: Very high perplexity with higher learning rate
    {'n_components': 2, 'perplexity': 70, 'learning_rate': 300, 'early_exaggeration': 12, 'init': 'pca'}
]

# Function to apply K-means clustering
def apply_kmeans(num_clusters):
    # Apply t-SNE to reduce dimensionality
    tsne_1 = TSNE(n_components=2, random_state=42)
    data_reduced = tsne_1.fit_transform(data)

    # # Apply PCA
    # pca = PCA(n_components=2)
    # data_reduced = pca.fit_transform(data)

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(data_reduced)

    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'K-Means Clustering with {num_clusters} Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(f'new_kmeans_{num_clusters}_clusters.png')
    plt.close()
    print(f"K-Means clustering plot saved as 'kmeans_{num_clusters}_clusters.png'.")

# Function to apply DBSCAN clustering
def apply_dbscan(eps, min_samples):
    # # Apply t-SNE to reduce dimensionality
    # tsne_2 = TSNE(n_components=2, random_state=42)
    # data_reduced = tsne_2.fit_transform(data)

    # Apply PCA
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(data)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data_reduced)

    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(f'pca_dbscan_eps{eps}_min_samples{min_samples}.png')
    plt.close()
    print(f"DBSCAN clustering plot saved as 'dbscan_eps{eps}_min_samples{min_samples}.png'.")

# # Apply t-SNE with various settings
# for params in params_list:
#     apply_tsne(params)

# Apply K-means with different numbers of clusters
apply_kmeans(37)

# Apply DBSCAN with varying eps and min_samples
# apply_dbscan(eps=5, min_samples=5)






