import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import random

from Conv_model import Autoencoder  # Autoencoder model import

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Dataset path setting and loading
dataset = datasets.ImageFolder(root='../../data/oxford_custom', transform=transform)
data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

# Model loading
model = Autoencoder()
model.load_state_dict(torch.load('../model/Conv_models.pth'))
model.eval()

# Load data and labels, image paths
for inputs, labels in data_loader:
    paths = data_loader.dataset.samples
    encoded, _ = model(inputs)
    features = encoded.view(inputs.size(0), -1).detach().numpy()

# Dimensionality reduction with t-SNE
tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(features)

# K-means clustering on t-SNE results
kmeans = KMeans(n_clusters=37, random_state=0)
clusters = kmeans.fit_predict(tsne_results)

# Select a cluster number
selected_cluster = 10

# Save image paths from the selected cluster
selected_image_paths = [paths[i][0] for i in range(len(clusters)) if clusters[i] == selected_cluster]

# Randomly select 10 images
random_selected_images = random.sample(selected_image_paths, 10)

# Check and create the subfolder for saving
save_folder = 'check_data'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Save selected image names and paths to a file
with open(os.path.join(save_folder, 'selected_images.txt'), 'w') as file:
    for path in random_selected_images:
        file.write(path + '\n')

print(f"Selected image paths from cluster {selected_cluster} have been saved to {save_folder}/selected_images.txt.")
