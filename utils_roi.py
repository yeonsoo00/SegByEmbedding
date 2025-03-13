import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from skimage.color import rgb2lab
import random

random.seed(42)

# Pre-define RGB color vectors for embedding space
colors = [torch.tensor(group, dtype = torch.float32)/255.0 for group in [
    np.array([[71, 69, 130], [52, 40, 127], [189, 179, 201], [144, 139, 215], [93, 85, 133]]),  # Blue
    np.array([[173, 109, 66], [154, 94, 50], [202, 144, 97], [206, 130, 63], [161, 119, 94]]),  # Brown
    np.array([[67, 103, 62], [100, 117, 71], [112, 141, 109], [83, 97, 61], [104, 127, 121]]),  # Green
    np.array([[160, 247, 172], [191, 227, 205], [162, 213, 184], [175, 226, 161], [164, 210, 182]]), #Mint
    np.array([[238, 225, 156], [245, 207, 136], [200, 177, 127], [255, 254, 174], [251, 240, 113]])  # Yellow
]]

# Load rgb-image
def load_data(path2img):
    image = cv2.imread(path2img, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
    return image

# RGB vector embedding space
class ClusterEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_clusters):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256), 
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128), 
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64), 
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, embedding_dim)
        )
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, embedding_dim))

    def forward(self, x):
        x = self.fc(x)
        x = x / torch.norm(x, dim=1, keepdim=True)  # Normalize embeddings
        return x

def contrastive_loss(embeddings, labels, cluster_centers, margin=3.0):  # Increased margin
    intra_loss = 0  
    inter_loss = 0  
    num_clusters = cluster_centers.shape[0]
    
    # Intra-cluster compactness: Pull embeddings closer to their assigned cluster center
    for i in range(num_clusters):
        cluster_embeds = embeddings[labels == i]
        if cluster_embeds.shape[0] > 1:
            intra_loss += torch.mean(1 - torch.cosine_similarity(cluster_embeds, cluster_centers[i].unsqueeze(0)))

    # Inter-cluster separation: Push clusters apart based on similarity
    num_pairs = 0
    for i in range(num_clusters):
        for j in range(num_clusters):
            if i != j:
                similarity = torch.cosine_similarity(cluster_centers[i].unsqueeze(0), cluster_centers[j].unsqueeze(0))
                distance_penalty = torch.exp(-torch.norm(cluster_centers[i] - cluster_centers[j]))  # Exponential scaling
                inter_loss += torch.clamp(margin - similarity, min=0) * distance_penalty  # Stronger push

                num_pairs += 1

    inter_loss /= num_pairs  # Normalize
    return intra_loss + 3.0 * inter_loss  # Increase weight of inter-cluster loss


def visualize_RGBspace(X, y):
    cluster_colors = np.array([
            [71, 69, 130],   # Blue
            [173, 109, 66],  # Brown
            [67, 103, 62],   # Green
            [160, 247, 172], # Mint
            [238, 225, 156]  # Yellow
        ]) / 255.0

    point_colors = np.array([cluster_colors[label] for label in y.numpy()])

    pca = PCA(n_components=3)
    projected_embeddings = pca.fit_transform(model(X).detach().numpy())

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(projected_embeddings[:, 0], projected_embeddings[:, 1], projected_embeddings[:, 2], color=point_colors, s=50)
    ax.save(os.path.join(save2, 'EmbeddingSpaceRGB.png'))
    