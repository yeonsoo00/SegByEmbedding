import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

image = Image.open('/home/yec23006/projects/research/KneeGrowthPlate/Knee_GrowthPlate/Images/CCC_K05_hK_FL1_s1_shift3_So.jpg').convert('RGB')
image_np = np.array(image)

image_pixels = image_np.reshape(-1, 3) / 255.0
image_pixels_tensor = torch.tensor(image_pixels, dtype=torch.float32)

# Convert group colors to tensors and normalize them
colors = [torch.tensor(group, dtype = torch.float32)/255.0 for group in [
    np.array([[71, 69, 130], [52, 40, 127], [189, 179, 201], [144, 139, 215], [93, 85, 133]]),  # Blue
    np.array([[173, 109, 66], [154, 94, 50], [202, 144, 97], [206, 130, 63], [161, 119, 94]]),  # Brown
    np.array([[67, 103, 62], [100, 117, 71], [112, 141, 109], [83, 97, 61], [104, 127, 121]]),  # Green
    np.array([[160, 247, 172], [191, 227, 205], [162, 213, 184], [175, 226, 161], [164, 210, 182]]), #Mint
    np.array([[238, 225, 156], [245, 207, 136], [200, 177, 127], [255, 254, 174], [251, 240, 113]])  # Yellow
]]

# Flatten and store cluster labels
X = torch.cat(colors)
y = torch.cat([torch.full((len(group),), i) for i, group in enumerate(colors)])

# Normalize to unit sphere
X = X / torch.norm(X, dim=1, keepdim=True)

# Learnable CNN-based embeddings
embedding_dim = 5
class ClusterEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_clusters):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), 
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
                distance_penalty = torch.exp(-torch.norm(cluster_centers[i] - cluster_centers[j])) 
                inter_loss += torch.clamp(margin - similarity, min=0) * distance_penalty 

                num_pairs += 1

    inter_loss /= num_pairs  # Normalize
    return intra_loss + 3.0 * inter_loss 


model = ClusterEmbedding(input_dim=3, embedding_dim=embedding_dim, num_clusters=len(colors))
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    optimizer.zero_grad()
    embeddings = model(X)
    loss = contrastive_loss(embeddings, y, model.cluster_centers) 
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Visualize embeddings
# Define the original colors in RGB (normalized to [0,1])
cluster_colors = np.array([
    [71, 69, 130],   # Blue
    [173, 109, 66],  # Brown
    [67, 103, 62],   # Green
    [160, 247, 172], # Mint
    [238, 225, 156]  # Yellow
]) / 255.0  # Normalize to [0,1]

# Convert cluster labels to corresponding colors
point_colors = np.array([cluster_colors[label] for label in y.numpy()])

pca = PCA(n_components=3)
projected_embeddings = pca.fit_transform(model(X).detach().numpy())

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(projected_embeddings[:, 0], projected_embeddings[:, 1], projected_embeddings[:, 2], color=point_colors, s=50)
# plt.show()

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(projected_embeddings[:, 0], projected_embeddings[:, 1], projected_embeddings[:, 2], color=point_colors, s=50)

# # Rotate for different angles
# for angle in range(0, 360, 45):  
#     ax.view_init(elev=20, azim=angle)  # Change elevation and azimuth
#     plt.pause(0.5)  # Pause to update the plot

# plt.show()

