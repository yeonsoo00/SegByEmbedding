import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
from sklearn.neighbors import LocalOutlierFactor

# Generate cluster centers evenly distributed on a sphere
def generate_spherical_points(n_clusters, dim):
    torch.manual_seed(42)  # Fix seed for reproducibility
    points = torch.randn(n_clusters, dim)
    return points / torch.norm(points, dim=1, keepdim=True)

class ClusterEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_clusters):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )
        self.cluster_centers = nn.Parameter(generate_spherical_points(num_clusters, embedding_dim))

    def forward(self, x):
        x = self.fc(x)
        x = x / torch.norm(x, dim=1, keepdim=True)  # Normalize
        return x

def clustering_loss(embeddings, labels, cluster_centers, num_clusters):
    intra_loss = 0
    inter_loss = 0
    cluster_embeddings = [embeddings[labels == i] for i in range(num_clusters)]
    
    # Intra-cluster compactness
    for i, cluster_embeds in enumerate(cluster_embeddings):
        if cluster_embeds.shape[0] > 1:
            intra_loss += torch.mean(1 - torch.cosine_similarity(cluster_embeds, cluster_centers[i].unsqueeze(0)))
    
    # Inter-cluster separation
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            inter_loss += torch.cosine_similarity(cluster_centers[i].unsqueeze(0), cluster_centers[j].unsqueeze(0))
    
    return intra_loss - 0.1 * inter_loss  # Adjust balance factor

def construct_embedding_space():
    # Construct Embedding Space
    num_clusters = 5
    points_per_cluster = 5
    embedding_dim = 3

    cluster_centers = generate_spherical_points(num_clusters, embedding_dim)

    # Generate points around each cluster center
    cluster_std = 0.05  # Spread of each cluster
    X = []
    y = []
    for i, center in enumerate(cluster_centers):
        points = center + cluster_std * torch.randn(points_per_cluster, embedding_dim)
        X.append(points)
        y.append(torch.full((points_per_cluster,), i))

    X = torch.cat(X)
    y = torch.cat(y)
    model = ClusterEmbedding(input_dim=embedding_dim, embedding_dim=embedding_dim, num_clusters=num_clusters)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Constructing Embedding Space")

    for epoch in range(100):
        optimizer.zero_grad()
        embeddings = model(X)

        loss = clustering_loss(embeddings, y, model.cluster_centers, num_clusters=num_clusters)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Visualize embeddings
    final_embeddings = model(X).detach().numpy()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(final_embeddings[:, 0], final_embeddings[:, 1], final_embeddings[:, 2], c=y.numpy(), cmap='jet')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Improved Cluster Embeddings on Sphere')
    plt.savefig('results/embedding/embeddingspace.png', dpi=fig.dpi)
    plt.close()

    return model


if __name__ == "__main__":
    embedding_dim = 3
    num_clusters = 5

    # Construct Embedding Space
    model = construct_embedding_space()

    # Load image and filter out background
    image = cv2.imread('/home/yec23006/projects/research/KneeGrowthPlate/Knee_GrowthPlate/Images/CCC_K05_hK_FL1_s1_shift3_So.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_white = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)  # Detect white areas
    _, binary_black = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)  # Detect black areas

    num_labels_w, labels_w, stats_w, _ = cv2.connectedComponentsWithStats(binary_white, connectivity=8)
    num_labels_b, labels_b, stats_b, _ = cv2.connectedComponentsWithStats(binary_black, connectivity=8)

    mask = np.ones_like(gray, dtype=np.uint8) * 255

    size_threshold = 5000 # Define size threshold (removing large background areas)

    # Remove large white areas
    for i in range(1, num_labels_w):  # Skip background label 0
        if stats_w[i, cv2.CC_STAT_AREA] > size_threshold:
            mask[labels_w == i] = 0  # Set large white areas to black

    # Remove large black areas
    for i in range(1, num_labels_b):  # Skip background label 0
        if stats_b[i, cv2.CC_STAT_AREA] > size_threshold:
            mask[labels_b == i] = 0  # Set large black areas to black
    filtered_image = cv2.bitwise_and(image, image, mask=mask)
    
    # save option
    plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    plt.savefig('results/embedding/filtered_img.png')
    plt.close()


    original_size = image.size
    image_data = np.array(image) / 255.0
    H, W, C = image_data.shape
    pixels = image_data.reshape(-1, C)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pixels = torch.tensor(pixels, dtype=torch.float32).to(device)
    

    with torch.no_grad():
        pixel_embeddings = model(pixels).cpu().numpy()
    # Detect outliers using Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    outlier_mask = lof.fit_predict(pixel_embeddings) == -1

    # Perform clustering on non-outlier embeddings
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixel_embeddings)
    labels[outlier_mask] = -1  # Assign outliers a special label

    segmented_image = labels.reshape(H, W)

    # Convert cluster labels back to original pixel values for visualization
    output_image = np.zeros((H, W, 3))
    unique_labels = np.unique(labels)
    fixed_colors = {
        0: [0, 0, 1],      # Blue
        1: [0, 1, 0],      # Green
        2: [1, 1, 0],      # Yellow
        3: [1, 0.5, 0],    # Orange
        4: [0.5, 0.5, 0],  # Olive
        -1: [1, 0, 0]      # Red (Outliers)
    }
    output_image = np.zeros((H, W, 3))
    for label, color in fixed_colors.items():
        output_image[segmented_image == label] = color  # Assign predefined colors


    # Resize back to original image size
    output_image_resized = Image.fromarray((output_image * 255).astype(np.uint8)).resize(original_size, Image.NEAREST)

    # Plot segmentation result
    plt.figure(figsize=(8, 8))
    plt.imshow(output_image_resized)
    plt.axis('off')
    plt.title('Pixel Classification Result with Outlier Detection')
    plt.savefig('/results/embedding/segmentation_result.png')
    plt.close()