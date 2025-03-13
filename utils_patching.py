import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed=42):
    random.seed(seed)                 
    np.random.seed(seed)                
    torch.manual_seed(seed)             
    torch.cuda.manual_seed(seed)        
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Patch extractor for Constructing the embedding space
def patch_extractor(img, num_patches=100, patch_size=64, intensity_threshold=10):
    h, w, _ = img.shape
    patches = []

    for _ in range(num_patches):
        y= np.random.randint(0, h - patch_size + 1)
        x= np.random.randint(0, w - patch_size + 1)
        patch = img[y:y+patch_size, x:x+patch_size, :]

        if np.mean(patch) > intensity_threshold :
            patches.append(patch)
    
    return np.array(patches)

class PatchEmbedding(nn.Module):
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
        x = x / torch.norm(x, dim=1, keepdim=True)
        return x
    

def extract_object_patches(image_path, patch_size=64, stride=32, intensity_threshold=10, save_dir="/home/yec23006/projects/research/KneeGrowthPlate/Embedding/results/patch_extraction"):
    """
    Load an image, apply Otsu's thresholding, extract object-containing patches,
    and save them with their positions.
    
    Args:
        image_path (str): Path to the input image.
        patch_size (int): Size of each patch.
        stride (int): Step size for moving the patch.
        intensity_threshold (int): Minimum mean intensity to keep a patch.
        save_dir (str): Directory to save extracted patches and positions.
    
    Returns:
        filtered_patches_path (str): Path to saved filtered patches.
        filtered_positions_path (str): Path to saved patch positions.
    """

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(otsu_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_mask = np.zeros_like(gray)
    cv2.drawContours(object_mask, contours, -1, (255), thickness=cv2.FILLED)
    
    # Extract patches only from object regions
    h, w, _ = image.shape
    patches, positions = [], []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch_mask = object_mask[y:y+patch_size, x:x+patch_size]
            if np.mean(patch_mask) > 128:
                patch = image[y:y+patch_size, x:x+patch_size, :]
                patches.append(patch)
                positions.append((y, x))

    patches = np.array(patches)
    positions = np.array(positions)
    
    filtered_patches, filtered_positions = [], []
    for patch, position in zip(patches, positions):
        if np.mean(patch) > intensity_threshold:
            filtered_patches.append(patch)
            filtered_positions.append(position)

    filtered_patches = np.array(filtered_patches)
    filtered_positions = np.array(filtered_positions)
    
    # Save results
    filtered_patches_path = os.path.join(save_dir, "filtered_patches.npy")
    filtered_positions_path = os.path.join(save_dir, "filtered_patch_positions.npy")
    np.save(filtered_patches_path, filtered_patches)
    np.save(filtered_positions_path, filtered_positions)
    
    print(f"Total extracted patches: {len(filtered_patches)}")
    
    return filtered_patches, filtered_positions


def reconstruct_image_from_patches(image_shape, patches, positions, labels):
    """
    Reconstruct an image from patches with color-coded overlay based on classification labels.

    Args:
        image_shape (tuple): The shape of the original image (H, W, C).
        patches (np.array): Array of extracted patches.
        positions (list): List of (y, x) coordinates for each patch.
        labels (np.array): Classification labels for each patch (0: non-columnar, 1: columnar).

    Returns:
        reconstructed_image (np.array): Reconstructed image with color overlay.
    """
    h, w, c = image_shape
    patch_size = patches.shape[1]
    reconstructed_image = np.zeros((h, w, c), dtype=np.uint8)
    count_map = np.zeros((h, w), dtype=np.uint8)

    # Color mapping for labels
    columnar_color = np.array([255, 0, 0], dtype=np.uint8)   # Red for columnar
    non_columnar_color = np.array([0, 0, 255], dtype=np.uint8)  # Blue for non-columnar

    # for patch, (y, x), label in zip(patches, positions, labels):
    #     color_overlay = columnar_color if label == 1 else non_columnar_color

    #     # Blend original patch with label color overlay
    #     blended_patch = (0.5 * patch + 0.5 * color_overlay).astype(np.uint8)

    #     # Assign patch to the reconstructed image
    #     reconstructed_image[y:y+patch_size, x:x+patch_size] += blended_patch
    #     count_map[y:y+patch_size, x:x+patch_size] += 1

    # # Normalize overlapping areas by averaging
    # mask = count_map > 0
    # reconstructed_image[mask] //= count_map[mask, None]

    for (y, x), label in zip(positions, labels):
        color = columnar_color if label == 1 else non_columnar_color

        # Fill patch region with the respective color
        reconstructed_image[y:y+patch_size, x:x+patch_size] = color

    return reconstructed_image


