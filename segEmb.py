import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import binary_fill_holes, label
from utils_roi import *
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim1', type=int, default=5, help='Dimension of embedding space for selecting Knee growth plate')
    parser.add_argument('--save2', type=str, default='/home/yec23006/projects/research/KneeGrowthPlate/Embedding/results/', help='Save visualization and results')
    parser.add_argument('--vis', type=int, default=1, help='Visualization option (1:True, 0:False)')
    parser.add_argument('--cluster_idx', type=int, default=1, help='0 : Blue, 1 : Brown, 2: Green, 3 : Mint, 4 : Yellow')

    args = parser.parse_args()
    dim1 = arg.dim1
    save2 = arg.save2
    vis_option = args.vis
    vis_option = True if vis_option == 1 else False
    cluster_idx = args.cluster_idx

    ##################################ROI Detection##########################################
    # Load data
    path2img = '/home/yec23006/projects/research/KneeGrowthPlate/Knee_GrowthPlate/Images/CCC_K05_hK_FL1_s1_shift3_So.jpg'
    image = load_data(path2img)
    image_np = np.array(image)
    image_pixels = image_np.reshape(-1, 3) / 255.0
    image_pixels_tensor = torch.tensor(image_pixels, dtype=torch.float32)

    # Construct Embedding Space for selecting the growth plate
    X = torch.cat(colors) 
    y = torch.cat([torch.full((len(group),), i) for i, group in enumerate(colors)])
    X = X / torch.norm(X, dim=1, keepdim=True)

    Embedding_dim = 5
    model1 = ClusterEmbedding(input_dim=3, embedding_dim=embedding_dim, num_clusters=len(colors))
    optimizer = optim.Adam(model1.parameters(), lr=0.01)

    for epoch in range(200):
        optimizer.zero_grad()
        embeddings = model(X)
        loss = contrastive_loss(embeddings, y, model.cluster_centers) 
        loss.backward()
        optimizer.step()

    if vis_option: # plot a embedding space with fixed color vectors
        visualize_RGBspace(X, y)
    
    # Segment ROI
    with torch.no_grad():
        pixel_embeddings = model1(image_pixels_tensor)
    
    cluster_center = model.cluster_centers[cluster_idx]
    cosine_similarities = torch.cosine_similarity(pixel_embeddings, cluster_center.unsqueeze(0), dim=1)

    color_threshold = 0.9
    roi_mask = (cosine_similarities > color_threshold).numpy().reshape(image_np.shape[:2])
    roi_mask_binary = (roi_mask * 255).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask_denoised = cv2.morphologyEx(roi_mask_binary, cv2.MORPH_OPEN, kernel, iterations=2)

    segmented_roi = np.zeros_like(image_np)
    segmented_roi[mask_denoised == 255] = image_np[mask_denoised == 255]

    # Generate Mask
    closed_mask = cv2.morphologyEx(mask_denoised, cv2.MORPH_CLOSE, kernel, iterations=10)
    morph_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    filled_mask = binary_fill_holes(morph_mask).astype(np.uint8)
    labeled_mask, num_features = label(filled_mask)

    unique, counts = np.unique(labeled_mask, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]

    if len(sorted_indices) > 2:
        largest_label = unique[sorted_indices[1]]
        second_largest_label = unique[sorted_indices[2]]

        # Create a mask with only the two largest objects
        final_selected_mask = np.logical_or(
            labeled_mask == largest_label, labeled_mask == second_largest_label
        ).astype(np.uint8) * 255
    else:
        final_selected_mask = filled_mask * 255

    final_selected_mask = cv2.morphologyEx(final_selected_mask, cv2.MORPH_CLOSE, kernel)
    final_selected_mask = cv2.morphologyEx(final_selected_mask, cv2.MORPH_OPEN, kernel)
    _, final_selected_mask = cv2.threshold(final_selected_mask, 127, 255, cv2.THRESH_BINARY)

    cv2.imwrite(os.path.join(save2, "ROI_mask.png"), final_selected_mask)
    print("ROI Mask saved")

    ##################################Patch Classification##########################################
