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
from sklearn.linear_model import LogisticRegression
from scipy.ndimage import binary_fill_holes, label
from utils_roi import *
from utils_patching import *
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim1', type=int, default=5, help='Dimension of embedding space for selecting Knee growth plate')
    parser.add_argument('--save2', type=str, default='/home/yec23006/projects/research/KneeGrowthPlate/Embedding/results/', help='Save visualization and results')
    parser.add_argument('--vis', type=int, default=1, help='Visualization option (1:True, 0:False)')
    parser.add_argument('--cluster_idx', type=int, default=1, help='0 : Blue, 1 : Brown, 2: Green, 3 : Mint, 4 : Yellow')

    args = parser.parse_args()
    dim1 = args.dim1
    save2 = args.save2
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
    model1 = ClusterEmbedding(input_dim=3, embedding_dim=Embedding_dim, num_clusters=len(colors))
    optimizer = optim.Adam(model1.parameters(), lr=0.01)

    for epoch in range(200):
        optimizer.zero_grad()
        embeddings = model1(X)
        loss = contrastive_loss(embeddings, y, model1.cluster_centers, weight=3.0) 
        loss.backward()
        optimizer.step()

    if vis_option: # plot a embedding space with fixed color vectors
        visualize_RGBspace(X, y)
    
    # Segment ROI
    with torch.no_grad():
        pixel_embeddings = model1(image_pixels_tensor)
    
    cluster_center = model1.cluster_centers[cluster_idx]
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

    # Separate two components from final_selected_mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_selected_mask, connectivity=8)
    object_stats = stats[1:]  
    object_centroids = centroids[1:]

    kgp_y_idx = np.argmin(object_stats[:, 1])
    selected_object = (labels == (kgp_y_idx + 1)).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(save2, "Growth_plate_mask.png", selected_object))

    ##################################Patch Classification##########################################

    # Get overlap image of growth plate
    growth_plate_img = image_np*(selected_object[:, :, np.newaxis]//255)
    cv2.imwrite(os.path.join(save2, "Growth_plate.png", growth_plate_img))
    
    # Select coordinates for Patch Embedding space
    set_seed(42)
    columnar = image[6900:7050, 5100:5500, :]
    noncolumnar = image[6700:6800, 5100:5500, :]

    num_patches = 100
    patch_size = 64
    columnar_patches = patch_extractor(columnar, num_patches, patch_size)
    noncolumnar_patches = patch_extractor(noncolumnar, num_patches, patch_size)

    all_patches = np.vstack((columnar_patches, noncolumnar_patches))
    labels = np.array([1]*len(columnar_patches) + [0]*len(noncolumnar_patches)) # 1 : columnar, 0 : noncolumnar
    all_patches = all_patches.astype(np.float32)/255.0
    
    # Constructing Patch Embedding space
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = patch_size * patch_size * 3
    num_clusters = 2
    patch_embedding_dim = 10

    model2 = PatchEmbedding(input_dim, patch_embedding_dim, num_clusters).to(device)
    optimizer = optim.Adam(model2.parameters(), lr=0.001)

    data_tensor = torch.tensor(all_patches).to(device)
    label_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    data_loader = DataLoader(TensorDataset(data_tensor, label_tensor), batch_size=32, shuffle=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        for batch_data, batch_labels in data_loader:
            optimizer.zero_grad()
            embeddings = model2(batch_data.view(batch_data.size(0), -1))
            loss = contrastive_loss(embeddings, batch_labels, model2.cluster_centers)
            loss.backward()
            optimizer.step()
    
    with torch.no_grad():
        embeddings = model2(data_tensor.view(data_tensor.size(0), -1)).cpu().numpy()
    
    if vis_option:
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
    
    # Extract patches
    filtered_patches, filtered_positions = extract_object_patches(path2img, save_dir = os.path.join(save2))
    data_tensor_eval = torch.tensor(filtered_patches).to(device)
    data_tensor_eval = data_tensor_eval.to(torch.float32)/255.0
    data_tensor_eval = data_tensor_eval.permute(0, 3, 1, 2)
    data_tensor_eval = data_tensor_eval.reshape(data_tensor_eval.shape[0], -1)  

    model2.eval()
    with torch.no_grad():
        embeddings_eval = model2(data_tensor_eval).cpu().numpy()
    
    if vis_option:
        all_embeddings = np.vstack((embeddings, embeddings_eval))
        pca = PCA(n_components=2)
        reduced_all_embeddings = pca.fit_transform(all_embeddings)

        reduced_embeddings = reduced_all_embeddings[:len(embeddings)]  # Reference embeddings
        reduced_embeddings_eval = reduced_all_embeddings[len(embeddings):]  # Patch embeddings

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_embeddings[:num_patches, 0], reduced_embeddings[:num_patches, 1], c='r', label='Columnar')
        plt.scatter(reduced_embeddings[num_patches:, 0], reduced_embeddings[num_patches:, 1], c='b', label='Non-Columnar')
        plt.scatter(reduced_embeddings_eval[:, 0], reduced_embeddings_eval[:, 1], c='g', label='Patches')
        plt.legend()
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title("Patch Embeddings Visualization (Fixed PCA)")
        plt.savefig(os.path.join(save2, 'patch_embedding_vis.png'))
        plt.close()

    # Columnar/Noncolumnar Classifier
    classifier = LogisticRegression()
    classifier.fit(embeddings, labels) # Train
    predicted_labels = classifier.predict(embeddings_eval) # Eval

    

    # Patch reconstruction to original position
    reconstructed_image = reconstruct_image_from_patches(path2img, filtered_patches, filtered_positions, predicted_labels)
    reconstructed_image = cv2.cvtColor(reconstructed_image)
    cv2.imwrite(os.path.join(save2, "reconstructed_patches.png"), reconstructed_image)