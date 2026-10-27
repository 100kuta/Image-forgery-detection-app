import cv2
import numpy as np
import matplotlib.pyplot as plt

def highlight_forged_regions(image_path, forged_blocks, block_size=32):
    """
    Highlight forged regions on the original image with bounding boxes.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.resize(image, (512, 512))
    
    grid_cols = 512 // block_size
    for idx in forged_blocks:
        row = (idx // grid_cols) * block_size
        col = (idx % grid_cols) * block_size
        cv2.rectangle(image, (col, row), (col + block_size, row + block_size), (0, 0, 255), 2)
    
    return image

def plot_feature_distributions(pca_features, dct_features, pca_threshold, dct_threshold, show=True):
    """
    Plot histogram distributions of PCA and DCT features with threshold lines.
    """
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(pca_features.flatten(), bins=50, alpha=0.7, color='steelblue', label="PCA Features")
    plt.axvline(pca_threshold, color='crimson', linestyle='--', label=f"PCA Thresh ({pca_threshold:.2f})")
    plt.title("PCA Feature Distribution")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(dct_features.flatten(), bins=50, alpha=0.7, color='seagreen', label="DCT Features")
    plt.axvline(dct_threshold, color='crimson', linestyle='--', label=f"DCT Thresh ({dct_threshold:.2f})")
    plt.title("DCT Feature Distribution")
    plt.legend()
    
    plt.tight_layout()
    if show:
        plt.show()
