import cv2
import numpy as np
from sklearn.decomposition import PCA

def apply_pca(blocks):
    """
    Apply PCA to each block and return eigenvalues as features.
    """
    pca_features = []
    for block in blocks:
        pca = PCA(n_components=1)
        reshaped_block = block.flatten().reshape(-1, 1)
        pca.fit(reshaped_block)
        pca_features.append(pca.explained_variance_ratio_[0])
    return np.array(pca_features)

def apply_dct(blocks):
    """
    Apply DCT to each block and return top DCT coefficients.
    """
    dct_features = []
    for block in blocks:
        dct_block = cv2.dct(np.float32(block))
        sorted_dct = np.sort(dct_block.flatten())
        dct_features.append(sorted_dct[:10])  # Retain top 10 coefficients
    return np.array(dct_features)
