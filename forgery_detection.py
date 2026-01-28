from scipy.spatial.distance import euclidean

def detect_forgery(pca_features, dct_features, threshold=0.5):
    """
    Detect forgery by comparing PCA and DCT features of blocks.
    """
    forged_blocks = []
    num_blocks = len(pca_features)
    
    for i in range(num_blocks):
        for j in range(i+1, num_blocks):
            # Compare PCA and DCT features
            pca_dist = abs(pca_features[i] - pca_features[j])
            dct_dist = euclidean(dct_features[i], dct_features[j])
            
            # Mark as forged if both distances are below thresholds
            if pca_dist < threshold and dct_dist < threshold:
                forged_blocks.append(i)
                forged_blocks.append(j)
    
    return list(set(forged_blocks))  # Return unique forged block indices
