import numpy as np
from preprocess import preprocess_image
from feature_extraction import apply_pca, apply_dct
from forgery_detection import detect_forgery
from visualization import highlight_forged_regions
import cv2

# Input image
image_path = "archit6.JPG"

# Step 1: Preprocess image with smaller blocks
blocks = preprocess_image(image_path, block_size=16)

# Step 2: Extract features and normalize
pca_features = apply_pca(blocks)
dct_features = apply_dct(blocks)

# Normalize features
pca_features = (pca_features - pca_features.mean()) / pca_features.std()
dct_features = (dct_features - dct_features.mean()) / dct_features.std()

# Step 3: Dynamic thresholding
pca_threshold = pca_features.mean() + 1 * pca_features.std()
dct_threshold = dct_features.mean() + 1 * dct_features.std()

# Detect forged regions
forged_blocks = []
for i, (pca_feat, dct_feat) in enumerate(zip(pca_features, dct_features)):
    if np.any(pca_feat > pca_threshold) or np.any(dct_feat > dct_threshold):
        forged_blocks.append(i)

print("Detected forged blocks:", forged_blocks)

# Step 4: Visualize results
if forged_blocks:
    highlighted_image = highlight_forged_regions(image_path, forged_blocks, block_size=16)
    output_path = "results/highlighted_image.jpg"
    cv2.imwrite(output_path, highlighted_image)
    print(f"Output saved at {output_path}")
    
    # Display image with highlighted forgery
    cv2.imshow("Forged Regions", highlighted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No forged regions detected.")
