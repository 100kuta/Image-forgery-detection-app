import cv2
import numpy as np
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# Adjustable Parameters
BLOCK_SIZE = 16          # Size of the blocks to divide the image
STEP = 8                 # Step size for overlapping blocks
PCA_COMPONENTS = 64      # Number of PCA components to retain
DUPLICATE_THRESHOLD = 0.1  # Threshold for duplicate detection
MIN_DISTANCE = 20        # Minimum distance between matched blocks to filter false positives


# Functions
def preprocess_image(image_path):
    """Load and convert the image to grayscale."""
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray_image


def divide_into_blocks(image, block_size, step):
    """Divide the image into overlapping blocks."""
    blocks = []
    coordinates = []
    for i in range(0, image.shape[0] - block_size + 1, step):
        for j in range(0, image.shape[1] - block_size + 1, step):
            block = image[i:i + block_size, j:j + block_size]
            blocks.append(block)
            coordinates.append((i, j))
    return np.array(blocks), coordinates


def apply_dct(blocks):
    """Apply DCT to each block."""
    dct_features = [dct(dct(block.T, norm='ortho').T, norm='ortho') for block in blocks]
    return np.array(dct_features)


def apply_pca(features, n_components):
    """Apply PCA to reduce dimensionality."""
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features.reshape(features.shape[0], -1))
    return pca_features


def detect_duplicates(features, threshold):
    """Detect duplicate blocks based on Euclidean distance."""
    sorted_indices = np.lexsort(features.T)
    sorted_features = features[sorted_indices]
    duplicate_pairs = []
    for i in range(len(sorted_features) - 1):
        if euclidean(sorted_features[i], sorted_features[i + 1]) < threshold:
            duplicate_pairs.append((sorted_indices[i], sorted_indices[i + 1]))
    return duplicate_pairs


def filter_duplicates(pairs, coordinates, min_distance):
    """Filter false positives based on minimum spatial distance."""
    filtered_pairs = []
    for idx1, idx2 in pairs:
        coord1 = coordinates[idx1]
        coord2 = coordinates[idx2]
        if np.linalg.norm(np.array(coord1) - np.array(coord2)) > min_distance:
            filtered_pairs.append((coord1, coord2))
    return filtered_pairs


def highlight_regions(image, pairs, block_size):
    """Highlight the detected duplicated regions on the original image."""
    highlighted_image = image.copy()
    for (coord1, coord2) in pairs:
        cv2.rectangle(highlighted_image, coord1, (coord1[0] + block_size, coord1[1] + block_size), (0, 255, 0), 2)
        cv2.rectangle(highlighted_image, coord2, (coord2[0] + block_size, coord2[1] + block_size), (0, 255, 0), 2)
    return highlighted_image


def main(image_path):
    # Step 1: Preprocess the image
    original_image, gray_image = preprocess_image(image_path)

    # Step 2: Divide the image into blocks
    blocks, coordinates = divide_into_blocks(gray_image, BLOCK_SIZE, STEP)

    # Step 3: Apply DCT
    dct_features = apply_dct(blocks)

    # Step 4: Apply PCA
    pca_features = apply_pca(dct_features, PCA_COMPONENTS)

    # Step 5: Detect duplicates
    duplicate_pairs = detect_duplicates(pca_features, DUPLICATE_THRESHOLD)

    # Step 6: Filter duplicates
    filtered_pairs = filter_duplicates(duplicate_pairs, coordinates, MIN_DISTANCE)

    # Step 7: Highlight regions
    highlighted_image = highlight_regions(original_image, filtered_pairs, BLOCK_SIZE)

    # Display results
    print(f"Detected forged blocks: {filtered_pairs}")
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Detected Forged Regions')
    plt.show()


if __name__ == "__main__":
    # Replace 'image_path.jpg' with the path to your test image
    main("beach_wood_copy.png")
