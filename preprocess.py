import cv2
import numpy as np

def preprocess_image(image_path, block_size=32):
    """
    Load, resize, and divide the image into blocks.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    
    # Divide into non-overlapping blocks
    blocks = [
        image[i:i+block_size, j:j+block_size]
        for i in range(0, image.shape[0], block_size)
        for j in range(0, image.shape[1], block_size)
    ]
    return blocks
