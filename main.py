import os
import sys
import argparse
import numpy as np
import cv2

from preprocess import preprocess_image
from feature_extraction import apply_pca, apply_dct
from visualization import highlight_forged_regions, plot_feature_distributions

def run_detection(image_path="archit6.jpg", block_size=16, save_plot=False, show_gui=False):
    if not os.path.exists(image_path):
        # Check samples folder
        alt_path = os.path.join("samples", os.path.basename(image_path))
        if os.path.exists(alt_path):
            image_path = alt_path
        else:
            raise FileNotFoundError(f"Input image not found: {image_path}")

    print(f"[*] Processing image: {image_path}")
    print(f"[*] Block size: {block_size}x{block_size}")

    # Step 1: Preprocess image into blocks
    blocks = preprocess_image(image_path, block_size=block_size)
    print(f"[*] Total image blocks extracted: {len(blocks)}")

    # Step 2: Extract features and normalize
    pca_features = apply_pca(blocks)
    dct_features = apply_dct(blocks)

    # Normalize features
    pca_features = (pca_features - pca_features.mean()) / (pca_features.std() + 1e-8)
    dct_features = (dct_features - dct_features.mean()) / (dct_features.std() + 1e-8)

    # Step 3: Dynamic thresholding
    pca_threshold = pca_features.mean() + 1.0 * pca_features.std()
    dct_threshold = dct_features.mean() + 1.0 * dct_features.std()

    # Detect forged regions
    forged_blocks = []
    for i, (pca_feat, dct_feat) in enumerate(zip(pca_features, dct_features)):
        if np.any(pca_feat > pca_threshold) or np.any(dct_feat > dct_threshold):
            forged_blocks.append(i)

    print(f"[+] Detected candidate forged blocks: {len(forged_blocks)}")

    # Step 4: Visualize and save results
    os.makedirs("results", exist_ok=True)
    if forged_blocks:
        highlighted_image = highlight_forged_regions(image_path, forged_blocks, block_size=block_size)
        output_path = os.path.join("results", f"highlighted_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, highlighted_image)
        print(f"[✓] Output saved to: {output_path}")

        if save_plot:
            plot_feature_distributions(pca_features, dct_features, pca_threshold, dct_threshold, show=False)

        if show_gui:
            cv2.imshow("Forged Regions", highlighted_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("[!] No forged regions detected above threshold.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy-Move Image Forgery Detection via PCA & DCT")
    parser.add_argument("--image", type=str, default="archit6.jpg", help="Path to input image")
    parser.add_argument("--block-size", type=int, default=16, help="Block size (default: 16)")
    parser.add_argument("--plot", action="store_true", help="Generate and save feature distribution plots")
    parser.add_argument("--gui", action="store_true", help="Display interactive window")
    args = parser.parse_args()

    run_detection(image_path=args.image, block_size=args.block_size, save_plot=args.plot, show_gui=args.gui)
