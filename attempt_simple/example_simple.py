"""
Simplified glacier detection using traditional computer vision
Good for cases where glaciers have distinct color/brightness
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology


def simple_glacier_detection(image_path):
    """
    Simplified glacier detection using color and brightness thresholds
    """
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to HSV and LAB color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Glaciers are typically:
    # - High brightness (L channel in LAB)
    # - Low saturation (S channel in HSV)
    # - Bluish-white color

    l_channel = lab[:, :, 0]
    s_channel = hsv[:, :, 1]

    # Threshold for glacier detection
    brightness_threshold = np.percentile(l_channel, 70)  # Top 30% brightness
    glacier_mask = (l_channel > brightness_threshold) & (s_channel < 80)

    # Clean up mask
    glacier_mask = morphology.remove_small_objects(glacier_mask, min_size=1000)
    glacier_mask = morphology.remove_small_holes(glacier_mask, area_threshold=1000)
    glacier_mask = ndimage.binary_fill_holes(glacier_mask)

    # Find contours
    contours = measure.find_contours(glacier_mask.astype(float), 0.5)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(glacier_mask, cmap='gray')
    axes[1].set_title('Glacier Mask')
    axes[1].axis('off')

    axes[2].imshow(image_rgb)
    for contour in contours:
        axes[2].plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
    axes[2].set_title('Detected Borders')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('simple_glacier_detection.png', dpi=300)
    plt.show()

    return glacier_mask, contours


if __name__ == "__main__":
    mask, contours = simple_glacier_detection("glacier.png")