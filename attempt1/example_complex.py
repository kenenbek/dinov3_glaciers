"""
Glacier Border Detection from Satellite Imagery
Uses DINOv3 for feature extraction and clustering/segmentation for border detection
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import TorchAoConfig, AutoImageProcessor, AutoModel
from torchao.quantization import Int4WeightOnlyConfig
from sklearn.cluster import KMeans
from scipy import ndimage
from skimage import measure, morphology
import cv2


def load_and_preprocess_image(image_path):
    """Load the satellite image"""
    image = Image.open(image_path).convert('RGB')
    return image


def extract_features_with_dinov3(image, patch_size=16):
    """
    Extract dense features from DINOv3 model
    Returns features for each patch of the image
    """
    processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16plus-pretrain-lvd1689m")

    # Use smaller model for faster processing, or use the larger one you specified
    quant_type = Int4WeightOnlyConfig(group_size=128)
    quantization_config = TorchAoConfig(quant_type=quant_type)

    model = AutoModel.from_pretrained(
        "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config
    )

    # Process image
    inputs = processor(images=image, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

    # Get patch features (not just pooled output)
    # last_hidden_state has shape [batch, num_patches, hidden_dim]
    patch_features = outputs.last_hidden_state

    return patch_features, inputs.pixel_values.shape


def segment_glacier(patch_features, input_shape, n_clusters=3):
    """
    Segment the image into regions (glacier, rock, snow, etc.)
    using K-means clustering on DINOv3 features
    """
    # Remove batch dimension and CLS token, convert BFloat16 to float32 for numpy compatibility
    features = patch_features[0, 1:, :].float().cpu().numpy()  # Skip CLS token

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    # Calculate actual patch grid dimensions from input
    # DINOv3 uses 16x16 patches by default
    patch_size = 16
    batch_size, channels, height, width = input_shape

    # Calculate number of patches in each dimension
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    print(f"Expected grid: {num_patches_h}x{num_patches_w} = {num_patches_h * num_patches_w}")

    # Verify we have the right number of patches
    expected_patches = num_patches_h * num_patches_w
    if len(labels) != expected_patches:
        print(f"Warning: Expected {expected_patches} patches but got {len(labels)}")
        # Fallback: try to make it as square as possible
        num_patches = int(np.sqrt(len(labels)))
        if num_patches * num_patches != len(labels):
            # If not perfect square, pad or truncate
            closest_square = num_patches * num_patches
            if len(labels) > closest_square:
                labels = labels[:closest_square]
            else:
                # Pad with the most common label
                most_common_label = np.bincount(labels).argmax()
                padding = np.full(closest_square - len(labels), most_common_label)
                labels = np.concatenate([labels, padding])
        labels_2d = labels.reshape(num_patches, num_patches)
    else:
        # Reshape using actual dimensions
        labels_2d = labels.reshape(num_patches_h, num_patches_w)

    return labels_2d, kmeans


def upscale_segmentation(labels_2d, target_size):
    """Upscale segmentation map to original image size"""
    labels_upscaled = cv2.resize(
        labels_2d.astype(np.float32),
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_NEAREST
    )
    return labels_upscaled.astype(np.int32)


def identify_glacier_cluster(segmentation_map, image_array):
    """
    Identify which cluster represents the glacier
    Glaciers typically have:
    - High brightness (ice/snow)
    - Large connected regions
    """
    unique_labels = np.unique(segmentation_map)
    cluster_stats = []

    for label in unique_labels:
        mask = (segmentation_map == label)
        mean_brightness = image_array[mask].mean()
        area = mask.sum()

        cluster_stats.append({
            'label': label,
            'brightness': mean_brightness,
            'area': area
        })

    # Sort by brightness (glaciers are typically bright)
    cluster_stats.sort(key=lambda x: x['brightness'], reverse=True)

    print("Cluster statistics:")
    for stats in cluster_stats:
        print(f"  Cluster {stats['label']}: brightness={stats['brightness']:.2f}, area={stats['area']}")

    return cluster_stats[0]['label']  # Return brightest cluster


def extract_glacier_borders(glacier_mask):
    """
    Extract and refine glacier borders
    """
    # Clean up the mask
    glacier_mask_clean = morphology.remove_small_objects(glacier_mask.astype(bool), min_size=500)
    glacier_mask_clean = morphology.remove_small_holes(glacier_mask_clean, area_threshold=500)
    glacier_mask_clean = ndimage.binary_fill_holes(glacier_mask_clean)

    # Find contours
    contours = measure.find_contours(glacier_mask_clean.astype(float), 0.5)

    # Detect edges using Canny
    edges = cv2.Canny((glacier_mask_clean * 255).astype(np.uint8), 100, 200)

    return glacier_mask_clean, contours, edges


def visualize_results(original_image, segmentation_map, glacier_mask, contours, edges):
    """
    Visualize the detection results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Satellite Image')
    axes[0, 0].axis('off')

    # Segmentation map
    axes[0, 1].imshow(segmentation_map, cmap='tab10')
    axes[0, 1].set_title('Segmentation Map (All Clusters)')
    axes[0, 1].axis('off')

    # Glacier mask
    axes[0, 2].imshow(glacier_mask, cmap='gray')
    axes[0, 2].set_title('Detected Glacier Mask')
    axes[0, 2].axis('off')

    # Edges
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title('Glacier Edges (Canny)')
    axes[1, 0].axis('off')

    # Overlay contours on original
    axes[1, 1].imshow(original_image)
    for contour in contours:
        axes[1, 1].plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
    axes[1, 1].set_title('Glacier Borders Overlay')
    axes[1, 1].axis('off')

    # Overlay mask on original
    overlay = np.array(original_image).copy()
    mask_overlay = np.zeros_like(overlay)
    mask_overlay[glacier_mask] = [0, 255, 255]  # Cyan for glacier
    blended = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)
    axes[1, 2].imshow(blended)
    axes[1, 2].set_title('Glacier Region Overlay')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('glacier_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_glacier_data(glacier_mask, contours, output_prefix='glacier'):
    """
    Save glacier mask and border coordinates
    """
    # Save mask
    cv2.imwrite(f'{output_prefix}_mask.png', (glacier_mask * 255).astype(np.uint8))

    # Save largest contour coordinates
    if contours:
        largest_contour = max(contours, key=len)
        np.savetxt(f'{output_prefix}_border_coords.txt', largest_contour,
                   fmt='%.2f', header='row col', comments='')
        print(f"Saved glacier border coordinates to {output_prefix}_border_coords.txt")

    # Calculate area (in pixels)
    area_pixels = glacier_mask.sum()
    print(f"Glacier area: {area_pixels} pixels")

    return area_pixels


def detect_glacier_borders(image_path, n_clusters=3, visualize=True):
    """
    Main function to detect glacier borders

    Args:
        image_path: Path to the satellite image
        n_clusters: Number of clusters for segmentation (3-5 typically works well)
        visualize: Whether to display visualization

    Returns:
        glacier_mask, contours, area_pixels
    """
    print("Loading image...")
    image = load_and_preprocess_image(image_path)
    image_array = np.array(image)

    print("Extracting features with DINOv3...")
    patch_features, input_shape = extract_features_with_dinov3(image)

    print("Segmenting image...")
    labels_2d, kmeans = segment_glacier(patch_features, input_shape)

    print("Upscaling segmentation...")
    segmentation_map = upscale_segmentation(labels_2d, image_array.shape[:2])

    print("Identifying glacier cluster...")
    # Convert to grayscale for brightness calculation
    image_gray = np.array(image.convert('L'))
    glacier_label = identify_glacier_cluster(segmentation_map, image_gray)

    glacier_mask = (segmentation_map == glacier_label)

    print("Extracting glacier borders...")
    glacier_mask_clean, contours, edges = extract_glacier_borders(glacier_mask)

    print("Saving results...")
    area_pixels = save_glacier_data(glacier_mask_clean, contours)

    if visualize:
        print("Visualizing results...")
        visualize_results(image, segmentation_map, glacier_mask_clean, contours, edges)

    return glacier_mask_clean, contours, area_pixels


if __name__ == "__main__":
    # Replace with your image path
    IMAGE_PATH = "glacier.png"

    # Detect glacier borders
    glacier_mask, contours, area = detect_glacier_borders(
        IMAGE_PATH,
        n_clusters=4,  # Adjust based on image complexity
        visualize=True
    )

    print(f"\nDetection complete!")
    print(f"Number of border contours: {len(contours)}")
    print(f"Glacier area: {area} pixels")