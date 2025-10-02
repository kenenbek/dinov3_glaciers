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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
    # Calculate actual patch grid dimensions from input (ViT patch size = 16)
    patch_size = 16
    batch_size, channels, height, width = input_shape
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    expected_patches = num_patches_h * num_patches_w

    # Extract only spatial patch tokens (exclude special tokens like CLS/register)
    tokens = patch_features[0]  # [num_tokens, hidden_dim]
    if tokens.shape[0] >= expected_patches:
        spatial_tokens = tokens[-expected_patches:]
    else:
        # Fallback if model returned fewer tokens than expected
        spatial_tokens = tokens[1:]  # best-effort skip CLS

    # Convert to float32 numpy
    features = spatial_tokens.float().cpu().numpy()

    # Standardize and reduce dimensionality for better clustering stability
    scaler = StandardScaler()
    features_std = scaler.fit_transform(features)

    # Keep up to 64 components or fewer if feature dim smaller
    n_components = min(64, features_std.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    features_pca = pca.fit_transform(features_std)

    # Perform clustering on PCA-reduced features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_pca)

    print(f"Features (tokens): {tokens.shape[0]}, spatial: {features.shape}, PCA: {features_pca.shape}")
    print(f"Patch grid: {num_patches_h} x {num_patches_w} = {expected_patches}, labels: {labels.shape}")

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


def identify_glacier_cluster(segmentation_map, image_rgb):
    """
    Identify which cluster represents the glacier using brightness, saturation, and area.
    Glaciers typically have high brightness and low saturation.
    """
    unique_labels = np.unique(segmentation_map)

    # Prepare color spaces
    image_rgb = np.asarray(image_rgb)
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    cluster_stats = []
    for label in unique_labels:
        mask = (segmentation_map == label)
        area = int(mask.sum())
        if area == 0:
            continue
        mean_brightness = float(image_gray[mask].mean())  # 0..255
        mean_saturation = float(image_hsv[..., 1][mask].mean())  # 0..255

        cluster_stats.append({
            'label': int(label),
            'brightness': mean_brightness,
            'saturation': mean_saturation,
            'area': area
        })

    # Normalize stats to 0..1 for fair scoring
    if not cluster_stats:
        # Fallback to largest label if something went wrong
        return int(unique_labels[np.argmax([(segmentation_map == l).sum() for l in unique_labels])])

    brightness_vals = np.array([c['brightness'] for c in cluster_stats])
    saturation_vals = np.array([c['saturation'] for c in cluster_stats])
    area_vals = np.array([c['area'] for c in cluster_stats], dtype=float)

    # Min-max normalize
    b_norm = (brightness_vals - brightness_vals.min()) / (brightness_vals.ptp() + 1e-6)
    s_norm = (saturation_vals - saturation_vals.min()) / (saturation_vals.ptp() + 1e-6)
    a_norm = (np.log1p(area_vals) - np.log1p(area_vals).min()) / (np.log1p(area_vals).ptp() + 1e-6)

    # Higher brightness, lower saturation, larger area
    score = 0.6 * b_norm + 0.3 * (1 - s_norm) + 0.1 * a_norm
    best_idx = int(np.argmax(score))
    best_label = int(cluster_stats[best_idx]['label'])

    print("Cluster statistics (brightness, saturation, area, score):")
    for i, c in enumerate(cluster_stats):
        print(f"  Cluster {c['label']}: {c['brightness']:.1f}, {c['saturation']:.1f}, {c['area']}, score={score[i]:.3f}")

    return best_label


def refine_glacier_mask(image_rgb, initial_mask, method='morph'):
    """
    Refine the glacier mask with morphology or GrabCut for sharper borders.
    method: 'morph' (default) or 'grabcut'
    """
    mask = initial_mask.astype(np.uint8)

    # Basic morphological smoothing (opening then closing)
    h, w = mask.shape
    # Kernel size scales with image size
    k = max(3, int(round(min(h, w) * 0.003)) | 1)  # ensure odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if method == 'grabcut':
        # Prepare GrabCut mask: 0=BG, 1=FG, 2=prob BG, 3=prob FG
        gc_mask = np.where(mask > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)

        # Use a tight rectangle around the mask to speed up
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            rect = (max(0, x0 - 5), max(0, y0 - 5), min(w - 1, x1 + 5) - max(0, x0 - 5), min(h - 1, y1 + 5) - max(0, y0 - 5))
        else:
            rect = (1, 1, w - 2, h - 2)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(image_bgr, gc_mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
            refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        except Exception as e:
            print(f"GrabCut failed: {e}. Falling back to morphology.")
            refined = mask
        return refined.astype(bool)

    return mask.astype(bool)


def extract_glacier_borders(glacier_mask):
    """
    Extract and refine glacier borders
    """
    # Clean up the mask
    glacier_mask_clean = morphology.remove_small_objects(glacier_mask.astype(bool), min_size=1000)
    glacier_mask_clean = morphology.remove_small_holes(glacier_mask_clean, area_threshold=1000)
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


def detect_glacier_borders(image_path, n_clusters=3, visualize=True, refine_method='morph'):
    """
    Main function to detect glacier borders

    Args:
        image_path: Path to the satellite image
        n_clusters: Number of clusters for segmentation (3-5 typically works well)
        visualize: Whether to display visualization
        refine_method: 'morph' or 'grabcut' for border refinement

    Returns:
        glacier_mask, contours, area_pixels
    """
    print("Loading image...")
    image = load_and_preprocess_image(image_path)
    image_array = np.array(image)

    print("Extracting features with DINOv3...")
    patch_features, input_shape = extract_features_with_dinov3(image)

    print("Segmenting image...")
    labels_2d, kmeans = segment_glacier(patch_features, input_shape, n_clusters=n_clusters)

    print("Upscaling segmentation...")
    segmentation_map = upscale_segmentation(labels_2d, image_array.shape[:2])

    print("Identifying glacier cluster...")
    glacier_label = identify_glacier_cluster(segmentation_map, image_array)

    initial_mask = (segmentation_map == glacier_label)

    print("Refining glacier mask...")
    refined_mask = refine_glacier_mask(image_array, initial_mask, method=refine_method)

    print("Extracting glacier borders...")
    glacier_mask_clean, contours, edges = extract_glacier_borders(refined_mask)

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
        n_clusters=5,  # Slightly more clusters helps separate ice/snow/rock/shadow
        visualize=True,
        refine_method='morph'  # try 'grabcut' for sharper edges if OpenCV is available
    )

    print(f"\nDetection complete!")
    print(f"Number of border contours: {len(contours)}")
    print(f"Glacier area: {area} pixels")