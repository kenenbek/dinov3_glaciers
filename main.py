import matplotlib
matplotlib.use('Agg')
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel
from skimage.transform import resize
from skimage.color import rgb2hsv

# --- 1. Load Model and Processor ---
processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-sat493m")
model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-sat493m")

# --- 2. Force the Model Configuration to Output Attentions ---
#
# THIS IS THE NEW, CRITICAL FIX.
# We manually set the configuration to ensure attention weights are returned.
#
model.config.output_attentions = True

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

try:
    # Make sure your image is named 'img.png' or change the path here
    image_path = 'img.png'
    image = Image.open(image_path).convert("RGB")
    print(f"Local image '{image_path}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{image_path}' was not found.")
    exit()

# --- 3. Preprocess the Image ---
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}


# --- 4. Model Inference ---
# We will still pass the flag here for good measure.
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)


# --- 5. Process Attention Maps ---
# Now, outputs.attentions should contain the attention data.
attentions = outputs.attentions[-1]  # Shape: (batch_size, num_heads, seq_length, seq_length)

# Average the attention weights across all heads for the [CLS] token.
num_heads = attentions.shape[1]
cls_attention = attentions[0, :, 0, 1:].mean(dim=0)

# Determine patch grid size (h, w)
num_patches = cls_attention.numel()
side = int(np.sqrt(num_patches))
if side * side == num_patches:
    h = w = side
else:
    # Fallback to using patch size if available
    patch_size = getattr(getattr(model.config, 'vision_config', model.config), 'patch_size', 16)
    img_height_processed = inputs['pixel_values'].shape[2]
    img_width_processed = inputs['pixel_values'].shape[3]
    w = img_width_processed // patch_size
    h = img_height_processed // patch_size

attention_map = cls_attention.reshape(h, w)

# Resize the attention map to the original image size for overlay
attention_map_resized = resize(attention_map.detach().cpu().numpy(), (image.height, image.width), preserve_range=True)

# --- 6. Extract Patch Features and Cluster (Self-supervised segmentation) ---
# tokens: (1, 1 + H*W, D) for ViT-like models. Exclude CLS (index 0)
last_hidden = outputs.last_hidden_state  # (B, N+1, D)
patch_tokens = last_hidden[0, 1:, :].detach().cpu().numpy()  # (H*W, D)

labels = None
# Optional: PCA to compact features for faster clustering
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
except Exception:
    PCA = None
    KMeans = None

if PCA is not None and KMeans is not None:
    # Reduce to 50D for speed without losing too much structure
    pca = PCA(n_components=min(50, patch_tokens.shape[1]))
    feats = pca.fit_transform(patch_tokens)
    # KMeans clustering into K segments
    K = 3  # you can tweak K
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
    labels = kmeans.fit_predict(feats)  # (H*W,)

if labels is None:
    # Fallback: simple 2-means via numpy (very basic). Not as good as sklearn.
    K = 2
    # init centroids randomly
    rng = np.random.default_rng(42)
    centroids = patch_tokens[rng.choice(patch_tokens.shape[0], size=K, replace=False)]
    for _ in range(10):
        dists = ((patch_tokens[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        for k in range(K):
            if np.any(labels == k):
                centroids[k] = patch_tokens[labels == k].mean(axis=0)

# Reshape labels to (h, w)
labels_2d = labels.reshape(h, w)

# --- 7. Heuristic to select "glacier-like" cluster ---
# Glacier/ice tends to be bright (high V) and low saturation (low S) in RGB.

# Downsample original image to patch grid to align per-patch stats
img_np = np.array(image).astype(np.float32) / 255.0
# skimage.resize expects (rows, cols, channels) -> (H, W, C)
img_small = resize(img_np, (h, w, 3), preserve_range=True, anti_aliasing=True)
img_hsv = rgb2hsv(img_small)
S = img_hsv[..., 1]
V = img_hsv[..., 2]

cluster_scores = []
for k in range(labels_2d.max() + 1):
    mask_k = (labels_2d == k)
    if mask_k.sum() == 0:
        cluster_scores.append(-np.inf)
        continue
    # score: bright but low saturation
    score = (V[mask_k].mean() - 0.6 * S[mask_k].mean())
    cluster_scores.append(score)

glacier_k = int(np.argmax(cluster_scores))
print(f"Selected cluster {glacier_k} as glacier-like (scores={cluster_scores}).")

# Binary mask at patch resolution
glacier_mask_patch = (labels_2d == glacier_k).astype(np.float32)

# Upsample glacier mask to image resolution (nearest neighbor)
mask_up = resize(glacier_mask_patch, (image.height, image.width), order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)

# --- 8. Visualize and Save Results ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Original
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

# Attention overlay
axes[0, 1].imshow(image)
axes[0, 1].imshow(attention_map_resized, cmap='jet', alpha=0.45)
axes[0, 1].set_title("DINOv3 CLS-attention Overlay")
axes[0, 1].axis('off')

# KMeans segmentation (all clusters)
axes[1, 0].imshow(labels_2d, cmap='tab20')
axes[1, 0].set_title(f"Patch Clusters (K={labels_2d.max()+1})")
axes[1, 0].axis('off')

# Glacier mask overlay
axes[1, 1].imshow(image)
axes[1, 1].imshow(mask_up, cmap='Blues', alpha=0.45)
axes[1, 1].set_title("Glacier-like Region (heuristic)")
axes[1, 1].axis('off')

plt.tight_layout()

# Save out figures and masks
plt.savefig('dinov3_glacier_demo.png', dpi=200)

# Save raw mask for downstream use
mask_uint8 = (mask_up * 255).astype(np.uint8)
Image.fromarray(mask_uint8).save('glacier_mask.png')

# Also save attention overlay for convenience
attn_vis = (attention_map_resized - np.min(attention_map_resized))
if attn_vis.max() > 0:
    attn_vis = attn_vis / attn_vis.max()
attn_vis_uint8 = (attn_vis * 255).astype(np.uint8)
Image.fromarray(attn_vis_uint8).save('attention_map.png')
