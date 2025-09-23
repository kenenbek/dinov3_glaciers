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

# Try to switch attention implementation to 'eager' so we can request attentions
try:
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    elif hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"
    elif hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
except Exception as _e:
    print(f"Warning: Could not set attn_implementation to 'eager': {_e}")

# --- 2. Request attentions if supported ---
# We manually set the configuration to ensure attention weights are returned.
try:
    model.config.output_attentions = True
    want_attentions = True
except Exception as _e:
    print(f"Warning: Could not enable output_attentions on config: {_e}")
    want_attentions = False

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

# Derive patch grid from preprocessed tensor and model patch size
patch_size = getattr(getattr(model.config, 'vision_config', model.config), 'patch_size', 16)
H_proc = int(inputs['pixel_values'].shape[2])
W_proc = int(inputs['pixel_values'].shape[3])
h = H_proc // patch_size
w = W_proc // patch_size
num_patch_tokens = h * w

# --- 4. Model Inference ---
# Prefer attentions if possible; if it errors, fall back gracefully
with torch.no_grad():
    try:
        outputs = model(**inputs, output_attentions=want_attentions)
        attentions_available = want_attentions and getattr(outputs, 'attentions', None) is not None
    except ValueError as e:
        print(f"Model forward with attentions failed, retrying without attentions: {e}")
        outputs = model(**inputs, output_attentions=False)
        attentions_available = False


# --- 5. Attention or CLS-sim Heatmap ---
if attentions_available:
    # outputs.attentions: tuple of tensors; use last layer
    attentions = outputs.attentions[-1]  # (batch, heads, seq, seq)
    # Average across heads for [CLS] -> patch attention, take only H*W patch tokens
    cls_attention = attentions[0, :, 0, 1:1+num_patch_tokens].mean(dim=0)  # (H*W,)
    fmap_1d = cls_attention
    map_name = "CLS Attention"
else:
    # Fallback: use cosine similarity between CLS token and patch tokens as a saliency map
    last_hidden = outputs.last_hidden_state  # (1, N+1+reg, D)
    cls = last_hidden[0, 0, :]
    patches = last_hidden[0, 1:1+num_patch_tokens, :]  # take first H*W as spatial tokens
    cls_n = cls / (cls.norm(p=2) + 1e-6)
    patches_n = patches / (patches.norm(p=2, dim=1, keepdim=True) + 1e-6)
    fmap_1d = (patches_n @ cls_n)  # (H*W,)
    map_name = "CLS–Patch Cosine Similarity"

fmap_2d = fmap_1d.reshape(h, w)

# Resize the map to the original image size for overlay
saliency_map_resized = resize(fmap_2d.detach().cpu().numpy(), (image.height, image.width), preserve_range=True)

# --- 6. Extract Patch Features and Cluster (Self-supervised segmentation) ---
# tokens: (1, 1 + H*W, D) for ViT-like models. Exclude CLS (index 0) and any register tokens after H*W
last_hidden = outputs.last_hidden_state  # (B, N+1+reg, D)
patch_tokens = last_hidden[0, 1:1+num_patch_tokens, :].detach().cpu().numpy()  # (H*W, D)

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

# Normalize saliency for visualization
sal = saliency_map_resized
sal = (sal - sal.min())
if sal.max() > 0:
    sal = sal / sal.max()

# --- 8. Visualize and Save Results ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Original
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

# Saliency overlay (attention or similarity)
axes[0, 1].imshow(image)
axes[0, 1].imshow(sal, cmap='jet', alpha=0.45)
axes[0, 1].set_title(f"DINOv3 {map_name} Overlay")
axes[0, 1].axis('off')

# Patch clustering (all clusters)
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

# Also save saliency (attention/similarity) map for convenience
sal_uint8 = (sal * 255).astype(np.uint8)
Image.fromarray(sal_uint8).save('saliency_map.png')
