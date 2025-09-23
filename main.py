import matplotlib
matplotlib.use('Agg')
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel
from skimage.transform import resize
from skimage.color import rgb2hsv

# Optional filters/morphology for better separation/refinement
try:
    from skimage.filters import sobel
    from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk
    HAVE_MORPH = True
except Exception:
    HAVE_MORPH = False

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

# Compute per-patch color/texture/saliency features aligned to (h, w)
img_np = np.array(image).astype(np.float32) / 255.0
img_small = resize(img_np, (h, w, 3), preserve_range=True, anti_aliasing=True)
img_hsv = rgb2hsv(img_small)
R, G, B = img_small[..., 0], img_small[..., 1], img_small[..., 2]
S = img_hsv[..., 1]
V = img_hsv[..., 2]
blue_ratio = B / (R + G + B + 1e-6)

# Texture via Sobel on grayscale
gray_small = 0.299 * R + 0.587 * G + 0.114 * B
if HAVE_MORPH:
    tex = sobel(gray_small)
else:
    # Simple gradient approximation if sobel unavailable
    gy = np.zeros_like(gray_small)
    gx = np.zeros_like(gray_small)
    gy[1:, :] = np.abs(gray_small[1:, :] - gray_small[:-1, :])
    gx[:, 1:] = np.abs(gray_small[:, 1:] - gray_small[:, :-1])
    tex = (gx + gy) * 0.5

# Patch-level saliency (normalize fmap_2d to [0,1])
sal_patch = fmap_2d.detach().cpu().numpy()
sal_patch = (sal_patch - sal_patch.min())
if sal_patch.max() > 1e-6:
    sal_patch = sal_patch / sal_patch.max()

# Normalize auxiliary features to [0,1] for balanced clustering/scoring
feature_list = []
for comp in [R, G, B, S, V, blue_ratio, tex, sal_patch]:
    comp_n = comp - comp.min()
    maxv = comp_n.max()
    if maxv > 1e-6:
        comp_n = comp_n / maxv
    feature_list.append(comp_n)

aux_feats = np.stack(feature_list, axis=-1).reshape(-1, len(feature_list))  # (H*W, F)

# Build clustering features: DINO features (optionally PCA) + auxiliary features
if PCA is not None:
    pca = PCA(n_components=min(50, patch_tokens.shape[1]))
    dino_feats = pca.fit_transform(patch_tokens)
else:
    dino_feats = patch_tokens

feats = np.concatenate([dino_feats, aux_feats], axis=1)

if KMeans is not None:
    K = 5
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
    labels = kmeans.fit_predict(feats)
else:
    # Fallback: simple numpy KMeans-like (K=4)
    K = 4
    rng = np.random.default_rng(42)
    centroids = feats[rng.choice(feats.shape[0], size=K, replace=False)]
    for _ in range(15):
        dists = ((feats[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        for k in range(K):
            if np.any(labels == k):
                centroids[k] = feats[labels == k].mean(axis=0)

# Reshape labels to (h, w)
labels_2d = labels.reshape(h, w)

# --- 7. Score clusters to select glacier ---
# Use normalized V (bright), low S, higher blue_ratio, moderate-to-low texture, and higher saliency
cluster_scores = []
cluster_debug = []
for k in range(labels_2d.max() + 1):
    m = (labels_2d == k)
    if m.sum() == 0:
        cluster_scores.append(-np.inf)
        cluster_debug.append((k, 0, 0, 0, 0, 0))
        continue
    V_k = V[m].mean()
    S_k = S[m].mean()
    Bk = blue_ratio[m].mean()
    T_k = tex[m].mean()
    Sal_k = sal_patch[m].mean()
    # Water penalty: very dark patches are unlikely to be glacier
    water_pen = max(0.0, 0.45 - V_k)
    # Score weights (tunable)
    score = 1.2 * V_k - 0.8 * S_k + 0.5 * Bk + 0.3 * Sal_k - 0.6 * T_k - 0.8 * water_pen
    cluster_scores.append(score)
    cluster_debug.append((k, V_k, S_k, Bk, T_k, Sal_k))

glacier_k = int(np.argmax(cluster_scores))
print("Cluster scoring (k, V, S, blue_ratio, texture, sal):")
for k, V_k, S_k, Bk, T_k, Sal_k in cluster_debug:
    print(f"  {k}: V={V_k:.3f}, S={S_k:.3f}, B={Bk:.3f}, T={T_k:.3f}, Sal={Sal_k:.3f}, score={cluster_scores[k]:.3f}")
print(f"Selected cluster {glacier_k} as glacier-like.")

# Binary mask at patch resolution (and refine)
mask_patch = (labels_2d == glacier_k)

# Simple morphological refinement at patch scale (optional)
if HAVE_MORPH:
    mask_patch_ref = binary_opening(mask_patch, footprint=disk(1))
    mask_patch_ref = binary_closing(mask_patch_ref, footprint=disk(1))
else:
    mask_patch_ref = mask_patch

# Upsample refined mask to image resolution (nearest neighbor)
mask_up = resize(mask_patch_ref.astype(np.float32), (image.height, image.width), order=0, preserve_range=True, anti_aliasing=False)

# Image-level morphological cleanup: remove tiny regions, smooth edges
if HAVE_MORPH:
    # Remove very small objects: <0.2% of image area
    area_thresh = int(0.002 * mask_up.size)
    mask_bool = mask_up > 0.5
    mask_bool = remove_small_objects(mask_bool, min_size=area_thresh)
    mask_bool = binary_opening(mask_bool, footprint=disk(2))
    mask_bool = binary_closing(mask_bool, footprint=disk(2))
    mask_up = mask_bool.astype(np.float32)

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

# Glacier mask overlay (refined)
axes[1, 1].imshow(image)
axes[1, 1].imshow(mask_up, cmap='Blues', alpha=0.45)
axes[1, 1].set_title("Glacier-like Region (refined)")
axes[1, 1].axis('off')

plt.tight_layout()

# Save out figures and masks
plt.savefig('dinov3_glacier_demo.png', dpi=200)

# Save raw mask for downstream use
mask_uint8 = (mask_up * 255).astype(np.uint8)
Image.fromarray(mask_uint8).save('glacier_mask_refined.png')

# Also save saliency (attention/similarity) map for convenience
sal_uint8 = (sal * 255).astype(np.uint8)
Image.fromarray(sal_uint8).save('saliency_map.png')
