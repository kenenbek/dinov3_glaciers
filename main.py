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
    from skimage.filters import sobel, threshold_otsu
    from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk
    from skimage.segmentation import slic
    HAVE_MORPH = True
except Exception:
    HAVE_MORPH = False

# Optional CRF
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    HAVE_CRF = True
except Exception:
    HAVE_CRF = False

# Optional ML
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    SKLEARN_OK = True
except Exception:
    PCA = None
    KMeans = None
    LogisticRegression = None
    SKLEARN_OK = False

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

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

# --- 2. Request attentions and hidden states if supported ---
try:
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    want_attentions = True
    want_hidden = True
except Exception as _e:
    print(f"Warning: Could not enable some outputs on config: {_e}")
    want_attentions = False
    want_hidden = False

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
with torch.no_grad():
    try:
        outputs = model(**inputs, output_attentions=want_attentions, output_hidden_states=want_hidden)
        attentions_available = want_attentions and getattr(outputs, 'attentions', None) is not None
        hidden_available = want_hidden and getattr(outputs, 'hidden_states', None) is not None
    except ValueError as e:
        print(f"Model forward with extras failed, retrying without them: {e}")
        outputs = model(**inputs, output_attentions=False, output_hidden_states=False)
        attentions_available = False
        hidden_available = False

# --- 5. Attention or CLS-sim Heatmap ---
if attentions_available:
    # outputs.attentions: tuple of tensors; use last layer
    attentions = outputs.attentions[-1]  # (B, heads, seq, seq)
    cls_attention = attentions[0, :, 0, 1:1+num_patch_tokens].mean(dim=0)  # (H*W,)
    fmap_1d = cls_attention
    map_name = "CLS Attention"
else:
    last_hidden = outputs.last_hidden_state  # (1, N+1+reg, D)
    cls = last_hidden[0, 0, :]
    patches = last_hidden[0, 1:1+num_patch_tokens, :]
    cls_n = cls / (cls.norm(p=2) + 1e-6)
    patches_n = patches / (patches.norm(p=2, dim=1, keepdim=True) + 1e-6)
    fmap_1d = (patches_n @ cls_n)  # (H*W,)
    map_name = "CLS–Patch Cosine Similarity"

fmap_2d = fmap_1d.reshape(h, w)

# Resize the map to the original image size for overlay
saliency_map_resized = resize(fmap_2d.detach().cpu().numpy(), (image.height, image.width), preserve_range=True)

# --- 6. Extract Patch Features ---
# Base tokens from last hidden state
last_hidden = outputs.last_hidden_state  # (B, N+1+reg, D)
patch_tokens_last = last_hidden[0, 1:1+num_patch_tokens, :].detach().cpu().numpy()  # (H*W, D)

# Multi-layer features: average of last 4 hidden layers' patch tokens
if hidden_available and outputs.hidden_states is not None and len(outputs.hidden_states) >= 4:
    patch_layers = []
    for layer_out in outputs.hidden_states[-4:]:
        patch_layers.append(layer_out[0, 1:1+num_patch_tokens, :].detach().cpu().numpy())
    patch_tokens_multi = np.mean(np.stack(patch_layers, axis=0), axis=0)  # (H*W, D)
else:
    patch_tokens_multi = patch_tokens_last

# Optional: PCA for compactness
if PCA is not None:
    pca = PCA(n_components=min(64, patch_tokens_multi.shape[1]))
    dino_feats = pca.fit_transform(patch_tokens_multi)
else:
    dino_feats = patch_tokens_multi

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
    gy = np.zeros_like(gray_small)
    gx = np.zeros_like(gray_small)
    gy[1:, :] = np.abs(gray_small[1:, :] - gray_small[:-1, :])
    gx[:, 1:] = np.abs(gray_small[:, 1:] - gray_small[:, :-1])
    tex = (gx + gy) * 0.5

# Patch-level saliency normalized to [0,1]
sal_patch = fmap_2d.detach().cpu().numpy()
sal_patch = (sal_patch - sal_patch.min())
if sal_patch.max() > 1e-6:
    sal_patch = sal_patch / sal_patch.max()

# Normalize auxiliary features to [0,1]
aux_features = []
for comp in [R, G, B, S, V, blue_ratio, tex, sal_patch]:
    comp_n = comp - comp.min()
    maxv = comp_n.max()
    if maxv > 1e-6:
        comp_n = comp_n / maxv
    aux_features.append(comp_n)
aux_feats = np.stack(aux_features, axis=-1).reshape(-1, len(aux_features))  # (H*W, F)

# Combine features: DINO + auxiliary
feats = np.concatenate([dino_feats, aux_feats], axis=1)  # (H*W, D')

# --- 7. Seed generation (adaptive thresholds) ---
# Heuristics for glacier (bright, low saturation, bluish, salient)
V_q = np.quantile(V, 0.75)
S_q = np.quantile(S, 0.25)
B_q = np.quantile(blue_ratio, 0.75)
Sal_q = np.quantile(sal_patch, 0.75)
T_q = np.quantile(tex, 0.50)

pos_seed = (V >= max(0.6, V_q)) & (S <= min(0.45, S_q)) & (blue_ratio >= max(0.3, B_q*0.9)) & (sal_patch >= max(0.5, Sal_q))
# Negatives: water (dark, blue), vegetation (green/high S), rocks (high texture, low blue)
neg_seed_water = (V <= min(0.45, np.quantile(V, 0.35))) & (blue_ratio >= max(0.35, B_q))
neg_seed_veg = (S >= max(0.55, np.quantile(S, 0.65))) & (G >= R) & (G >= B)
neg_seed_rock = (tex >= max(0.5, T_q)) & (blue_ratio <= min(0.3, np.quantile(blue_ratio, 0.35)))
neg_seed = neg_seed_water | neg_seed_veg | neg_seed_rock

# Guarantee some seeds by taking top-k by saliency and brightness if needed
if pos_seed.sum() < max(10, 0.01 * num_patch_tokens):
    k = int(max(10, 0.01 * num_patch_tokens))
    flat_scores = (0.6*V + 0.2*(1-S) + 0.2*blue_ratio + 0.3*sal_patch).reshape(-1)
    top_idx = np.argsort(flat_scores)[-k:]
    pos_seed = pos_seed.reshape(-1)
    pos_seed[top_idx] = True
    pos_seed = pos_seed.reshape(h, w)

if neg_seed.sum() < max(10, 0.01 * num_patch_tokens):
    k = int(max(10, 0.01 * num_patch_tokens))
    flat_scores = (0.5*(1-V) + 0.2*S + 0.2*(1-blue_ratio) + 0.1*(tex)).reshape(-1)
    top_idx = np.argsort(flat_scores)[-k:]
    neg_seed = neg_seed.reshape(-1)
    neg_seed[top_idx] = True
    neg_seed = neg_seed.reshape(h, w)

seed_y = np.full((h, w), -1, dtype=np.int32)
seed_y[pos_seed] = 1
seed_y[neg_seed] = 0

# --- 8. Train patch-level classifier on seeds ---
probs_patch = None
if SKLEARN_OK and LogisticRegression is not None and (seed_y >= 0).sum() >= 20 and len(np.unique(seed_y[seed_y>=0])) == 2:
    X_train = feats[seed_y.reshape(-1) >= 0]
    y_train = seed_y.reshape(-1)[seed_y.reshape(-1) >= 0]
    clf = LogisticRegression(max_iter=500, class_weight='balanced', solver='lbfgs')
    clf.fit(X_train, y_train)
    probs_patch = clf.predict_proba(feats)[:, 1]  # glacier probability
else:
    # Fallback: KMeans with 2 clusters, choose cluster more glacier-like by score
    K = 2
    if SKLEARN_OK and KMeans is not None:
        kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
        labels = kmeans.fit_predict(feats)
    else:
        # Simple numpy KMeans
        rng = np.random.default_rng(42)
        centroids = feats[rng.choice(feats.shape[0], size=K, replace=False)]
        for _ in range(20):
            dists = ((feats[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = dists.argmin(axis=1)
            for k in range(K):
                if np.any(labels == k):
                    centroids[k] = feats[labels == k].mean(axis=0)
    labels_2 = labels.reshape(h, w)
    # Score clusters using previous scoring function
    cluster_scores = []
    for k in range(K):
        m = (labels_2 == k)
        if m.sum() == 0:
            cluster_scores.append(-np.inf)
            continue
        V_k = V[m].mean(); S_k = S[m].mean(); Bk = blue_ratio[m].mean(); T_k = tex[m].mean(); Sal_k = sal_patch[m].mean()
        water_pen = max(0.0, 0.45 - V_k)
        score = 1.2 * V_k - 0.8 * S_k + 0.5 * Bk + 0.3 * Sal_k - 0.6 * T_k - 0.8 * water_pen
        cluster_scores.append(score)
    glacier_k = int(np.argmax(cluster_scores))
    probs_patch = (labels_2 == glacier_k).astype(np.float32).reshape(-1)

probs_2d = probs_patch.reshape(h, w)

# --- 9. Superpixel-guided smoothing at image resolution ---
prob_img = resize(probs_2d, (image.height, image.width), order=1, preserve_range=True, anti_aliasing=True)
prob_img = np.clip(prob_img, 0.0, 1.0).astype(np.float32)

if HAVE_MORPH:
    # SLIC superpixels on original image
    n_segments = max(200, (image.height * image.width) // 1500)
    segments = slic(img_np, n_segments=n_segments, compactness=10, sigma=1, start_label=0)
    # Average probability within each superpixel
    prob_sp = prob_img.copy()
    for seg_id in np.unique(segments):
        mask_sp = (segments == seg_id)
        mean_p = prob_img[mask_sp].mean()
        prob_sp[mask_sp] = mean_p
    prob_img = prob_sp

# Optional CRF refinement
if HAVE_CRF:
    H_img, W_img = image.height, image.width
    d = dcrf.DenseCRF2D(W_img, H_img, 2)
    # Unary from softmax expects shape (num_classes, H*W)
    unary = unary_from_softmax(np.vstack([1 - prob_img.reshape(-1), prob_img.reshape(-1)])).astype(np.float32)
    d.setUnaryEnergy(unary)
    # Pairwise Gaussian
    d.addPairwiseGaussian(sxy=3, compat=3)
    # Pairwise bilateral using RGB image
    img_uint8 = (img_np * 255).astype(np.uint8)
    d.addPairwiseBilateral(sxy=50, srgb=13, rgbim=img_uint8, compat=5)
    Q = d.inference(5)
    prob_img = np.array(Q[1]).reshape(H_img, W_img).astype(np.float32)

# Morphological cleanup
if HAVE_MORPH:
    mask_bin = prob_img > (threshold_otsu(prob_img) if prob_img.std() > 1e-3 else 0.5)
    area_thresh = int(0.002 * mask_bin.size)
    mask_bin = remove_small_objects(mask_bin, min_size=area_thresh)
    mask_bin = binary_opening(mask_bin, footprint=disk(2))
    mask_bin = binary_closing(mask_bin, footprint=disk(2))
else:
    mask_bin = prob_img > 0.5

# Normalize saliency for visualization
sal = saliency_map_resized
sal = (sal - sal.min())
if sal.max() > 0:
    sal = sal / sal.max()

# --- 10. Visualize and Save Results ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Original
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

# Saliency overlay
axes[0, 1].imshow(image)
axes[0, 1].imshow(sal, cmap='jet', alpha=0.45)
axes[0, 1].set_title(f"DINOv3 {map_name} Overlay")
axes[0, 1].axis('off')

# Seeds visualization
seed_vis = np.zeros((h, w, 3), dtype=np.float32)
seed_vis[pos_seed] = [0.1, 0.8, 0.1]  # green
seed_vis[neg_seed] = [0.8, 0.1, 0.1]  # red
axes[0, 2].imshow(resize(seed_vis, (image.height, image.width), order=0, preserve_range=True))
axes[0, 2].set_title("Seeds (green=glacier, red=non-glacier)")
axes[0, 2].axis('off')

# Patch probability map
axes[1, 0].imshow(resize(probs_2d, (image.height, image.width), preserve_range=True), cmap='viridis', vmin=0, vmax=1)
axes[1, 0].set_title("Glacier Probability (patch->pixel)")
axes[1, 0].axis('off')

# Superpixel/CRF refined probability
axes[1, 1].imshow(prob_img, cmap='viridis', vmin=0, vmax=1)
axes[1, 1].set_title("Refined Probability (SLIC/CRF)")
axes[1, 1].axis('off')

# Final mask overlay
axes[1, 2].imshow(image)
axes[1, 2].imshow(mask_bin, cmap='Blues', alpha=0.45)
axes[1, 2].set_title("Glacier Mask (refined)")
axes[1, 2].axis('off')

plt.tight_layout()

# Save out figures and masks
plt.savefig('dinov3_glacier_better.png', dpi=200)

# Save raw mask and probability
mask_uint8 = (mask_bin.astype(np.uint8) * 255)
Image.fromarray(mask_uint8).save('glacier_mask_refined.png')
prob_uint8 = (np.clip(prob_img, 0, 1) * 255).astype(np.uint8)
Image.fromarray(prob_uint8).save('glacier_probability.png')

# Also save saliency map for convenience
sal_uint8 = (sal * 255).astype(np.uint8)
Image.fromarray(sal_uint8).save('saliency_map.png')
