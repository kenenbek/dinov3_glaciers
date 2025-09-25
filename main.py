import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, AutoModel

from skimage.transform import resize
from skimage.color import rgb2hsv
from skimage.filters import sobel, threshold_otsu
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk
from skimage.segmentation import slic, find_boundaries, random_walker

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# --- 1. Load Model and Processor ---
processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-sat493m")
model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-sat493m")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- 2. Load Image ---
image_path = 'img.png'
image = Image.open(image_path).convert("RGB")
img_np = np.array(image).astype(np.float32) / 255.0
H_img, W_img = image.height, image.width

# --- 3. Preprocess and Forward ---
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

# --- 4. Derive patch grid and CLS–Patch similarity map ---
patch_size = getattr(getattr(model.config, 'vision_config', model.config), 'patch_size', 16)
H_proc = int(inputs['pixel_values'].shape[2])
W_proc = int(inputs['pixel_values'].shape[3])
h = H_proc // patch_size
w = W_proc // patch_size
num_patch_tokens = h * w

last_hidden = outputs.last_hidden_state  # (1, N+1(+reg), D)
cls = last_hidden[0, 0, :]
patches = last_hidden[0, 1:1+num_patch_tokens, :]
cls_n = cls / (cls.norm(p=2) + 1e-6)
patches_n = patches / (patches.norm(p=2, dim=1, keepdim=True) + 1e-6)
fmap_1d = (patches_n @ cls_n)  # (H*W,)
fmap_2d = fmap_1d.reshape(h, w)

# Resize saliency to image size and normalize
saliency_map_resized = resize(fmap_2d.detach().cpu().numpy(), (H_img, W_img), preserve_range=True)
sal = saliency_map_resized
sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)

# --- 5. Patch features (DINO + color/texture/saliency) ---
patch_tokens_last = last_hidden[0, 1:1+num_patch_tokens, :].detach().cpu().numpy()  # (H*W, D)

# Compact DINO features for clustering/classification
pca = PCA(n_components=min(64, patch_tokens_last.shape[1]), random_state=42)
dino_feats = pca.fit_transform(patch_tokens_last)

# Color/texture features aligned to (h, w)
img_small = resize(img_np, (h, w, 3), preserve_range=True, anti_aliasing=True)
img_hsv = rgb2hsv(img_small)
R, G, B = img_small[..., 0], img_small[..., 1], img_small[..., 2]
S = img_hsv[..., 1]
V = img_hsv[..., 2]
blue_ratio = B / (R + G + B + 1e-6)

# Texture via Sobel on grayscale
gray_small = 0.299 * R + 0.587 * G + 0.114 * B
tex = sobel(gray_small)

# Patch-level saliency normalized to [0,1]
sal_patch = fmap_2d.detach().cpu().numpy()
sal_patch = (sal_patch - sal_patch.min()) / (sal_patch.max() - sal_patch.min() + 1e-6)

# Normalize auxiliary features to [0,1]
aux_features = []
for comp in [R, G, B, S, V, blue_ratio, tex, sal_patch]:
    comp_n = (comp - comp.min()) / (comp.max() - comp.min() + 1e-6)
    aux_features.append(comp_n)
aux_feats = np.stack(aux_features, axis=-1).reshape(-1, len(aux_features))  # (H*W, F)

# Combine features: DINO + auxiliary
feats = np.concatenate([dino_feats, aux_feats], axis=1)  # (H*W, D')

# --- 6. Pseudo-label seeds (snow vs land) ---
# Snow/glacier tends to be bright (V), low saturation (S), bluish (blue_ratio), and salient (sal_patch)
V_q = np.quantile(V, 0.75)
S_q = np.quantile(S, 0.25)
B_q = np.quantile(blue_ratio, 0.75)
Sal_q = np.quantile(sal_patch, 0.75)
T_q = np.quantile(tex, 0.50)

pos_seed = (V >= max(0.6, V_q)) & (S <= min(0.45, S_q)) & (blue_ratio >= max(0.3, B_q*0.9)) & (sal_patch >= max(0.5, Sal_q))
neg_seed_water = (V <= min(0.45, np.quantile(V, 0.35))) & (blue_ratio >= max(0.35, B_q))
neg_seed_veg = (S >= max(0.55, np.quantile(S, 0.65))) & (G >= R) & (G >= B)
neg_seed_rock = (tex >= max(0.5, T_q)) & (blue_ratio <= min(0.3, np.quantile(blue_ratio, 0.35)))
neg_seed = neg_seed_water | neg_seed_veg | neg_seed_rock

seed_y = np.full((h, w), -1, dtype=np.int32)
seed_y[pos_seed] = 1
seed_y[neg_seed] = 0

# --- 7. Train patch-level classifier (LogReg) and predict ---
X_train = feats[seed_y.reshape(-1) >= 0]
y_train = seed_y.reshape(-1)[seed_y.reshape(-1) >= 0]
clf = LogisticRegression(max_iter=500, class_weight='balanced', solver='lbfgs', random_state=42)
clf.fit(X_train, y_train)
probs_patch = clf.predict_proba(feats)[:, 1]  # glacier probability

probs_2d = probs_patch.reshape(h, w)

# --- 8. Upscale and refine with superpixels + Random Walker ---
prob_img = resize(probs_2d, (H_img, W_img), order=1, preserve_range=True, anti_aliasing=True)
prob_img = np.clip(prob_img, 0.0, 1.0).astype(np.float32)

# SLIC superpixel smoothing
n_segments = max(200, (H_img * W_img) // 1500)
segments = slic(img_np, n_segments=n_segments, compactness=10, sigma=1, start_label=0)
prob_sp = prob_img.copy()
for seg_id in np.unique(segments):
    mask_sp = (segments == seg_id)
    mean_p = prob_img[mask_sp].mean()
    prob_sp[mask_sp] = mean_p
prob_img = prob_sp

# Random Walker refinement guided by the RGB image and seeds from prob_img
labels_seed = np.full((H_img, W_img), -1, dtype=np.int32)
labels_seed[prob_img > 0.8] = 1
labels_seed[prob_img < 0.2] = 0
probs_rw = random_walker(img_np, labels_seed, beta=120, mode='cg_mg', tol=1e-3, return_full_prob=True)
prob_img = probs_rw[1].astype(np.float32)

# --- 9. Morphological cleanup and border extraction ---
th = threshold_otsu(prob_img)
mask_bin = prob_img > th

area_thresh = int(0.002 * mask_bin.size)
mask_bin = remove_small_objects(mask_bin, min_size=area_thresh)
mask_bin = binary_opening(mask_bin, footprint=disk(2))
mask_bin = binary_closing(mask_bin, footprint=disk(2))

# Glacier borders
edges = find_boundaries(mask_bin, mode='thick')

# --- 10. Visualize and Save Results ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Original
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

# Saliency overlay
axes[0, 1].imshow(image)
axes[0, 1].imshow(sal, cmap='jet', alpha=0.45)
axes[0, 1].set_title("DINOv3 CLS–Patch Similarity")
axes[0, 1].axis('off')

# Seeds
seed_vis = np.zeros((h, w, 3), dtype=np.float32)
seed_vis[pos_seed] = [0.1, 0.8, 0.1]  # green
seed_vis[neg_seed] = [0.8, 0.1, 0.1]  # red
axes[0, 2].imshow(resize(seed_vis, (H_img, W_img), order=0, preserve_range=True))
axes[0, 2].set_title("Seeds (green=snow, red=non-snow)")
axes[0, 2].axis('off')

# Patch-level probability map (resized)
axes[1, 0].imshow(resize(probs_2d, (H_img, W_img), preserve_range=True), cmap='viridis', vmin=0, vmax=1)
axes[1, 0].set_title("Glacier Probability (patch->pixel)")
axes[1, 0].axis('off')

# Refined probability
axes[1, 1].imshow(prob_img, cmap='viridis', vmin=0, vmax=1)
axes[1, 1].set_title("Refined Probability (SLIC + RandomWalker)")
axes[1, 1].axis('off')

# Final mask + borders
axes[1, 2].imshow(image)
border_overlay = np.zeros((H_img, W_img, 4), dtype=np.float32)
border_overlay[..., 0] = edges.astype(np.float32)  # red borders
border_overlay[..., 3] = edges.astype(np.float32) * 0.9
axes[1, 2].imshow(mask_bin, cmap='Blues', alpha=0.45)
axes[1, 2].imshow(border_overlay)
axes[1, 2].set_title("Glacier Mask + Borders")
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('dinov3_glacier_segment.png', dpi=200)

# Save artifacts
Image.fromarray((mask_bin.astype(np.uint8) * 255)).save('glacier_mask.png')
Image.fromarray((np.clip(prob_img, 0, 1) * 255).astype(np.uint8)).save('glacier_probability.png')
Image.fromarray((sal * 255).astype(np.uint8)).save('saliency_map.png')

# Save borders as a colored PNG overlay with transparency
edges_rgba = np.zeros((H_img, W_img, 4), dtype=np.uint8)
edges_rgba[..., 0] = (edges.astype(np.uint8) * 255)  # red
edges_rgba[..., 3] = (edges.astype(np.uint8) * 255)  # alpha
Image.fromarray(edges_rgba, mode='RGBA').save('glacier_edges.png')
