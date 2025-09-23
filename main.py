import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel
from skimage.transform import resize

# --- 1. Load Model and Processor ---
processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-sat493m")
model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-sat493m")

# --- 2. Force the Model Configuration to Output Attentions ---
#
# THIS IS THE NEW, CRITICAL FIX.
# We manually set the configuration to ensure attention weights are returned.
#
model.config.output_attentions = True


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

# Reshape the attention map to a 2D grid
patch_size = model.config.patch_size
img_height_processed = inputs['pixel_values'].shape[2]
img_width_processed = inputs['pixel_values'].shape[3]

w = img_width_processed // patch_size
h = img_height_processed // patch_size
attention_map = cls_attention.reshape(h, w)

# Resize the attention map to the original image size for overlay
attention_map_resized = resize(attention_map.numpy(), (image.height, image.width))

# --- 6. Visualize the Attention Map ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Original Image
ax1.imshow(image)
ax1.set_title("Original Image")
ax1.axis('off')

# Image with Attention Overlay
ax2.imshow(image)
ax2.imshow(attention_map_resized, cmap='jet', alpha=0.5) # Overlay heatmap
ax2.set_title("Attention Map Overlay")
ax2.axis('off')

plt.tight_layout()
plt.show()