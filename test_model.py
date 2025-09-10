# test_model.py (Corrected and Optimized Version)

import torch
import torch.nn as nn
import numpy as np
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import math

# --- Add Cloned Repos to Python Path (Same as in train.py) ---
SWIN_TUNA_PATH = os.path.abspath('./swin_tuna')
if SWIN_TUNA_PATH not in sys.path:
    sys.path.append(SWIN_TUNA_PATH)

from mmseg.models.decode_heads import UPerHead
from swin_tuna.modeling.tuna import Tuna
from swin_tuna.modeling.tuna_injector import inject_attributes
from swin_tuna.utils.model_utils import freeze_model, c2_xavier_fill
from timm.models.swin_transformer_v2 import window_partition, window_reverse

# --- CONFIGURATION (Must match your training script) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 384
num_labels = 104
CHECKPOINT_PATH = "sota_checkpoints/sota_model_best_iter_1000_miou_0.0162.pth" # <-- UPDATE this to your best checkpoint
TEST_IMAGE_PATH = "1.png" # <--- IMPORTANT: SET THIS PATH

# --- MODEL DEFINITION (Copied directly from your FINAL train.py) ---

# --- START OF CRITICAL FIX: Use the correct forward pass with scaling ---
def tuna_swin_v2_block_forward_definitive(self, x):
    is_4d = x.ndim == 4
    if is_4d: B, H, W, C = x.shape; x = x.reshape(B, H * W, C)
    B, L, C = x.shape; H = W = int(math.sqrt(L))
    identity = x; x_norm1 = self.norm1(x)
    x_norm1_reshaped = x_norm1.reshape(B, H, W, C)
    x_windows = window_partition(x_norm1_reshaped, self.window_size)
    x_windows = x_windows.reshape(-1, self.window_size[0] * self.window_size[1], C)
    attn_windows = self.attn(x_windows, mask=self.attn_mask)
    attn_windows = attn_windows.reshape(-1, self.window_size[0], self.window_size[1], C)
    x_attn = window_reverse(attn_windows, self.window_size, (H, W))
    x_attn = x_attn.reshape(B, H * W, C)
    x_tuna_attn = self.tuna_1(x_norm1, (H, W))
    raw_attn = identity + self.drop_path1(x_attn)
    x = self.tuna_x_scale1 * raw_attn + self.tuna_scale1 * x_tuna_attn
    identity = x; x_norm2 = self.norm2(x); x_mlp = self.mlp(x_norm2)
    x_tuna_ffn = self.tuna_2(x_norm2, (H, W))
    raw_ffn = identity + self.drop_path2(x_mlp)
    x = self.tuna_x_scale2 * raw_ffn + self.tuna_scale2 * x_tuna_ffn
    if is_4d: x = x.reshape(B, H, W, C)
    return x
# --- END OF CRITICAL FIX ---

def inject_tuna_into_timm_swinv2(target_model: nn.Module):
    stages = [target_model.layers_0, target_model.layers_1, target_model.layers_2, target_model.layers_3]
    conv_size_list = [5, 5, 5, 3]; hidden_dim_list = [64, 64, 96, 192]
    for i, stage in enumerate(stages):
        for block in stage.blocks:
            dim = block.mlp.fc1.in_features
            inject_attributes(block, conv_size=conv_size_list[i], dim=dim, hidden_dim=hidden_dim_list[i])
            block.forward = tuna_swin_v2_block_forward_definitive.__get__(block, type(block))

class SOTAModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__(); self.backbone = backbone; self.head = head
    def forward(self, x):
        features = self.backbone(x)
        features_nchw = [feat.permute(0, 3, 1, 2) for feat in features]
        logits = self.head.forward(features_nchw)
        return logits

# --- START OF OPTIMIZATION: Upgraded inference function with TTA ---
def run_inference_with_tta(model, image_path, transform, device):
    """Loads an image, preprocesses it, and runs inference with TTA."""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    
    # Preprocess the original image
    transformed_original = transform(image=np.array(image))
    input_tensor_original = transformed_original['image'].unsqueeze(0).to(device)
    
    # Preprocess the flipped image
    input_tensor_flipped = torch.flip(input_tensor_original, dims=[3])
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            # 1. Forward pass on original image
            output_logits_original = model(input_tensor_original)
            
            # 2. Forward pass on flipped image
            output_logits_flipped = model(input_tensor_flipped)

    # 3. Flip the flipped output back to the original orientation
    output_logits_flipped_restored = torch.flip(output_logits_flipped, dims=[3])

    # 4. Average the probabilities
    final_prob = (torch.softmax(output_logits_original, dim=1) + torch.softmax(output_logits_flipped_restored, dim=1)) / 2.0
    
    # 5. Upsample and get final mask
    final_prob_upsampled = nn.functional.interpolate(
        final_prob, size=original_size[::-1], mode='bilinear', align_corners=False
    )
    predicted_mask = torch.argmax(final_prob_upsampled, dim=1).squeeze(0).cpu().numpy()
    
    return image, predicted_mask
# --- END OF OPTIMIZATION ---

if __name__ == '__main__':
    print("--- Setting up model for inference ---")
    
    model_name = "swinv2_large_window12to24_192to384"
    backbone = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3), img_size=IMG_SIZE)
    backbone_feature_channels = backbone.feature_info.channels()
    inject_tuna_into_timm_swinv2(backbone)
    head = UPerHead(in_channels=backbone_feature_channels, in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6), channels=512, dropout_ratio=0.1, num_classes=num_labels, norm_cfg=dict(type='BN', requires_grad=True), align_corners=False)
    model = SOTAModel(backbone, head)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
        sys.exit(1)
        
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.to(DEVICE)
    model.eval()
    
    print(f"--- Model loaded from {CHECKPOINT_PATH} ---")

    inference_transform = A.Compose([
        A.Resize(width=IMG_SIZE, height=IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image not found at {TEST_IMAGE_PATH}")
        print("Please update the TEST_IMAGE_PATH variable in this script.")
        sys.exit(1)
        
    original_image, prediction = run_inference_with_tta(model, TEST_IMAGE_PATH, inference_transform, DEVICE)
    
    print("--- Displaying results ---")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    ax[1].imshow(prediction, cmap='jet')
    ax[1].set_title("Predicted Segmentation Mask (with TTA)")
    ax[1].axis('off')
    
    plt.show()