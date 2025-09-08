# train.py (Final Definitive Version with Cache Fix)

import os

# --- FINAL FIX: Change the Hugging Face cache directory ---
# Set the cache to a folder on your D: drive to avoid filling up C:
cache_dir = "D:/huggingface_cache"
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
# --- END FIX ---


# --- Final Imports ---
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR
import sys
from tqdm import tqdm
import numpy as np
import timm
from torch.utils.data import Dataset, DataLoader
import torchmetrics
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset
import math

# --- Add Cloned Repos to Python Path ---
SWIN_TUNA_PATH = os.path.abspath('./swin_tuna')
if SWIN_TUNA_PATH not in sys.path:
    sys.path.append(SWIN_TUNA_PATH)

from mmseg.models.decode_heads import UPerHead
from swin_tuna.modeling.tuna import Tuna
from swin_tuna.modeling.tuna_injector import inject_attributes
from swin_tuna.utils.model_utils import freeze_model, c2_xavier_fill
from timm.models.swin_transformer_v2 import window_partition, window_reverse

# --- SOTA TRAINING CONFIGURATION ---
BATCH_SIZE = 8
NUM_WORKERS = 18 # Number of CPU cores for data loading.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 384
num_labels = 104
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.05
LR_POWER = 0.9
MAX_ITERATIONS = 100000
EVAL_INTERVAL = 4000
CHECKPOINT_DIR = "./sota_checkpoints"


# --- OPTIMIZED & QUALITY-FOCUSED TRANSFORMS ---
CROP_PRE_SIZE = int(IMG_SIZE * 1.25)
pre_transform = A.Compose([
    A.Resize(width=CROP_PRE_SIZE, height=CROP_PRE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
train_transform_random = A.Compose([
    A.RandomCrop(width=IMG_SIZE, height=IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(width=IMG_SIZE, height=IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# PyTorch Dataset class
class FoodSegPyTorchDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, is_train=False):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        image_data = example['pixel_values'] if self.is_train else np.array(example['image'])
        label_data = example['label'] if self.is_train else np.array(example['label'])
        image = np.array(image_data, dtype=np.float32)
        label = np.array(label_data, dtype=np.uint8)
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask'].long()
        return image, label

# Function to apply the one-time pre-processing transform
def apply_pre_transform(batch):
    transformed_batch = {'pixel_values': [], 'label': []}
    for img, lbl in zip(batch['image'], batch['label']):
        transformed = pre_transform(image=np.array(img), mask=np.array(lbl))
        transformed_batch['pixel_values'].append(transformed['image'])
        transformed_batch['label'].append(transformed['mask'])
    return transformed_batch


# --- MODEL DEFINITION (remains the same) ---
def tuna_swin_v2_block_forward_definitive(self, x):
    is_4d = x.ndim == 4
    if is_4d: B, H, W, C = x.shape; x = x.reshape(B, H * W, C)
    B, L, C = x.shape; H = W = int(math.sqrt(L))
    window_size_tuple = self.window_size; window_area = window_size_tuple[0] * window_size_tuple[1]
    identity = x; x = self.norm1(x); x = x.reshape(B, H, W, C)
    x_windows = window_partition(x, window_size_tuple)
    x_windows = x_windows.reshape(-1, window_area, C)
    attn_windows = self.attn(x_windows, mask=self.attn_mask)
    attn_windows = attn_windows.reshape(-1, window_size_tuple[0], window_size_tuple[1], C)
    x = window_reverse(attn_windows, window_size_tuple, (H, W))
    x = x.reshape(B, H * W, C); x = identity + self.drop_path1(x)
    x_after_attn = self.tuna_1(x, (H, W)); x = x + self.drop_path1(x_after_attn)
    identity = x; x = self.norm2(x); x = self.mlp(x); x = identity + self.drop_path2(x)
    x_after_ffn = self.tuna_2(x, (H, W)); x = x + x_after_ffn
    if is_4d: x = x.reshape(B, H, W, C)
    return x

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

def main():
    # --- STEP 1: LOAD AND PRE-PROCESS DATA ---
    print("--- Loading Hugging Face dataset ---")
    # The cache_dir will be automatically used thanks to the environment variables
    ds = load_dataset("EduardoPacheco/FoodSeg103")
    
    print(f"\n--- Applying one-time pre-processing transform with {NUM_WORKERS} processes ---")
    print(f"--- Cache will be stored in: {os.environ['HF_DATASETS_CACHE']} ---")
    processed_train_ds = ds['train'].map(
        apply_pre_transform, batched=True, batch_size=100,
        num_proc=NUM_WORKERS,
        remove_columns=['image', 'id', 'classes_on_image']
    )
    print("--- Pre-processing complete ---")

    train_pytorch_dataset = FoodSegPyTorchDataset(processed_train_ds, transform=train_transform_random, is_train=True)
    val_pytorch_dataset = FoodSegPyTorchDataset(ds['validation'], transform=val_transform, is_train=False)
    
    train_dataloader = DataLoader(
        train_pytorch_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_pytorch_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    print(f"\nDataLoaders created with Batch Size: {BATCH_SIZE} and Num Workers: {NUM_WORKERS}")

    # --- STEP 2 & 3: BUILD MODEL & SETUP TRAINING ---
    print("\n--- Building and setting up the SOTA Model ---")
    model_name = "swinv2_large_window12to24_192to384"
    backbone = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3), img_size=IMG_SIZE)
    backbone_feature_channels = backbone.feature_info.channels()
    inject_tuna_into_timm_swinv2(backbone)
    head = UPerHead(in_channels=backbone_feature_channels, in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6), channels=512, dropout_ratio=0.1, num_classes=num_labels, norm_cfg=dict(type='BN', requires_grad=True), align_corners=False, loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    for m in head.modules():
        if isinstance(m, nn.Conv2d): c2_xavier_fill(m)
    model = SOTAModel(backbone, head)
    freeze_model(model)
    for name, param in model.named_parameters():
        if 'tuna' in name or 'head' in name: param.requires_grad = True
    
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    scheduler = PolynomialLR(optimizer, total_iters=MAX_ITERATIONS, power=LR_POWER)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    miou_metric = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_labels).to(DEVICE)

    # --- STEP 4: TRAINING & VALIDATION LOOP ---
    print("\n--- Starting SOTA Training Run ---")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_miou = 0.0
    train_iter = iter(train_dataloader)
    
    for i in tqdm(range(MAX_ITERATIONS), desc="Total Iterations"):
        model.train()
        try:
            images, masks = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            images, masks = next(train_iter)

        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            outputs = model(images)
            outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        
        if (i + 1) % EVAL_INTERVAL == 0:
            model.eval()
            miou_metric.reset()
            val_progress_bar = tqdm(val_dataloader, desc=f"Iter {i+1} [Validation]", leave=False)
            
            with torch.no_grad():
                for val_images, val_masks in val_progress_bar:
                    val_images, val_masks = val_images.to(DEVICE), val_masks.to(DEVICE)
                    with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                        val_outputs = model(val_images)
                        val_outputs = nn.functional.interpolate(val_outputs, size=val_masks.shape[-2:], mode='bilinear', align_corners=False)
                    preds = torch.argmax(val_outputs, dim=1)
                    miou_metric.update(preds, val_masks)

            miou = miou_metric.compute()
            print(f"\nIteration {i+1}/{MAX_ITERATIONS} -> Validation mIoU: {miou:.4f}")
            
            if miou > best_miou:
                best_miou = miou
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"sota_model_best_iter_{i+1}_miou_{miou:.4f}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"🚀 New best model saved to {checkpoint_path} with mIoU: {best_miou:.4f}")

    print("\n--- Training Complete ---")
    print(f"Best Validation mIoU achieved: {best_miou:.4f}")


if __name__ == '__main__':
    main()