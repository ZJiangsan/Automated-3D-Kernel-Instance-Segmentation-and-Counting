#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 22:21:13 2025

@author: nibio
"""




import os, json, random, shutil
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations as A
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp


# ============================================================
# DIRECTORIES
# ============================================================
SRC_IMG_DIR = "kernel_annoation_new/images"
SRC_LAB_DIR = "kernel_annoation_new/labels"

PROC_IMG_DIR = "kernel_processed/images"
PROC_MSK_DIR = "kernel_processed/masks"

TRAIN_DIR = "kernel_dataset_split/train"
VAL_DIR   = "kernel_dataset_split/val"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# REFLECTION PADDING — ensure divisible by 32 (for U-Net)
# ============================================================
def pad_to_multiple_of_32_reflect(im):
    h, w = im.shape[:2]
    new_h = ((h + 31) // 32) * 32
    new_w = ((w + 31) // 32) * 32

    pad_bottom = new_h - h
    pad_right  = new_w - w

    if pad_bottom == 0 and pad_right == 0:
        return im

    return cv2.copyMakeBorder(
        im,
        0, pad_bottom,
        0, pad_right,
        borderType=cv2.BORDER_REFLECT_101
    )

def resize_keep_aspect_max_1024(im):
    h, w = im.shape[:2]
    max_side = max(h, w)

    # If already smaller than 1024, no change
    if max_side <= 1024:
        return im

    scale = 1024 / max_side
    new_w = int(w * scale)
    new_h = int(h * scale)

    im_resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return im_resized

# ============================================================
# CREATE NEEDED DIRECTORIES
# ============================================================
def ensure_dirs():
    for d in [PROC_IMG_DIR, PROC_MSK_DIR,
              f"{TRAIN_DIR}/images", f"{TRAIN_DIR}/masks",
              f"{VAL_DIR}/images", f"{VAL_DIR}/masks"]:
        os.makedirs(d, exist_ok=True)


# ============================================================
# LABELME JSON → BINARY MASK
# ============================================================
def labelme_to_mask(json_path, h, w):
    data = json.load(open(json_path))
    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data["shapes"]:
        pts = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)

    return mask


# ============================================================
# AUGMENTATION — NO ROTATE90 + strong color jitter
# ============================================================
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),

    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.15,
        rotate_limit=20,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.5
    ),

    # Color augmentations (fix "yellow confusion")
    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.5
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.25,
        contrast_limit=0.25,
        p=0.5
    ),
    A.CLAHE(clip_limit=3.0, p=0.25),
    A.RandomGamma(gamma_limit=(80,120), p=0.3),
    A.RGBShift(
        r_shift_limit=10,
        g_shift_limit=10,
        b_shift_limit=10,
        p=0.3
    ),
    A.ChannelDropout(p=0.1),

    # Texture robustness
    A.GaussianBlur(p=0.2),
    A.GaussNoise(p=0.2),
])


# ============================================================
# PREPROCESSING — resize ×0.5, augment, reflect-pad
# ============================================================
def preprocess():
    print("=== Preprocessing images ===")

    for fname in tqdm(os.listdir(SRC_IMG_DIR)):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(SRC_IMG_DIR, fname)
        json_path = os.path.join(SRC_LAB_DIR, fname.rsplit(".",1)[0] + ".json")

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        mask = labelme_to_mask(json_path, h, w)

        # Resize ×0.5
        img_rs = cv2.resize(img, (w//2, h//2))
        mask_rs = cv2.resize(mask, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
        
        # Resize so max dimension = 1024 while keeping aspect ratio
        img_rs = resize_keep_aspect_max_1024(img_rs)
        mask_rs = resize_keep_aspect_max_1024(mask_rs)


        # Reflect-pad original (no black borders)
        img_rs = pad_to_multiple_of_32_reflect(img_rs)
        mask_rs = pad_to_multiple_of_32_reflect(mask_rs)

        base = fname.split('.')[0]

        # Save original resized image
        cv2.imwrite(f"{PROC_IMG_DIR}/{base}_orig.png", img_rs)
        cv2.imwrite(f"{PROC_MSK_DIR}/{base}_orig.png", mask_rs)

        # Augment 40× per image
        for i in range(40):
            out = augment(image=img_rs, mask=mask_rs)
            ai = out["image"]
            am = out["mask"]

            # Reflect-pad augmented
            ai = pad_to_multiple_of_32_reflect(ai)
            am = pad_to_multiple_of_32_reflect(am)

            cv2.imwrite(f"{PROC_IMG_DIR}/{base}_aug{i}.png", ai)
            cv2.imwrite(f"{PROC_MSK_DIR}/{base}_aug{i}.png", am)


# ============================================================
# SPLIT DATASET INTO TRAIN / VAL
# ============================================================
def split_dataset():
    print("=== Splitting dataset ===")
    images = sorted(glob(f"{PROC_IMG_DIR}/*.png"))
    random.shuffle(images)

    val_count = int(len(images) * 0.2)

    val_set = images[:val_count]
    train_set = images[val_count:]

    def copy_pairs(files, dst_root):
        for img_path in files:
            base = os.path.basename(img_path)
            mask_path = os.path.join(PROC_MSK_DIR, base)
            shutil.copy(img_path, f"{dst_root}/images/{base}")
            shutil.copy(mask_path, f"{dst_root}/masks/{base}")

    copy_pairs(train_set, TRAIN_DIR)
    copy_pairs(val_set,   VAL_DIR)

    print("Train:", len(train_set), "Val:", len(val_set))


# ============================================================
# DATASET CLASS
# ============================================================
class SegDataset(Dataset):
    def __init__(self, root):
        self.imgs = sorted(glob(f"{root}/images/*.png"))
        self.msks = sorted(glob(f"{root}/masks/*.png"))
        self.transform = A.Compose([A.Normalize()])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        msk = cv2.imread(self.msks[idx], 0)

        t = self.transform(image=img, mask=msk)
        img, msk = t["image"], t["mask"]

        img = torch.tensor(img).permute(2,0,1).float()
        msk = torch.tensor(msk).float()
        return img, msk


# ============================================================
# BCE + Dice Loss
# ============================================================
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        pred = torch.sigmoid(pred)

        smooth = 1
        inter = (pred * target).sum()
        dice = 1 - (2*inter + smooth) / (pred.sum() + target.sum() + smooth)

        return bce + dice


# ============================================================
# VISUALIZATION
# ============================================================
def visualize(model, dataset, epoch, name="val", save_dir="kenelSegNov_predictions"):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    img, msk = dataset[0]
    img_in = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_in))[0,0].cpu().numpy()

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.title("Image");     plt.imshow(img.permute(1,2,0).cpu().numpy())
    plt.subplot(1,3,2); plt.title("GT Mask");   plt.imshow(msk.cpu(), cmap="gray")
    plt.subplot(1,3,3); plt.title("Prediction"); plt.imshow(pred > 0.5, cmap="gray")

    plt.savefig(f"{save_dir}/{name}_epoch_{epoch}.png")
    plt.show()
    plt.close()



# ============================================================
# MAIN
# ============================================================
# if __name__ == "__main__":
ensure_dirs()
preprocess()
split_dataset()

####
print("=== Training U-Net ===")

train_ds = SegDataset(TRAIN_DIR)
val_ds   = SegDataset(VAL_DIR)

train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=4)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
).to(DEVICE)


## load the model 
# model.load_state_dict(torch.load("kernelSeg_best_unet_001.pth", map_location=DEVICE))


criterion = BCEDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.9, patience=5, verbose=True
)

best_val = float("inf")
patience = 0
EARLY_STOP = 100
EPOCHS = 30000000000

for epoch in range(1, EPOCHS+1):
    # ----------------- TRAIN -----------------
    model.train()
    train_loss = 0

    for img, msk in train_dl:
        img, msk = img.to(DEVICE), msk.to(DEVICE)

        optimizer.zero_grad()
        pred = model(img)
        loss = criterion(pred[:,0,:,:], msk)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dl)

    # ----------------- VAL -----------------
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for img, msk in val_dl:
            img, msk = img.to(DEVICE), msk.to(DEVICE)
            pred = model(img)
            loss = criterion(pred[:,0,:,:], msk)
            val_loss += loss.item()

    val_loss /= len(val_dl)

    print(f"Epoch {epoch}: Train={train_loss:.4f}   Val={val_loss:.4f}")

    # LR step
    scheduler.step(val_loss)

    # Visualize every 20 epochs
    if epoch % 20 == 0:
        visualize(model, train_ds, epoch, "train")
        visualize(model, val_ds,   epoch, "val")

    # Early stopping
    if val_loss < best_val:
        best_val = val_loss
        patience = 0
        torch.save(model.state_dict(), "kernelSeg_best_unet_001.pth")
    else:
        patience += 1

    if patience >= EARLY_STOP:
        print("EARLY STOPPING TRIGGERED")
        break





#############
# visualize the augmentation effect

import os, json, random
import cv2
import numpy as np
import albumentations as A

# --- CONFIG ---
SRC_IMG_DIR = "kernel_annoation_new/images"
SRC_LAB_DIR = "kernel_annoation_new/labels"
N_AUG = 6  # number of augmented examples to show
OUTPUT_FILE = "augmentation_visualization.png"

# --- Same augmentation pipeline as training ---
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20,
                       border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
    A.CLAHE(clip_limit=3.0, p=0.25),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
    A.ChannelDropout(p=0.1),
    A.GaussianBlur(p=0.2),
    A.GaussNoise(p=0.2),
])

def labelme_to_mask(json_path, h, w):
    data = json.load(open(json_path))
    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in data["shapes"]:
        pts = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask

# --- Pick a random image ---
fnames = [f for f in os.listdir(SRC_IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
fname = random.choice(fnames)
print(f"Selected: {fname}")

img = cv2.imread(os.path.join(SRC_IMG_DIR, fname))
h, w = img.shape[:2]

json_path = os.path.join(SRC_LAB_DIR, fname.rsplit(".", 1)[0] + ".json")
mask = labelme_to_mask(json_path, h, w)

# Resize to manageable size for display
scale = 512 / max(h, w)
img = cv2.resize(img, (int(w * scale), int(h * scale)))
mask = cv2.resize(mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)

# --- Generate augmented samples ---
aug_imgs = []
aug_masks = []
for _ in range(N_AUG):
    out = augment(image=img, mask=mask)
    aug_imgs.append(out["image"])
    aug_masks.append(out["mask"])

# --- Layout: 2 rows (images on top, masks on bottom), N_AUG columns ---
all_imgs = aug_imgs
all_masks = aug_masks

# Pad all to same size
target_h = max(im.shape[0] for im in all_imgs)
target_w = max(im.shape[1] for im in all_imgs)

def pad_to(im, th, tw):
    ph = th - im.shape[0]
    pw = tw - im.shape[1]
    if len(im.shape) == 3:
        return np.pad(im, ((0, ph), (0, pw), (0, 0)), mode='constant')
    else:
        return np.pad(im, ((0, ph), (0, pw)), mode='constant')

padded_imgs = [pad_to(im, target_h, target_w) for im in all_imgs]
padded_masks = [pad_to(m, target_h, target_w) for m in all_masks]

# Convert masks to 3-channel for stacking
masks_3ch = [cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) for m in padded_masks]

row_top = np.hstack(padded_imgs)
row_bot = np.hstack(masks_3ch)
canvas = np.vstack([row_top, row_bot])

# Convert BGR to RGB for saving
canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
cv2.imwrite(OUTPUT_FILE, cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR))
print(f"Saved to {OUTPUT_FILE}")










