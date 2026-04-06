#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:11:39 2026

@author: nibio
"""




import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from glob import glob
from tqdm import tqdm

# --- CONFIG ---
COB_ROOT = "cob_new_shit"
MODEL_PATH = "kernelSeg_best_unet_001.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load model ---
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- Helper: same preprocessing as training ---
def pad_to_multiple_of_32_reflect(im):
    h, w = im.shape[:2]
    new_h = ((h + 31) // 32) * 32
    new_w = ((w + 31) // 32) * 32
    pad_bottom = new_h - h
    pad_right = new_w - w
    if pad_bottom == 0 and pad_right == 0:
        return im, 0, 0
    padded = cv2.copyMakeBorder(im, 0, pad_bottom, 0, pad_right,
                                 borderType=cv2.BORDER_REFLECT_101)
    return padded, pad_bottom, pad_right

def resize_keep_aspect_max_1024(im):
    h, w = im.shape[:2]
    max_side = max(h, w)
    if max_side <= 1024:
        return im, 1.0
    scale = 1024 / max_side
    im_resized = cv2.resize(im, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA)
    return im_resized, scale

# --- Normalize (same as albumentations default) ---
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def normalize(img):
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img

# --- Process each cob ---
cob_dirs = sorted(glob(os.path.join(COB_ROOT, "cob*_out")))

for cob_dir in cob_dirs:
    img_dir = os.path.join(cob_dir, "images_or")
    out_dir = os.path.join(cob_dir, "semantics")
    os.makedirs(out_dir, exist_ok=True)

    img_files = sorted(glob(os.path.join(img_dir, "*.png")) +
                       glob(os.path.join(img_dir, "*.jpg")))

    cob_name = os.path.basename(cob_dir)
    print(f"\nProcessing {cob_name}: {len(img_files)} images")

    for img_path in tqdm(img_files):
        img = cv2.imread(img_path)
        orig_h, orig_w = img.shape[:2]

        # Resize x0.5
        img_rs = cv2.resize(img, (orig_w // 2, orig_h // 2))

        # Resize max 1024
        img_rs, scale = resize_keep_aspect_max_1024(img_rs)
        rs_h, rs_w = img_rs.shape[:2]

        # Pad
        img_pad, pad_b, pad_r = pad_to_multiple_of_32_reflect(img_rs)

        # Normalize and to tensor
        img_norm = normalize(img_pad)
        tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

        # Predict
        with torch.no_grad():
            pred = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

        # Remove padding
        if pad_b > 0:
            pred = pred[:-pad_b, :]
        if pad_r > 0:
            pred = pred[:, :-pad_r]

        # Upsample to original resolution
        mask = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary = (mask > 0.5).astype(np.uint8) * 255

        # Save
        out_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        cv2.imwrite(os.path.join(out_dir, out_name), binary)

    print(f"  Saved {len(img_files)} masks to {out_dir}")






