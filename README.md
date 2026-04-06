# Automated-3D-Kernel-Instance-Segmentation-and-Counting

Non-destructive 3D maize kernel counting and row estimation from handheld smartphone video using surface-supervised NeRF reconstruction and multi-stage clustering.


## Overview

This repository contains the full pipeline for automated 3D maize kernel phenotyping, as described in:

> **Non-Destructive 3D Maize Kernel Counting and Row Estimation from Handheld Video Using Surface-Supervised Reconstruction and Multi-Stage Clustering**
> Jiangsan Zhao, Xuean Cui

The pipeline takes handheld video of intact maize cobs as input and outputs per-cob kernel counts and kernel row numbers. It achieves a mean absolute error of 10.33 kernels and 0.44 rows across nine cobs ranging from 314 to 729 kernels.

## Pipeline Steps

```
1. Video → Frame extraction (ffmpeg)
2. Frames → Camera poses (COLMAP via ns-process-data)
3. Frames → 2D kernel surface masks (U-Net)
4. Frames + poses → RGB NeRF training
5. Masks + pre-trained NeRF → Segmentation NeRF fine-tuning
6. Segmentation NeRF → 3D point cloud export
7. Point cloud → Kernel instance clustering + row counting
```

## Requirements

- NVIDIA GPU with CUDA support (required for tiny-cuda-nn)
- Python 3.8+
- [Nerfstudio 1.1.5](https://docs.nerf.studio/)

Install dependencies:

```bash
pip install nerfstudio==1.1.5
pip install segmentation_models_pytorch albumentations open3d scikit-learn scikit-image scipy opencv-python matplotlib tqdm
```


## Data Preparation

### 1. Record video

Record a slow 360° handheld video of each maize cob using a smartphone (e.g., iPhone 13 Pro). Move from tip to butt. Indoor lighting is sufficient.

### 2. Extract frames

```bash
ffmpeg -i video.mp4 -vf fps=4 frames/frame_%05d.png
```

### 3. Run COLMAP for camera poses

Use Nerfstudio's data processing tool:

```bash
ns-process-data images --data frames/ --output-dir cob_data/cob1_out/ --camera-type pinhole
```

This creates the required `transforms.json` and `images/` folder.

### Expected folder structure

After data preparation, each cob folder should look like:

```
cob_data/
├── cob1_out/
│   ├── images/           # extracted frames
│   └── transforms.json   # camera poses
├── cob2_out/
│   ├── ...
...
```

## Usage

### Step 1: Annotate kernel surfaces

Manually annotate kernel surfaces in a small number of representative images (we used 3 images across all cobs) using [anylabeling](https://github.com/vietanhdev/anylabeling)). Save annotations as JSON files.

```
kernel_annotation_new/
├── images/
│   ├── frame_00026.png
│   ├── frame_00216.png
│   └── frame_00273.png
└── labels/
    ├── frame_00026.json
    ├── frame_00216.json
    └── frame_00273.json
```

### Step 2: Train U-Net segmentation model

```bash
python train_unet.py
```

This trains a U-Net (ResNet-34 encoder) on the annotated images with 40× data augmentation per image. The best model weights are saved as `kernelSeg_best_unet.pth`.

### Step 3: Generate segmentation masks for all cobs

```bash
python batch_inference.py
```

This runs the trained U-Net on all images in each cob folder and saves binary masks to a `semantics/` subfolder.

### Step 4: Train RGB NeRF

```bash
python train_nerf_RGBvsSeg.py
```

Trains a Nerfacto-based NeRF model on the original RGB images for each cob. Checkpoints are saved per cob.

### Step 5: Fine-tune NeRF on segmentation masks

Before running fine-tuning, replace the contents of each `images/` folder with the corresponding masks from `semantics/`, lower the learning rate:

```bash
# For each cob folder:
cp cob_data/cob1_out/semantics/* cob_data/cob1_out/images/
```

Then run:

```bash
python train_nerf_RGBvsSeg.py
```

This fine-tunes the pre-trained RGB NeRF to reconstruct binary segmentation masks, increasing density in kernel regions.

### Step 6: Export 3D point clouds

```bash
python export_pointcloud.py
```

Extracts dense point clouds from the segmentation NeRF with density thresholding (≥70).

### Step 7: Kernel clustering and row counting

```bash
python cluster_and_count.py
```

Performs PCA alignment, brightness filtering, multi-stage DBSCAN clustering, and radial midsection row counting. Outputs per-cob kernel counts and row estimates, and saves labeled point clouds as `.ply` files.

## Results

| Cob | Kernel Count (Pred) | Kernel Count (GT) | Absolute Error | Row Count (Pred) | Row Count (GT) | Row Error |
|-----|--------------------:|-------------------:|---------------:|-----------------:|---------------:|----------:|
| 1   | 707                 | 716                | 9              | 20               | 20             | 0         |
| 2   | 342                 | 337                | 5              | 15               | 14             | 1         |
| 3   | 735                 | 729                | 6              | 17               | 18             | 1         |
| 4   | 325                 | 314                | 11             | 18               | 17             | 1         |
| 5   | 535                 | 539                | 4              | 16               | 16             | 0         |
| 6   | 480                 | 463                | 17             | 18               | 18             | 0         |
| 7   | 651                 | 659                | 8              | 18               | 18             | 0         |
| 8   | 504                 | 496                | 8              | 17               | 16             | 1         |
| 9   | 699                 | 724                | 25             | 18               | 18             | 0         |

**Kernel count MAE: 10.33 | Row count MAE: 0.44**

## Citation

If you use this code in your research, please cite:

```bibtex
@article{zhao2026kernel,
  title={Non-Destructive 3D Maize Kernel Counting and Row Estimation from Handheld Video Using Surface-Supervised Reconstruction and Multi-Stage Clustering},
  author={Zhao, Jiangsan and Cui, Xuean},
  journal={},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).



This research was funded by the Research Council of Norway (RCN), grant numbers 344343 and 352849, and the Innovation Program of Chinese Academy of Agricultural Sciences (CAAS-ZDRW202407).
