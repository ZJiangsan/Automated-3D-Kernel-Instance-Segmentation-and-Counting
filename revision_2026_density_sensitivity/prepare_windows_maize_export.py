"""Prepare a self-contained COB-1 density-threshold export from the published code.
Run this in the existing Windows nerfstudio (1.1.5, tcnn) environment.
The original script is fetched from the authors' GitHub repository unless a local copy is given.
"""
from pathlib import Path
import argparse
import urllib.request

URL = "https://raw.githubusercontent.com/ZJiangsan/Automated-3D-Kernel-Instance-Segmentation-and-Counting/main/export_pointcloud.py"
BASE = Path(r"V:\Databaser\LT\exchange\jizh\backup_nas\cob_new_shit")

TAIL = r'''
# COB-1 sensitivity export; generated from original research code.
# Do not remove the __main__ guard: ParallelDataManager uses multiprocessing.
def run_cob1_sensitivity():
    import gc
    import time
    import numpy as np
    import torch
    import open3d as o3d
    from pathlib import Path
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
    from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig, ParallelDataManager
    from nerfstudio.cameras.rays import RayBundle
    from nerfstudio.data.scene_box import SceneBox

    base = Path(r"V:\Databaser\LT\exchange\jizh\backup_nas\cob_new_shit")
    folder_i = "cob1_out"
    folder_path = base / folder_i
    checkpoint = base / f"best_checkpoint_nerfacto_cobNew{folder_i}Seg_RGB_pc_001.pth"
    outdir = base / "density_sensitivity_windows" / folder_i
    outdir.mkdir(parents=True, exist_ok=True)
    thresholds = (40, 50, 60, 70, 80, 90, 100)
    assert folder_path.is_dir(), f"Missing folder: {folder_path}"
    assert checkpoint.is_file(), f"Missing checkpoint: {checkpoint}"
    print("Dataset:", folder_path, flush=True)
    print("Checkpoint:", checkpoint, flush=True)
    print("Output directory:", outdir, flush=True)

    dc = NerfstudioDataParserConfig()
    dc.data = folder_path
    dc.downscale_factor = 1
    dc.train_split_fraction = 0.99
    dm_cfg = ParallelDataManagerConfig()
    dm_cfg.dataparser.data = folder_path
    dm_cfg.dataparser = dc
    dm_cfg.train_num_rays_per_batch = 1024 * 10
    dm_cfg.eval_num_rays_per_batch = 1024 * 10
    dm_cfg.camera_res_scale_factor = 1
    manager = ParallelDataManager(dm_cfg)

    img_num = len(os.listdir(folder_path / "images"))
    cfg = NerfactoModelConfig()
    bounds = torch.tensor([[-1., -1., -1.], [1., 1., 1.]])
    model = NerfactoModel(cfg, SceneBox(aabb=bounds), num_train_data=img_num, training=True).cuda()
    load_checkpoint_multiple(str(checkpoint), model)
    # Keep training=True during model construction to match the original code.
    # No gradients are calculated during export.
    print("CHECKPOINT LOADED SUCCESSFULLY", flush=True)

    torch.set_grad_enabled(False)
    points_chunks, rgb_chunks, density_chunks = [], [], []
    sampled_total = 0
    iteration = 0
    begin = time.time()
    while sampled_total < 1_000_000:  # original export's progress target
        ray_bundle_train, _batch = manager.next_train(0)
        n = ray_bundle_train.origins.shape[0]
        ray_bundle = RayBundle(
            origins=ray_bundle_train.origins.float().cuda(),
            directions=ray_bundle_train.directions.float().cuda(),
            nears=torch.full((n, 1), 0.01, dtype=torch.float32, device="cuda"),
            fars=torch.full((n, 1), 100., dtype=torch.float32, device="cuda"),
            pixel_area=ray_bundle_train.pixel_area.float().cuda(),
            camera_indices=ray_bundle_train.camera_indices.long().cuda(),
        )
        with torch.no_grad():
            outputs = model.get_export_outputs(ray_bundle)
            xyz = outputs["point_location"].reshape(-1, 3)
            density = outputs["density"].reshape(-1)
            rgb = outputs["rgb"].reshape(-1, 3)
            # Identical spatial crop to original export_pointcloud.py
            lo = torch.tensor([-0.5, -0.5, -0.7], device=xyz.device)
            hi = torch.tensor([0.5, 0.5, 0.9], device=xyz.device)
            keep = ((xyz >= lo) & (xyz <= hi)).all(dim=1) & (density >= 40)
            points_chunks.append(xyz[keep].cpu().numpy())
            rgb_chunks.append(rgb[keep].cpu().numpy())
            density_chunks.append(density[keep].cpu().numpy())
            sampled_total += outputs["point_location"].shape[0]
        iteration += 1
        print(f"Batch {iteration}: sampled={sampled_total:,}, retained >=40={sum(map(len, density_chunks)):,}", flush=True)
        del outputs, ray_bundle, ray_bundle_train
    xyz_all = np.concatenate(points_chunks, axis=0)
    rgb_all = np.concatenate(rgb_chunks, axis=0)
    den_all = np.concatenate(density_chunks, axis=0)
    del points_chunks, rgb_chunks, density_chunks
    print(f"All candidate points: {len(den_all):,}; elapsed {time.time()-begin:.1f}s", flush=True)
    summary = []
    for t in thresholds:
        mask = den_all >= t
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_all[mask].astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(rgb_all[mask].astype(np.float64))
        destination = outdir / f"{folder_i}SEG_density_pc_001_threshold_{t:03d}.ply"
        assert o3d.io.write_point_cloud(str(destination), pcd), f"Failed writing {destination}"
        print(f"Threshold {t:3d}: {mask.sum():,} points -> {destination}", flush=True)
        summary.append((t, int(mask.sum()), str(destination)))
    import csv
    with open(outdir / "point_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["density_threshold", "n_points", "pointcloud_path"])
        w.writerows(summary)
    print("COB-1 EXPORT COMPLETE. Original .ply unchanged.", flush=True)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_cob1_sensitivity()
'''

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--source", type=Path, help="Local copy of original export_pointcloud.py (optional)")
    pa.add_argument("--output", type=Path, default=Path.cwd() / "export_cob1_density_windows.py")
    args = pa.parse_args()
    if args.source:
        source = args.source.read_text(encoding="utf-8")
    else:
        print("Retrieving original author code from GitHub...")
        with urllib.request.urlopen(URL, timeout=30) as response:
            source = response.read().decode("utf-8-sig")
    start = source.find("\nbase_dir = 'cob_new'")
    if start == -1:
        raise RuntimeError("Original code changed; cannot identify export loop. Nothing written.")
    prefix = source[:start]
    # GitHub copy contains two misplaced future imports. Keep exactly one, at the beginning.
    prefix = "from __future__ import annotations\n" + "\n".join(
        line for line in prefix.splitlines() if line.strip() != "from __future__ import annotations"
    ) + "\n"
    out = prefix + TAIL
    compile(out, str(args.output), "exec")
    args.output.write_text(out, encoding="utf-8")
    print("Created:", args.output)
    print("Run in Spyder using %run", args.output)
    print("First checks: checkpoint loads and threshold-70 point cloud is produced.")

if __name__ == "__main__":
    main()