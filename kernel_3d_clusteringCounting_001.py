#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 11:14:38 2025

@author: nibio
"""



import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from skimage.measure import label
from scipy.ndimage import gaussian_filter

# ============================================================
# === MULTI-STAGE DBSCAN REFINEMENT (MAX-BRIGHTNESS RULE) ====
# ============================================================

KERNEL_MAX_POINTS = 200   # threshold A

def refine_large_cluster(pts, cols, eps=0.0035, min_pts=25, depth=1):

    if len(pts) < KERNEL_MAX_POINTS or depth > 3:
        return [(pts, cols)]

    brightness = cols.max(axis=1)

    base_p = 25 if depth == 1 else 30
    ratio = len(pts) / KERNEL_MAX_POINTS

    if ratio < 1.5:       p = base_p - 5
    elif ratio < 2.5:     p = base_p
    else:                 p = base_p + 5

    p = np.clip(p, 25, 60)
    thr = np.percentile(brightness, p)

    mask = brightness > thr
    pts_f = pts[mask]
    cols_f = cols[mask]

    if len(pts_f) < min_pts:
        return [(pts, cols)]

    labels = DBSCAN(eps=eps, min_samples=min_pts).fit(pts_f).labels_
    K = labels.max() + 1

    if K <= 1:
        return [(pts, cols)]

    refined = []
    for k in range(K):
        pts_k = pts_f[labels == k]
        cols_k = cols_f[labels == k]
        refined.extend(refine_large_cluster(pts_k, cols_k, eps, min_pts, depth+1))

    return refined



def refine_2nd(refined_clusters, eps=0.0025, min_pts=20):
    """
    Second-stage refinement:
    - operates ONLY on final clusters from refine_large_cluster
    - splits only obviously oversized clusters
    """

    if len(refined_clusters) == 0:
        return []

    # --------------------------------------------------
    # Compute cluster sizes
    # --------------------------------------------------
    sizes = np.array([len(p) for p, _ in refined_clusters])

    # remove tiny noise before median
    valid = sizes >= min_pts
    if valid.sum() == 0:
        return refined_clusters

    median_size = np.median(sizes[valid])

    new_clusters = []

    for pts, cols in refined_clusters:

        # ----------------------------------------------
        # Normal kernel → keep
        # ----------------------------------------------
        if len(pts) <= 2.55 * median_size:
            new_clusters.append((pts, cols))
            continue

        # ----------------------------------------------
        # Oversized kernel → force split
        # ----------------------------------------------
        labels = DBSCAN(
            eps=eps,
            min_samples=min_pts
        ).fit(pts).labels_

        valid_labels = [l for l in set(labels) if l >= 0]

        # DBSCAN failed → fallback PCA cut
        if len(valid_labels) <= 1:
            pts_c = pts - pts.mean(axis=0)
            _, _, vh = np.linalg.svd(pts_c, full_matrices=False)
            axis = vh[0]
            t = pts_c @ axis
            cut = np.median(t)

            m1 = t <= cut
            m2 = t > cut

            if m1.sum() >= min_pts and m2.sum() >= min_pts:
                new_clusters.append((pts[m1], cols[m1]))
                new_clusters.append((pts[m2], cols[m2]))
            else:
                new_clusters.append((pts, cols))

        else:
            for l in valid_labels:
                p = pts[labels == l]
                c = cols[labels == l]
                if len(p) >= min_pts:
                    new_clusters.append((p, c))

    return new_clusters






# ============================================================
# === COLUMN COUNTING ON FINAL LABELED POINT CLOUD ============
# ============================================================

def estimate_kernel_columns_and_save_from_final(out_points, base_dir, folder_i):

    pts = np.asarray(out_points)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    # restrict to biological mid-section
    z_min, z_max = np.percentile(z, [25, 75])
    mid_mask = (z > z_min) & (z < z_max)

    x_mid = x[mid_mask]
    y_mid = y[mid_mask]
    z_mid = z[mid_mask]

    if len(z_mid) < 500:
        print("Not enough mid-section points for column counting.")
        return 0

    n_slices = 10
    zs = np.linspace(z_min, z_max, n_slices)

    slice_counts = []
    slice_masks = []

    for i in range(n_slices - 1):
        smask = (z_mid >= zs[i]) & (z_mid < zs[i+1])
        if smask.sum() < 200:
            continue

        xs = x_mid[smask]
        ys = y_mid[smask]

        H, _, _ = np.histogram2d(xs, ys, bins=256)
        Hs = gaussian_filter(H, sigma=2)

        mask2d = Hs > (0.3 * Hs.max())
        labeled = label(mask2d)
        n_regions = labeled.max()

        slice_counts.append(n_regions)
        slice_masks.append(mask2d)

    if len(slice_counts) == 0:
        return 0
    #
    print("column counts {}".format(slice_counts))
    # final column estimate = median of slice counts
    final_cols = int(round(np.median(slice_counts)))

    # pick slice closest to final column count
    diffs = [abs(c-final_cols) for c in slice_counts]
    chosen_idx = np.argmin(diffs)
    chosen_mask = slice_masks[chosen_idx]

    # save slice image
    save_path = os.path.join(base_dir, f"{folder_i}_columnSlice_18022026.png")
    plt.figure(figsize=(5,5))
    plt.imshow(chosen_mask, cmap="gray", origin="lower")
    # plt.title(f"Representative slice (columns={final_cols})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved representative slice: {save_path}")

    return final_cols



# ============================================================
# === MAIN PIPELINE ==========================================
# ============================================================

base_dir = 'cob_new'
folder_ls = [
    name for name in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, name)) and name.endswith("_out")
]


kernel_counts = []
column_counts = []

for folder_i in folder_ls:
    print("====================================")
    print("folder_i =", folder_i)
    folder_path = os.path.join(base_dir, folder_i)

    pcd_file_i = f"{folder_i}SEG_density_pc_001.ply"
    point_path = os.path.join(folder_path, pcd_file_i)

    # Load
    pcd = o3d.io.read_point_cloud(point_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # ----------------------------------------------------------
    # 1. REMOVE DARK POINTS
    # ----------------------------------------------------------
    brightness = colors.max(axis=1)
    mask_not_dark = brightness > 0.65
    points = points[mask_not_dark]
    colors = colors[mask_not_dark]

    # ----------------------------------------------------------
    # 2. PCA ALIGNMENT
    # ----------------------------------------------------------
    center = points.mean(axis=0)
    pts_centered = points - center

    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis /= np.linalg.norm(axis)

    z_axis = np.array([0,0,1])
    v = np.cross(axis, z_axis)
    s = np.linalg.norm(v)
    c = np.dot(axis, z_axis)

    if s < 1e-6:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s*s + 1e-9))

    points_aligned = pts_centered @ R.T
    colors_aligned = colors.copy()

    # ensure Z is longest
    ranges = np.ptp(points_aligned, axis=0)
    max_dim = np.argmax(ranges)
    if max_dim != 2:
        perm = [0,1,2]
        perm[2], perm[max_dim] = perm[max_dim], perm[2]
        points_aligned = points_aligned[:, perm]

    # ----------------------------------------------------------
    # 3. RECENTER Z
    # ----------------------------------------------------------
    z = points_aligned[:, 2]
    zmid = 0.5 * (z.min() + z.max())
    points_aligned[:, 2] -= zmid

    # ----------------------------------------------------------
    # 4. REGION SLICING
    # ----------------------------------------------------------
    z = points_aligned[:, 2]
    zmin, zmax = z.min(), z.max()
    dz = zmax - zmin

    z1 = -0.4 * dz
    z2 =  0.4 * dz

    brightness2 = colors_aligned.max(axis=1)

    # ----------------------------------------------------------
    # 5. DENSITY + BRIGHTNESS FILTERING
    # ----------------------------------------------------------
    nn = NearestNeighbors(n_neighbors=5, radius=0.004).fit(points_aligned)
    distances, _ = nn.kneighbors(points_aligned)
    local_density = np.sum(distances < 0.004, axis=1)

    tip_vals  = brightness2[z >  z2]
    mid_vals  = brightness2[(z >= z1) & (z <= z2)]
    butt_vals = brightness2[z <  z1]

    T_tip    = np.percentile(tip_vals, 30) if len(tip_vals)>20 else 0.70
    T_middle = np.percentile(mid_vals, 30) if len(mid_vals)>20 else 0.70
    T_butt   = np.percentile(butt_vals, 30) if len(butt_vals)>20 else 0.70

    mask_tip     = (z >  z2) & (brightness2 > T_tip)    & (local_density > 3)
    mask_middle  = (z >= z1) & (z <= z2) & (brightness2 > T_middle) & (local_density > 3)
    mask_butt    = (z <  z1) & (brightness2 > T_butt)    & (local_density > 3)

    mask_filtered = mask_tip | mask_middle | mask_butt
    pts_filtered = points_aligned[mask_filtered]
    col_filtered = colors_aligned[mask_filtered]

    print("After filtering:", len(pts_filtered))

    # ----------------------------------------------------------
    # 6. STAGE-1 DBSCAN
    # ----------------------------------------------------------
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(pts_filtered)
    labels = np.array(pcd_tmp.cluster_dbscan(eps=0.0035, min_points=25))

    K1 = labels.max() + 1
    print("Initial clusters:", K1)

    # ----------------------------------------------------------
    # 7. MULTI-STAGE REFINEMENT
    # ----------------------------------------------------------
    refined_clusters = []
    for k in range(K1):
        pts_k = pts_filtered[labels == k]
        col_k = col_filtered[labels == k]

        if len(pts_k) == 0:
            continue

        if len(pts_k) > KERNEL_MAX_POINTS:
            # refined_clusters.extend(refine_large_cluster(pts_k, col_k))
            stage1 = refine_large_cluster(pts_k, col_k)
            stage2 = refine_2nd(stage1)
            refined_clusters.extend(stage2)

        else:
            refined_clusters.append((pts_k, col_k))

    final_cluster_count = len(refined_clusters)
    kernel_counts.append(final_cluster_count)

    print("Final kernel clusters:", final_cluster_count)

    # ----------------------------------------------------------
    # 8. SAVE LABELED PLY (FINISHED KERNEL SET)
    # ----------------------------------------------------------
    all_pts = []
    all_cols = []
    all_labels = []

    for i, (p, c) in enumerate(refined_clusters):
        all_pts.append(p)
        all_cols.append(c)
        all_labels.append(np.full(len(p), i))

    all_pts = np.concatenate(all_pts)
    all_cols = np.concatenate(all_cols)
    all_labels = np.concatenate(all_labels)

    cmap = plt.get_cmap("tab20")
    rgb = np.zeros((len(all_labels), 3))
    for i in range(final_cluster_count):
        rgb[all_labels == i] = cmap(i % 20)[:3]

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(all_pts)
    out.colors = o3d.utility.Vector3dVector(rgb)

    save_path = os.path.join(base_dir, f"refined_labeled_18022026_{pcd_file_i}")
    o3d.io.write_point_cloud(save_path, out)
    print("Saved:", save_path)

    # ----------------------------------------------------------
    # 9. COLUMN COUNTING ON FINAL POINT CLOUD
    # ----------------------------------------------------------
    out_points = np.asarray(out.points)
    col_count = estimate_kernel_columns_and_save_from_final(out_points, base_dir, folder_i)
    column_counts.append(col_count)

    print("Estimated kernel columns:", col_count)


print("\nKernel counts:", kernel_counts)
print("Column counts:", column_counts)



print("cob list pred = {}".format(kernel_counts))
cob_count_gt = [716, 337, 729, 314, 539, 463, 659, 496, 724]
print("cob list gt = {}".format(cob_count_gt))

print("\nPred:", kernel_counts)
print("GT  :", cob_count_gt)
if len(kernel_counts) == len(cob_count_gt):
    diff = np.array(cob_count_gt) - np.array(kernel_counts)
    print("Diff:", diff)
    print("kernel MAE :", np.mean(np.abs(diff)))
    


column_counts_gt = [20,14,18,17,16,18,18,16,18]
print("Column counts:", column_counts)
print("Column column_counts_gt:", column_counts_gt)
column_mae = np.mean(np.abs(np.array(column_counts_gt) - np.array(column_counts)))
print("Column count MAE :", column_mae)




