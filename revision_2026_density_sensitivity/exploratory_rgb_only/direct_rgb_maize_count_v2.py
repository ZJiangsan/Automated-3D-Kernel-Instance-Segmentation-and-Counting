"""Second, independent attempt at maize kernel instance proposals from RGB NeRF 3D geometry.

No 2D segmentation masks, segmentation NeRF weights, ground-truth counts,
or count-based parameter tuning.  Input is the existing RGB-only NeRF PLY.

Spyder: %run C:/Users/jizh/direct_rgb_maize_count_v2.py
This script does not overwrite the first attempt's outputs.
"""
from pathlib import Path
import csv
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import (gaussian_filter, distance_transform_edt,
                           binary_closing, binary_opening, binary_fill_holes,
                           label as ndi_label, maximum_filter)
from scipy.spatial import cKDTree
from skimage.morphology import h_maxima, remove_small_objects, disk
from skimage.segmentation import watershed

ROOT = Path('V:/Databaser/LT/exchange/jizh/backup_nas/cob_new_shit')
COB = 'cob1_out'
FOLDER = ROOT / 'direct_rgb_3d' / COB
SOURCE = FOLDER / f'{COB}_RGB_only_density70.ply'
OUTPUT = FOLDER / 'rgb_outer_surface_v2'

# Fixed beforehand; none are optimized against the manual kernel count.
VOXEL = 0.0015
MAX_POINTS = 1_200_000
Z_BINS = 600
ANGLE_BINS = 720
MIN_SAMPLES_PER_CELL = 2
SURFACE_QUANTILE = 0.90   # Outer surface, robust to isolated points
SMOOTH_SIGMA = (1.2, 1.2)
BASELINE_SIGMA = (14.0, 18.0)
SEED_MIN_SPACING = 9     # Cylindrical grid pixels
MIN_COMPONENT_PIXELS = 25
MAX_COMPONENT_PIXELS = 4500


def periodic_filter(a, sigma):
    return gaussian_filter(a, sigma=sigma, mode=('nearest', 'wrap'))


def circular_fill(values, valid):
    """Nearest valid data in angular direction, periodic at theta=0 / 2pi."""
    if not np.any(valid):
        raise RuntimeError('No reliable cylindrical surface cells')
    tiled = np.concatenate([values, values, values], axis=1)
    tiled_valid = np.tile(valid, (1, 3))
    idx = distance_transform_edt(~tiled_valid, return_distances=False,
                                 return_indices=True)
    return tiled[tuple(idx)][:, ANGLE_BINS:2*ANGLE_BINS]


def cob_frame(points):
    # Prefer robust covariance based on a fixed subsample of points.
    center = np.median(points, axis=0)
    X = points-center
    rng = np.random.default_rng(21)
    sample = X[rng.choice(len(X), min(len(X), 170000), replace=False)]
    eigenvals, eigenvecs = np.linalg.eigh(np.cov(sample.T))
    axis = eigenvecs[:, np.argmax(eigenvals)]
    unit = np.eye(3)[np.argmin(np.abs(axis))]
    ex = unit - unit.dot(axis)*axis
    ex /= np.linalg.norm(ex)
    ey = np.cross(axis, ex)
    z = X @ axis
    u = X @ ex
    v = X @ ey
    theta = np.mod(np.arctan2(v, u), 2*np.pi)
    radius = np.hypot(u, v)
    return z, theta, radius


def binned_quantile(cell, value, n_cells, q):
    """Compute groupwise empirical quantile without allocating a list per pixel."""
    order = np.lexsort((value, cell))
    sorted_cells = cell[order]
    sorted_values = value[order]
    unique, start, counts = np.unique(sorted_cells, return_index=True, return_counts=True)
    # q-th order statistic (nearest rank). Results reflect actual point radii.
    idx = start + np.minimum((q * (counts-1)).astype(np.int64), counts-1)
    quantiles = np.full(n_cells, np.nan, dtype=np.float32)
    quantiles[unique] = sorted_values[idx]
    lengths = np.zeros(n_cells, np.int32)
    lengths[unique] = counts
    return quantiles, lengths


def export_labeled_ply(points, labels, output):
    # Fixed deterministic palette; label 0 is dark gray.
    rgb = np.full((len(points), 3), 0.14, dtype=np.float64)
    valid = labels > 0
    if np.any(valid):
        c = np.asarray(plt.get_cmap('turbo')((labels[valid] * 0.61803398875) % 1))[:, :3]
        rgb[valid] = c
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(rgb)
    if not o3d.io.write_point_cloud(str(output), cloud, write_ascii=False):
        raise RuntimeError('PLY write failed: ' + str(output))


def run():
    if not SOURCE.is_file():
        raise FileNotFoundError('RGB-only PLY not found: ' + str(SOURCE))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pcd = o3d.io.read_point_cloud(str(SOURCE))
    if not pcd.has_points() or not pcd.has_colors():
        raise RuntimeError('Source must contain XYZ and RGB from RGB-only NeRF')

    pcd = pcd.voxel_down_sample(VOXEL)
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if len(pts) > MAX_POINTS:
        ids = np.random.default_rng(25).choice(len(pts), MAX_POINTS, replace=False)
        pts, colors = pts[ids], colors[ids]
    print('Downsampled RGB points:', len(pts), flush=True)

    z, theta, rad = cob_frame(pts)
    zmin, zmax = np.quantile(z, [0.015, 0.985])
    ok = (z >= zmin) & (z <= zmax)
    pts, colors, z, theta, rad = (arr[ok] for arr in (pts, colors, z, theta, rad))
    iz = np.clip(((z-zmin)/(zmax-zmin)*Z_BINS).astype(np.int32), 0, Z_BINS-1)
    it = np.minimum((theta/(2*np.pi)*ANGLE_BINS).astype(np.int32), ANGLE_BINS-1)
    cell = iz * ANGLE_BINS + it
    ncell = Z_BINS * ANGLE_BINS

    radius_q, counts = binned_quantile(cell, rad.astype(np.float32), ncell, SURFACE_QUANTILE)
    outer = radius_q.reshape(Z_BINS, ANGLE_BINS)
    density = counts.reshape(Z_BINS, ANGLE_BINS)
    reliable = density >= MIN_SAMPLES_PER_CELL
    print('Reliable surface pixels:', int(reliable.sum()), '/', reliable.size, flush=True)
    surface = circular_fill(outer, reliable)

    # Remove cross-sectional taper before identifying local 3D bumps.
    # The longitudinal mean radius also suppresses angular appearance bias.
    lowfreq = periodic_filter(surface, BASELINE_SIGMA)
    fine = periodic_filter(surface, SMOOTH_SIGMA)
    relief = fine - lowfreq
    # Convert to robust standard deviations to keep scale independent of cob size.
    med = float(np.median(relief[reliable]))
    mad = float(np.median(np.abs(relief[reliable] - med)))
    robust_std = max(1.4826 * mad, 1.e-7)
    relief_z = (relief-med)/robust_std

    # Only above-local-background peaks are kernel candidates. H-maxima removes
    # weak local maxima; spacing is enforced independently of the expected count.
    # More conservative than arbitrary thresholding that yielded 1074 fragments.
    prominence = 1.0
    maxima = h_maxima(relief_z, prominence)
    candidate = maxima & (relief_z > 0.65)
    peak_vals = np.where(candidate, relief_z, -np.inf)
    localmax = maximum_filter(peak_vals, size=(2*SEED_MIN_SPACING+1,
                                                2*SEED_MIN_SPACING+1),
                              mode=('nearest', 'wrap'))
    candidate &= peak_vals == localmax
    coords = np.column_stack(np.nonzero(candidate))
    print('Prominent radial seed candidates:', len(coords), flush=True)

    # Circular region mask excludes unreliable regions with no meaningful
    # 3D surface; modest imputation supports small local gaps.
    valid3d = binary_closing(reliable, structure=disk(2))
    valid3d |= reliable
    valid3d = remove_small_objects(valid3d, min_size=MIN_COMPONENT_PIXELS)
    # Do not artificially force the count: each seed must come from 3D relief.
    seeds = np.zeros(relief_z.shape, np.int32)
    seed_count = 0
    for i, k in coords:
        if valid3d[i, k]:
            seed_count += 1
            seeds[i, k] = seed_count
    print('Seeds inside valid surface:', seed_count, flush=True)
    if seed_count == 0:
        raise RuntimeError('No defensible 3D relief peaks: RGB NeRF may lack usable kernel geometry')

    # Periodic theta seam via 3-fold tiled watershed, distinct copies of IDs.
    tiled = np.tile(relief_z, (1, 3))
    mask3 = np.tile(valid3d, (1, 3))
    markers = np.tile(seeds, (1, 3))
    markers[:, :ANGLE_BINS][markers[:, :ANGLE_BINS]>0] += seed_count
    markers[:, 2*ANGLE_BINS:][markers[:, 2*ANGLE_BINS:]>0] += 2*seed_count
    raw = watershed(-tiled, markers, mask=mask3, connectivity=1)
    mid = raw[:, ANGLE_BINS:2*ANGLE_BINS]
    regions = np.where(mid > 0, ((mid-1) % seed_count)+1, 0).astype(np.int32)
    region_size = np.bincount(regions.ravel(), minlength=seed_count+1)
    good = np.where((region_size >= MIN_COMPONENT_PIXELS) &
                    (region_size <= MAX_COMPONENT_PIXELS))[0]
    good = good[good > 0]
    remap = np.zeros(seed_count+1, np.int32)
    remap[good] = np.arange(1, len(good)+1)
    labels = remap[regions]

    # Only assign points located near the estimated outer radial shell,
    # avoiding drawing interior volume samples as kernel segmentation.
    shell_tol = max(float(np.median(rad))*0.025, VOXEL*2)
    r_ref = surface[iz, it]
    is_outer = (rad >= r_ref-shell_tol) & (rad <= r_ref+shell_tol)
    point_labels = np.where(is_outer, labels[iz,it], 0)
    n = len(good)
    print('DIRECT RGB OUTER-SURFACE REGION COUNT:', n, flush=True)
    print('Labeled outer points:', int(np.count_nonzero(point_labels)), '/', len(point_labels), flush=True)

    ply_path = OUTPUT / 'cob1_rgb_outer_surface_instances_v2.ply'
    export_labeled_ply(pts, point_labels, ply_path)
    # Save the outer-surface point cloud with original RGB for independent visual inspection.
    outer_pcd = o3d.geometry.PointCloud()
    outer_pcd.points = o3d.utility.Vector3dVector(pts[is_outer])
    outer_pcd.colors = o3d.utility.Vector3dVector(colors[is_outer])
    original_rgb_outer = OUTPUT / 'cob1_rgb_outer_surface_only_v2.ply'
    o3d.io.write_point_cloud(str(original_rgb_outer), outer_pcd, write_ascii=False)

    fig, axes = plt.subplots(4, 1, figsize=(16, 16), constrained_layout=True)
    radius_scale = np.percentile(surface[reliable], [2,98])
    lims = np.percentile(relief_z[reliable], [2,98])
    axes[0].imshow(surface, origin='lower', aspect='auto', cmap='viridis',
                   vmin=radius_scale[0], vmax=radius_scale[1]);axes[0].set_title('90th-percentile outer radial surface (RGB NeRF only)')
    axes[1].imshow(relief_z, origin='lower', aspect='auto', cmap='RdBu_r',
                   vmin=lims[0], vmax=lims[1]);axes[1].set_title('Local geometric protrusions after taper removal (robust standardized units)')
    axes[2].imshow(relief_z, origin='lower', aspect='auto', cmap='gray',
                   vmin=lims[0], vmax=lims[1]);
    if len(coords):axes[2].scatter(coords[:,1], coords[:,0], s=2, c='red', alpha=.8)
    axes[2].set_title(f'Unsupervised radial maxima: {seed_count} peaks (not adjusted to kernel count)')
    axes[3].imshow(labels, origin='lower', aspect='auto', cmap='nipy_spectral', interpolation='nearest');axes[3].set_title(f'Watershed of 3D-only relief: {n} candidate regions')
    for ax in axes:
        ax.set_xlabel('Angle around cob (periodic)')
        ax.set_ylabel('Position along cob')
    figure_path = OUTPUT / 'cob1_rgb_outer_surface_diagnostics_v2.png'
    fig.savefig(str(figure_path), dpi=160)
    plt.close(fig)

    with (OUTPUT / 'cob1_rgb_outer_surface_v2_summary.csv').open('w',newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['cob','source','method','candidate_count','seed_count',
                    'voxelized_points','reliable_surface_cells','labeled_surface_points',
                    'ground_truth_not_used'])
        w.writerow([COB,str(SOURCE),'RGB NeRF 90pct radial-surface geometric watershed v2',
                    n,seed_count,len(pts),int(reliable.sum()),
                    int(np.count_nonzero(point_labels)),'yes'])
    print('Saved:', figure_path, flush=True)
    print('Saved:', ply_path, flush=True)
    print('Saved:', original_rgb_outer, flush=True)
    print('IMPORTANT: Regions are unvalidated proposals, not proven one-kernel instances.', flush=True)


if __name__ == '__main__':
    run()