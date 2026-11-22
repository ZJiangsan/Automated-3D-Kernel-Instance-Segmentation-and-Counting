# Reviewer revision: NeRF density-threshold sensitivity (November 2026)

This folder archives the **additional 63-case sensitivity analysis** supporting the revised maize-kernel-counting manuscript. It does **not** replace the original published-method source files in the repository root or the original reported MAE.

## What was evaluated

Nine maize cobs (COB-1 through COB-9) were re-exported from their segmentation-guided NeRF checkpoints at density thresholds **40, 50, 60, 70, 80, 90 and 100**: 9 × 7 = **63** point clouds. For each cob, the seven cutoffs use the **same newly sampled candidate-point pool**. The original downstream filtering, multi-stage DBSCAN, brightness-guided refinement and row-count procedure was run with unchanged parameters on every point cloud.

The `results/` directory contains the actual 63 cases and threshold-level statistics. The source inputs are the original trained checkpoints and dataset folders, which are **not included** here. Large intermediate PLYs are likewise not included. The original baseline PLYs and counting script remain unchanged.

| Threshold | Kernel MAE | Kernel RMSE | Kernel signed bias (pred − GT) | Row MAE |
|---|---:|---:|---:|---:|
| 40 | 11.56 | 12.48 | +4.67 | 0.44 |
| 50 | 10.89 | 12.00 | +3.78 | 0.44 |
| 60 | 11.11 | 12.30 | +4.00 | 0.44 |
| 70 | 11.89 | 12.61 | +5.44 | 0.56 |
| 80 | 13.44 | 14.31 | +4.56 | 0.56 |
| 90 | 14.11 | 14.86 | +4.33 | 0.56 |
| 100 | 14.22 | 15.33 | +4.22 | 0.33 |

**Important distinction:** The submitted manuscript's original mean absolute kernel-counting error is **10.33 kernels**, repeatedly reproduced by running the **original counting code on the original saved point-cloud PLY files**. The **11.89** figure at cutoff 70 belongs to these **separately regenerated point clouds**. These are different inputs, not evidence that the original counting script fails to reproduce its baseline. The cause of the re-export differences was not isolated; do not ascribe them definitively to random ray sampling.

Threshold 50 gives the lowest MAE on the new samples. Threshold 70 was the previously chosen original operating threshold and is **not** presented as retrospectively optimal.

## Scripts (Windows, Nerfstudio 1.1.5)

1. `prepare_windows_maize_export.py`: takes the repository-root `export_pointcloud.py`, generates the COB-1 checkpoint-based export driver `export_cob1_density_windows.py` with a proper `__main__` guard. The generation script intentionally checks the expected original source structure; it fails rather than silently adapting to a changed root script.
2. `run_remaining_maize_cobs_windows.py`: reuses that COB-1 export driver for COBs 2–9 in separate Python processes.
3. `count_maize_density_sensitivity_windows.py`: retrieves the repository-root `cluster_and_count.py` and runs the **original algorithms** against each threshold PLY in a separate subprocess. It writes per-run logs, a case-level CSV and threshold aggregate statistics. The scripts have the original workstation paths as defaults: edit `BASE`, `ROOT`, `SOURCE` or `OUT` for a different installation. This experiment was run in an environment with `nerfstudio`, `torch`, `tinycudann`, `open3d`, `scikit-learn`, `scikit-image`, `scipy`, `numpy` and `matplotlib`.
4. `summarize_results.py`: pure-standard-library recomputation/verification using the archived CSV. It requires **no Nerfstudio or GPU**.

### Re-run outline

Use the **Windows Anaconda Prompt** with the working Nerfstudio environment (do not launch the export driver as Spyder `%run` when Python multiprocessing cannot import `__main__`). Example, from a checkout of this repository:

```powershell
python revision_2026_density_sensitivity/prepare_windows_maize_export.py --source export_pointcloud.py --output C:\Users\jizh\export_cob1_density_windows.py
python C:\Users\jizh\export_cob1_density_windows.py
python revision_2026_density_sensitivity/run_remaining_maize_cobs_windows.py
python revision_2026_density_sensitivity/count_maize_density_sensitivity_windows.py --all
python revision_2026_density_sensitivity/summarize_results.py
```

The export is GPU-dependent; the analysis script can be run separately once all threshold PLYs exist. `count_maize_density_sensitivity_windows.py --all` skips successful cases already in its local result CSV. It does not overwrite original baseline point clouds.

### Verification / interpretive limits

`results/kernel_count_sensitivity.csv` contains **63 completed cases** and the recorded per-case wall time; its `log` column is archived as relative filenames rather than exposing a research workstation's private storage paths. The actual detailed logs and generated PLYs are not archived. `results/threshold_metrics.csv` holds the rounded threshold summaries; `summarize_results.py` checks consistency with the case-level values.

These tests investigate density-cutoff sensitivity **within nine evaluated cobs and one set of trained checkpoints**, keeping postprocessing unchanged. They do not establish genotype transfer, cross-environment generalization, repeated export variance, or superiority to 3D Gaussian Splatting.

## Separate exploratory RGB-only attempt

The optional script in `exploratory_rgb_only/` is a **non-validated COB-1-only** geometric instance proposal experiment. It does **not** appear in the quantitative density-sensitivity table and should not be treated as a validated competing counting baseline.
