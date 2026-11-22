# -*- coding: utf-8 -*-
"""Run the author's ORIGINAL multi-stage kernel clustering on all 63 density clouds.

No re-training; no change in clustering thresholds. Uses a separate Python process
per cob/threshold to release Open3D/NumPy memory. Original PLYs remain untouched.

Usage in Spyder: %run C:/Users/jizh/count_maize_density_sensitivity_windows.py --cob 1 --threshold 70
                 %run C:/Users/jizh/count_maize_density_sensitivity_windows.py --all

The original clustering code is downloaded from the user's public GitHub repo.
"""
import argparse
import csv
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(r"V:\Databaser\LT\exchange\jizh\backup_nas\cob_new_shit")
OUTPUT = ROOT / "density_sensitivity_windows"
RESULTS = OUTPUT / "kernel_count_sensitivity.csv"
SOURCE_URL = (
    "https://raw.githubusercontent.com/"
    "ZJiangsan/Automated-3D-Kernel-Instance-Segmentation-and-Counting/"
    "main/cluster_and_count.py"
)
THRESHOLDS = (40, 50, 60, 70, 80, 90, 100)
GT_KERNELS = (716, 337, 729, 314, 539, 463, 659, 496, 724)
GT_ROWS = (20, 14, 18, 17, 16, 18, 18, 16, 18)
FIELDS = (
    "cob", "threshold", "gt_kernels", "pred_kernels", "error", "abs_error",
    "gt_rows", "pred_rows", "row_abs_error", "seconds", "status", "log"
)


def fetch_author_source():
    dst = OUTPUT / "original_cluster_and_count.py"
    if dst.exists():
        src = dst.read_text(encoding="utf-8")
    else:
        print("Retrieving original author clustering script from GitHub...", flush=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=40) as response:
            src = response.read().decode("utf-8-sig")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src, encoding="utf-8")
    required = [
        "def refine_large_cluster(", "def refine_2nd(",
        "def estimate_kernel_columns_and_save_from_final(",
        "base_dir = 'cob_new'", "kernel_counts = []", "for folder_i in folder_ls:",
        "point_path = os.path.join(folder_path, pcd_file_i)",
        "save_path = os.path.join(base_dir, f\"refined_labeled_18022026_{pcd_file_i}\")",
        "col_count = estimate_kernel_columns_and_save_from_final(out_points, base_dir, folder_i)",
        'print("\\nKernel counts:", kernel_counts)',
    ]
    missing = [needle for needle in required if needle not in src]
    if missing:
        raise RuntimeError("Original source does not match expected version; no experiment run. Missing: " + str(missing))
    return src


def generate_case_script(source, cob, threshold, save_labeled=True):
    # Preserve all the original clustering and row-counting algorithms unchanged.
    prefix, main = source.split("base_dir = 'cob_new'", 1)
    main = "base_dir = 'cob_new'" + main
    first, rest = main.split("kernel_counts = []", 1)
    # Single cob only. All 63 runs use fresh independent Python processes.
    main = (
        "base_dir = " + repr(str(ROOT)) + "\n"
        + f"folder_ls = ['cob{cob}_out']\n"
        + f"THRESHOLD = {threshold}\n"
        + "kernel_counts = []" + rest
    )
    # No summary based on all nine ground truths in a one-cob subprocess.
    main = main.split('print("\\nKernel counts:", kernel_counts)', 1)[0]
    old_point = "point_path = os.path.join(folder_path, pcd_file_i)"
    new_point = (
        "point_path = os.path.join(base_dir, 'density_sensitivity_windows', folder_i, "
        "f'{folder_i}SEG_density_pc_001_threshold_{THRESHOLD:03d}.ply')"
    )
    main = main.replace(old_point, new_point, 1)
    old_save = 'save_path = os.path.join(base_dir, f"refined_labeled_18022026_{pcd_file_i}")'
    if save_labeled:
        new_save = (
            "save_path = os.path.join(base_dir, 'density_sensitivity_windows', 'labeled', "
            "f'{folder_i}_threshold_{THRESHOLD:03d}_labeled.ply')\n"
            "    os.makedirs(os.path.dirname(save_path), exist_ok=True)"
        )
        main = main.replace(old_save, new_save, 1)
    else:
        main = main.replace(old_save, "save_path = os.devnull", 1)
    old_column = "col_count = estimate_kernel_columns_and_save_from_final(out_points, base_dir, folder_i)"
    new_column = (
        "col_count = estimate_kernel_columns_and_save_from_final("
        "out_points, os.path.join(base_dir, 'density_sensitivity_windows', 'column_slices'), "
        "f'{folder_i}_threshold_{THRESHOLD:03d}')"
    )
    main = main.replace(old_column, new_column, 1)
    # Do not open/show a matplotlib GUI in any subprocess.
    prefix = prefix.replace("import matplotlib.pyplot as plt", "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt", 1)
    prefix = prefix.replace("    plt.show()", "    # plt.show() disabled for automated batch")
    main = main.replace("kernel_counts = []", "os.makedirs(os.path.join(base_dir, 'density_sensitivity_windows', 'column_slices'), exist_ok=True)\nkernel_counts = []", 1)
    main += (
        "\nprint('RESULT_KERNEL_COUNT=' + str(kernel_counts[0]), flush=True)\n"
        "print('RESULT_ROW_COUNT=' + str(column_counts[0]), flush=True)\n"
    )
    full = prefix + main
    compile(full, "generated_original_clustering.py", "exec")
    return full


def load_records():
    if not RESULTS.exists():
        return {}
    with RESULTS.open(newline="", encoding="utf-8") as handle:
        return {(int(x["cob"]), int(x["threshold"])): x for x in csv.DictReader(handle)}


def save_records(records):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    temp = RESULTS.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=FIELDS)
        w.writeheader()
        for key in sorted(records):
            w.writerow(records[key])
    temp.replace(RESULTS)


def summary(records):
    good = {k: v for k, v in records.items() if v.get("status") == "OK"}
    path = OUTPUT / "threshold_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "completed_cobs", "kernel_MAE", "kernel_RMSE", "kernel_bias", "row_MAE"])
        for thr in THRESHOLDS:
            rows = [v for (c, t), v in good.items() if t == thr]
            if not rows:
                continue
            errs = [float(v["error"]) for v in rows]
            mae = sum(abs(v) for v in errs) / len(errs)
            rmse = (sum(v * v for v in errs) / len(errs)) ** .5
            bias = sum(errs) / len(errs)
            row_mae = sum(float(v["row_abs_error"]) for v in rows) / len(rows)
            writer.writerow([thr, len(rows), f"{mae:.4f}", f"{rmse:.4f}", f"{bias:.4f}", f"{row_mae:.4f}"])
            print(f"Threshold {thr:3d}: {len(rows):d}/9 cobs, kernel MAE {mae:.2f}, RMSE {rmse:.2f}", flush=True)
    print("Saved:", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all 63 cases, skip completed cases")
    parser.add_argument("--cob", type=int, default=1)
    parser.add_argument("--threshold", type=int, default=70)
    parser.add_argument("--rerun", action="store_true", help="Repeat already-successful cases")
    parser.add_argument("--no-labeled", action="store_true", help="Do not save labeled point clouds")
    args = parser.parse_args()
    source = fetch_author_source()
    records = load_records()
    cases = [(c, t) for c in range(1, 10) for t in THRESHOLDS] if args.all else [(args.cob, args.threshold)]
    if not args.all and (args.cob not in range(1, 10) or args.threshold not in THRESHOLDS):
        parser.error("COB must be 1-9 and threshold must be one of 40,50,60,70,80,90,100")
    for index, (cob, thr) in enumerate(cases, 1):
        if (cob, thr) in records and records[(cob, thr)].get("status") == "OK" and not args.rerun:
            print(f"[{index}/{len(cases)}] cob{cob} threshold {thr}: already complete; skipping")
            continue
        inp = OUTPUT / f"cob{cob}_out" / f"cob{cob}_outSEG_density_pc_001_threshold_{thr:03d}.ply"
        if not inp.exists():
            print("ERROR: missing point cloud:", inp)
            continue
        generated = OUTPUT / "generated_clustering" / f"cluster_cob{cob}_t{thr:03d}.py"
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text(generate_case_script(source, cob, thr, save_labeled=not args.no_labeled), encoding="utf-8")
        log = generated.with_suffix(".log")
        print(f"[{index}/{len(cases)}] Running ORIGINAL clustering on cob{cob}, density {thr}...", flush=True)
        started = time.monotonic()
        with log.open("w", encoding="utf-8", errors="replace") as lf:
            proc = subprocess.run([sys.executable, str(generated)], stdout=lf, stderr=subprocess.STDOUT, check=False)
        seconds = time.monotonic() - started
        txt = log.read_text(encoding="utf-8", errors="replace")
        count = re.findall(r"RESULT_KERNEL_COUNT=(\d+)", txt)
        row_count = re.findall(r"RESULT_ROW_COUNT=(\d+)", txt)
        ok = proc.returncode == 0 and bool(count) and bool(row_count)
        pred = int(count[-1]) if count else ""
        rows = int(row_count[-1]) if row_count else ""
        err = pred - GT_KERNELS[cob - 1] if ok else ""
        record = dict(cob=cob, threshold=thr, gt_kernels=GT_KERNELS[cob - 1], pred_kernels=pred,
                      error=err, abs_error=abs(err) if ok else "", gt_rows=GT_ROWS[cob - 1],
                      pred_rows=rows, row_abs_error=abs(rows - GT_ROWS[cob - 1]) if ok else "",
                      seconds=round(seconds, 1), status="OK" if ok else f"FAILED(code={proc.returncode})", log=str(log))
        records[(cob, thr)] = record
        save_records(records)
        print("Result:", record["status"], "kernel count:", pred, "row count:", rows,
              f"({seconds:.1f}s). Log: {log}", flush=True)
        if not ok:
            print("Last 3000 characters of log:\n", txt[-3000:], flush=True)
            print("STOPPED on failure; no further cobs processed.")
            break
        if cob == 1 and thr == 70 and pred != 707:
            print("NOTE: threshold-70 baseline differs from published COB-1 count (707)."
                  " This may reflect random point sampling; investigate before reporting claims.", flush=True)
    summary(records)
    print("Full results:", RESULTS)

if __name__ == "__main__":
    main()