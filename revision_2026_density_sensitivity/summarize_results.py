#!/usr/bin/env python3
"""Recalculate nine-cob sensitivity statistics from archived case-level results.

Standard-library only. Fails if any of the expected 63 cases are absent or failed.
"""
import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DETAIL = HERE / "results" / "kernel_count_sensitivity.csv"
SUMMARY = HERE / "results" / "threshold_metrics.csv"
THRESHOLDS = (40, 50, 60, 70, 80, 90, 100)
COBS = range(1, 10)


def load_detail(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))
    cases = {}
    for row in records:
        key = int(row["cob"]), int(row["threshold"])
        if key in cases:
            raise ValueError(f"Duplicate case {key}")
        if row["status"] != "OK":
            raise ValueError(f"Unsuccessful case {key}: {row['status']}")
        if int(row["pred_kernels"]) - int(row["gt_kernels"]) != int(row["error"]):
            raise ValueError(f"Kernel error mismatch in {key}")
        if abs(int(row["error"])) != int(row["abs_error"]):
            raise ValueError(f"Absolute error mismatch in {key}")
        if abs(int(row["pred_rows"]) - int(row["gt_rows"])) != int(row["row_abs_error"]):
            raise ValueError(f"Row error mismatch in {key}")
        cases[key] = row
    expected = {(c, t) for c in COBS for t in THRESHOLDS}
    if set(cases) != expected:
        raise ValueError(f"Expected {len(expected)} cases, found {len(cases)}; missing: {expected-set(cases)}")
    return cases


def main():
    cases = load_detail(DETAIL)
    with SUMMARY.open(newline="", encoding="utf-8-sig") as f:
        reported = {int(x["threshold"]): x for x in csv.DictReader(f)}
    print("Threshold  Cobs  Kernel_MAE  Kernel_RMSE  Bias  Row_MAE")
    for t in THRESHOLDS:
        rows = [cases[c, t] for c in COBS]
        errs = [int(row["error"]) for row in rows]
        rerr = [int(row["row_abs_error"]) for row in rows]
        calculated = {
            "kernel_MAE": sum(abs(x) for x in errs) / 9,
            "kernel_RMSE": math.sqrt(sum(x*x for x in errs) / 9),
            "kernel_bias": sum(errs) / 9,
            "row_MAE": sum(rerr) / 9,
        }
        for key, value in calculated.items():
            if abs(float(reported[t][key]) - value) > 0.00011:
                raise AssertionError(f"Threshold {t}: {key} CSV mismatch")
        if int(reported[t]["completed_cobs"]) != 9:
            raise AssertionError(f"Threshold {t}: not nine cobs")
        print(f"{t:9d}  9  {calculated['kernel_MAE']:10.4f}  "
              f"{calculated['kernel_RMSE']:11.4f}  "
              f"{calculated['kernel_bias']:5.4f}  {calculated['row_MAE']:7.4f}")
    print("PASS: 63 cases and 7 threshold summaries validated.")


if __name__ == "__main__":
    main()
