"""Run existing verified COB-1 Windows NeRF exporter for COBs 2..9.

Open/run this in the same Anaconda nerfstudio environment. No changes are
made to original checkpoint files or historical point clouds.

Uses separate Python processes per cob to release CUDA/CPU allocations.
"""
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(r"C:\Users\jizh\export_cob1_density_windows.py")
OUT = Path(r"C:\Users\jizh\maize_density_generated")
BASE = Path(r"V:\Databaser\LT\exchange\jizh\backup_nas\cob_new_shit")
COBS = range(2, 10)  # COB-1 already completed


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Verified COB-1 exporter not found: {SOURCE}")
    source = SOURCE.read_text(encoding="utf-8")
    needle = 'folder_i = "cob1_out"'
    if source.count(needle) != 1:
        raise RuntimeError("COB-1 script differs from tested version; no changes made")
    if "CHECKPOINT LOADED SUCCESSFULLY" not in source:
        raise RuntimeError("Expected tested exporter checkpoint check not found")
    OUT.mkdir(exist_ok=True, parents=True)
    for i in COBS:
        cob = f"cob{i}_out"
        checkpoint = BASE / f"best_checkpoint_nerfacto_cobNew{cob}Seg_RGB_pc_001.pth"
        if not (BASE / cob).is_dir() or not checkpoint.is_file():
            raise FileNotFoundError(f"Missing source data/checkpoint for {cob}")
        generated = OUT / f"export_{cob}_density.py"
        generated.write_text(source.replace(needle, f'folder_i = "{cob}"', 1), encoding="utf-8")
        log = OUT / f"export_{cob}.log"
        print(f"\n===== {cob}: starting, log={log} =====", flush=True)
        start = time.perf_counter()
        with log.open("w", encoding="utf-8", errors="replace") as f:
            result = subprocess.run([sys.executable, "-u", str(generated)], cwd=str(OUT), stdout=f, stderr=subprocess.STDOUT)
        print(f"{cob} exit code: {result.returncode}; elapsed {time.perf_counter()-start:.1f}s", flush=True)
        if result.returncode != 0:
            print(f"ERROR: stopped at {cob}. Last log lines:", flush=True)
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            print("\n".join(lines[-50:]), flush=True)
            sys.exit(result.returncode)
        expected = BASE / "density_sensitivity_windows" / cob / "point_counts.csv"
        if not expected.is_file():
            raise RuntimeError(f"Exporter exited successfully but results missing: {expected}")
        print(f"Saved 7 thresholds: {expected}", flush=True)
    print("\nALL REMAINING COBS EXPORTED. Original PLYs unchanged.")


if __name__ == "__main__":
    main()