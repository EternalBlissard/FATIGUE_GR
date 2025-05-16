from __future__ import annotations
import os
import math
import concurrent.futures
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import signal

# Important variables
SAMPLING_RATE: int = 1259  # Hz
WINDOW_SECONDS: int = 20
WINDOW_SAMPLES: int = WINDOW_SECONDS * SAMPLING_RATE  # 25 180

# Paths
FILTERED_DIR: str = "path/to/filtered/data"
OUT_ROOT: str = "path/to/windowed/EMG/features"

# Number of workers for parallel processing
N_WORKERS: int = 16

# Participants and gestures to process
PARTICIPANTS: List[int] = [p for p in range(1, 45) if p not in {10, 39, 40}]
GESTURES: List[int] = list(range(1, 6))

def _extract_features(x: np.ndarray, fs: int = SAMPLING_RATE) -> Dict[str, float]:
    """Return EMG features for a 1-D window."""
    feats: Dict[str, float] = {
        "RMS": float(np.sqrt(np.mean(x ** 2, dtype=np.float64))),
        "MIN": float(np.min(x)),
        "MAX": float(np.max(x)),
        "STD": float(np.std(x, ddof=0)),
        "MAV": float(np.mean(np.abs(x))),
        "VAR": float(np.var(x, ddof=0)),
    }

    nperseg = min(1024, len(x))  # Welch
    freqs, pxx = signal.welch(x, fs=fs, nperseg=nperseg)
    feats.update(
        {
            "PSD_MIN": float(pxx.min()),
            "PSD_MAX": float(pxx.max()),
            "PSD_STD": float(pxx.std()),
            "PSD_MEAN": float(pxx.mean()),
        }
    )

    total = pxx.sum()
    if total > 0:
        cumsum = np.cumsum(pxx)
        feats["MDF"] = float(np.interp(0.5 * total, cumsum, freqs))
    else:
        feats["MDF"] = float("nan")

    return feats


def _process_file(participant: int, gesture: int) -> None:
    in_path = os.path.join(
        FILTERED_DIR, f"P{participant}", f"Gesture_{gesture}_EMG_filtered.csv"
    )
    if not os.path.exists(in_path):
        print(f"[skip] {in_path}")
        return

    df = pd.read_csv(in_path)

    # Identify EMG channels (numeric, not time)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skip_cols = [c for c in df.columns if "time" in c.lower()]
    channels = [c for c in numeric_cols if c not in skip_cols]

    num_windows = math.floor(len(df) / WINDOW_SAMPLES)
    if num_windows == 0:
        print(f"[warn] File shorter than one 20 s window: {in_path}")
        return

    rows: List[Dict[str, float]] = []
    for w in range(num_windows):
        start = w * WINDOW_SAMPLES
        end = start + WINDOW_SAMPLES
        row: Dict[str, float] = {"Interval": w}
        for ch in channels:
            feats = _extract_features(df[ch].to_numpy(dtype=np.float64)[start:end])
            for name, val in feats.items():
                row[f"{ch}_{name}"] = val
        rows.append(row)

    out_dir = os.path.join(OUT_ROOT, f"P{participant}")
    
    
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"Gesture_{gesture}_emg_features_20s.csv")
    out_df = pd.DataFrame(rows)
    # Sort the columns in alphabetical order for better readability
    out_df = out_df.reindex(sorted(out_df.columns), axis=1)
    out_df.to_csv(out_csv, index=False)
    print(f"Successfully processed Participant {participant} Gesture {gesture} → found {num_windows} windows → saved to {out_csv}")

# Wrapper for multiprocessing
def _proc_star(args: Tuple[int, int]):
    return _process_file(*args)

def main():
    tasks = [(p, g) for p in PARTICIPANTS for g in GESTURES]
    os.makedirs(OUT_ROOT, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
        list(exe.map(_proc_star, tasks))


if __name__ == "__main__":
    main()
