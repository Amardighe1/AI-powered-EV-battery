"""Dataset loader, windowing engine, and leakage verification.

Extracts:
 - Tabular features (34-dim) via compute_features(window[60,14], soc=soc_t)
 - Raw windows (60, 14) for 1D-CNN
 - Window labels at the last sample (t=59)
 - Stride: 5s for Train/Val, 1s for Test
 - Fits StandardScaler strictly on Train, applies to Val and Test
 - Performs strict dataset leakage checks
"""

import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

from .features import compute_features, FEATURE_NAMES


def extract_windows_from_run(file_path: str, stride: int = 5):
    """Extract sliding windows of length 60 from a single run file.
    
    Returns:
        feats_list: list of 34-dim feature vectors
        raw_windows_list: list of (60, 14) sensor arrays
        labels_list: list of integer labels (0..5)
        times_list: list of float timestamps at window end
        t_to_runaway_list: list of float time-to-runaway values
    """
    data = np.load(file_path)
    sensors = data["sensors"]      # (N, 14)
    labels = data["labels"]        # (N,)
    soc_series = data["soc_series"] # (N,)
    times = data["time"]           # (N,)
    t_runaway = data["time_to_runaway_s"] # (N,)

    n_samples = len(sensors)
    feats = []
    raw_windows = []
    win_labels = []
    win_times = []
    win_t_runaway = []

    # Windows of 60 samples: first window ends at t=59
    for end_idx in range(59, n_samples, stride):
        start_idx = end_idx - 59
        window = sensors[start_idx : end_idx + 1]  # shape (60, 14)
        soc_val = soc_series[end_idx]

        f = compute_features(window, soc=soc_val)
        feats.append(f)
        raw_windows.append(window.astype(np.float32))
        win_labels.append(labels[end_idx])
        win_times.append(times[end_idx])
        win_t_runaway.append(t_runaway[end_idx])

    return (
        np.array(feats, dtype=np.float32),
        np.array(raw_windows, dtype=np.float32),
        np.array(win_labels, dtype=np.int64),
        np.array(win_times, dtype=np.float32),
        np.array(win_t_runaway, dtype=np.float32),
    )


def build_and_cache_dataset(
    manifest_path: str = "data/manifest.json",
    cache_path: str = "data/dataset_cache.npz",
    artifacts_dir: str = "artifacts",
):
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Leakage check: verify disjoint seeds and runs across splits
    train_runs = [m for m in manifest if m["split"] == "train"]
    val_runs = [m for m in manifest if m["split"] == "val"]
    test_runs = [m for m in manifest if m["split"] == "test"]

    train_seeds = set(m["seed"] for m in train_runs)
    val_seeds = set(m["seed"] for m in val_runs)
    test_seeds = set(m["seed"] for m in test_runs)

    assert len(train_seeds & val_seeds) == 0, "DATA LEAKAGE: Overlapping seeds between Train and Val!"
    assert len(train_seeds & test_seeds) == 0, "DATA LEAKAGE: Overlapping seeds between Train and Test!"
    assert len(val_seeds & test_seeds) == 0, "DATA LEAKAGE: Overlapping seeds between Val and Test!"
    assert len(train_runs) + len(val_runs) + len(test_runs) == 390

    print(f"Leakage Check PASSED: 0 seed overlap between {len(train_runs)} Train, {len(val_runs)} Val, {len(test_runs)} Test runs.")

    # Process Train (stride = 5)
    print("Extracting Train windows (stride=5s)...")
    X_train_list, raw_train_list, y_train_list = [], [], []
    for m in train_runs:
        feats, raw, labs, _, _ = extract_windows_from_run(m["file_path"], stride=5)
        X_train_list.append(feats)
        raw_train_list.append(raw)
        y_train_list.append(labs)
    X_train = np.vstack(X_train_list)
    raw_train = np.vstack(raw_train_list)
    y_train = np.concatenate(y_train_list)

    # Process Val (stride = 5)
    print("Extracting Val windows (stride=5s)...")
    X_val_list, raw_val_list, y_val_list = [], [], []
    for m in val_runs:
        feats, raw, labs, _, _ = extract_windows_from_run(m["file_path"], stride=5)
        X_val_list.append(feats)
        raw_val_list.append(raw)
        y_val_list.append(labs)
    X_val = np.vstack(X_val_list)
    raw_val = np.vstack(raw_val_list)
    y_val = np.concatenate(y_val_list)

    # Process Test (stride = 1)
    print("Extracting Test windows (stride=1s) for full-resolution alarm evaluation...")
    X_test_list, raw_test_list, y_test_list = [], [], []
    test_run_metadata = []
    for m in test_runs:
        feats, raw, labs, times, t_runaway = extract_windows_from_run(m["file_path"], stride=1)
        X_test_list.append(feats)
        raw_test_list.append(raw)
        y_test_list.append(labs)
        test_run_metadata.append({
            "run_id": m["run_id"],
            "scenario": m["scenario"],
            "seed": m["seed"],
            "n_windows": len(labs),
            "times": times.tolist(),
            "labels": labs.tolist(),
            "time_to_runaway_s": t_runaway.tolist(),
            "t_runaway_onset": m["t_runaway_onset"],
        })
    X_test = np.vstack(X_test_list)
    raw_test = np.vstack(raw_test_list)
    y_test = np.concatenate(y_test_list)

    print(f"Windows extracted: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

    # Class counts
    class_counts = {}
    for split_name, y_arr in [("train", y_train), ("val", y_val), ("test", y_test)]:
        u, c = np.unique(y_arr, return_counts=True)
        counts = {int(k): int(v) for k, v in zip(u, c)}
        for cls in range(6):
            counts.setdefault(cls, 0)
        class_counts[split_name] = counts

    # Standardize tabular features strictly using Train statistics
    print("Fitting StandardScaler on X_train only...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Standardize raw sensor channels for 1D-CNN using Train channel mean and std
    print("Computing channel-wise normalization parameters for 1D-CNN from Train...")
    channel_mean = np.mean(raw_train, axis=(0, 1), keepdims=True) # (1, 1, 14)
    channel_std = np.std(raw_train, axis=(0, 1), keepdims=True) + 1e-6
    raw_train_norm = (raw_train - channel_mean) / channel_std
    raw_val_norm = (raw_val - channel_mean) / channel_std
    raw_test_norm = (raw_test - channel_mean) / channel_std

    # Save scaler and channel stats to artifacts
    joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.joblib"))
    with open(os.path.join(artifacts_dir, "feature_names.json"), "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    np.savez_compressed(
        cache_path,
        X_train_scaled=X_train_scaled,
        X_val_scaled=X_val_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        raw_train_norm=raw_train_norm,
        raw_val_norm=raw_val_norm,
        raw_test_norm=raw_test_norm,
        channel_mean=channel_mean,
        channel_std=channel_std,
    )

    with open("data/test_run_metadata.json", "w") as f:
        json.dump(test_run_metadata, f)

    return {
        "class_counts": class_counts,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
    }


if __name__ == "__main__":
    res = build_and_cache_dataset()
    print("\nDataset building complete. Class counts per split:")
    print(json.dumps(res["class_counts"], indent=2))

