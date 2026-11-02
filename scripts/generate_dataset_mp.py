"""Multiprocessing dataset generator for Phase 2.

Generates 390 runs:
 - 60 runs for each runaway scenario (overcharge_runaway, internal_short_runaway) -> 120 runs
 - 30 runs for each of the 9 other scenarios -> 270 runs
Disjoint seeds: seed = 10000 + sc_idx * 100 + run_idx
Stratified 60/20/20 split by RUN per scenario:
 - 60 runs: 36 train, 12 val, 12 test
 - 30 runs: 18 train, 6 val, 6 test
Total: 234 Train, 78 Val, 78 Test.
"""

import os
import sys
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

sys.path.insert(0, ".")
from src.simulation.runner import generate_run
from src.simulation.scenarios import SCENARIO_NAMES
from src.ml.features import compute_run_soc_series

RUNAWAY_SCENARIOS = {"overcharge_runaway", "internal_short_runaway"}


def generate_single_run_worker(task):
    run_id, sc, seed, split, out_dir = task
    res = generate_run(sc, seed=seed)
    sensors = res["sensors"]
    labels = res["labels"]
    t_arr = res["time"]
    t_runaway_s = res["time_to_runaway_s"]
    meta = res["metadata"]

    # Precompute sensor-only Coulomb counted SOC series for fast windowing
    soc_series = compute_run_soc_series(sensors)

    # Save run file
    file_name = f"run_{run_id:03d}_{sc}_{split}.npz"
    file_path = os.path.join(out_dir, file_name)

    np.savez_compressed(
        file_path,
        time=t_arr,
        sensors=sensors,
        labels=labels,
        time_to_runaway_s=t_runaway_s,
        soc_series=soc_series,
    )

    info = {
        "run_id": run_id,
        "scenario": sc,
        "seed": seed,
        "split": split,
        "file_name": file_name,
        "file_path": file_path,
        "duration_s": float(t_arr[-1]),
        "n_samples": int(len(t_arr)),
        "t_runaway_onset": meta["t_runaway_onset"],
        "class_4_duration_s": meta["class_4_duration_s"],
        "max_t_core": meta["max_t_core"],
    }
    return info


def build_dataset_tasks(out_dir="data/raw_runs"):
    os.makedirs(out_dir, exist_ok=True)
    tasks = []
    run_id = 0

    for sc_idx, sc in enumerate(SCENARIO_NAMES):
        n_runs = 60 if sc in RUNAWAY_SCENARIOS else 30
        n_train = int(n_runs * 0.60)  # 36 or 18
        n_val = int(n_runs * 0.20)    # 12 or 6
        n_test = n_runs - n_train - n_val # 12 or 6

        for j in range(n_runs):
            seed = 10000 + sc_idx * 100 + j
            if j < n_train:
                split = "train"
            elif j < n_train + n_val:
                split = "val"
            else:
                split = "test"

            tasks.append((run_id, sc, seed, split, out_dir))
            run_id += 1

    return tasks


def main():
    tasks = build_dataset_tasks()
    print(f"Total tasks prepared: {len(tasks)} runs (234 Train, 78 Val, 78 Test)")
    
    n_workers = min(os.cpu_count() or 4, 12)
    print(f"Launching generation using ProcessPoolExecutor with {n_workers} worker processes...")
    t0 = time.perf_counter()

    manifest = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(generate_single_run_worker, task) for task in tasks]
        for idx, fut in enumerate(as_completed(futures), 1):
            info = fut.result()
            manifest.append(info)
            if idx % 50 == 0 or idx == len(tasks):
                elapsed = time.perf_counter() - t0
                print(f"  [{idx:3d}/{len(tasks):3d}] runs generated ({elapsed:.1f} s elapsed, {idx/elapsed:.1f} runs/s)")

    elapsed = time.perf_counter() - t0
    print(f"\nAll {len(tasks)} runs successfully generated in {elapsed:.2f} seconds!")

    # Sort manifest by run_id
    manifest.sort(key=lambda x: x["run_id"])
    with open("data/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("Manifest saved to data/manifest.json.")


if __name__ == "__main__":
    main()

