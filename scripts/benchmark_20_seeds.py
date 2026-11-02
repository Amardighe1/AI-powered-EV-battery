"""Benchmark script evaluating 20 random seeds across all 11 scenarios.

Outputs comprehensive markdown summary table.
"""

import os
import sys
import numpy as np

sys.path.insert(0, ".")
from src.simulation.runner import generate_run
from src.simulation.scenarios import SCENARIO_NAMES


def run_benchmark(n_seeds: int = 20):
    results = {}
    
    print(f"Running benchmark across {len(SCENARIO_NAMES)} scenarios x {n_seeds} seeds = {len(SCENARIO_NAMES)*n_seeds} simulations...")
    
    for sc in SCENARIO_NAMES:
        print(f"Processing scenario: {sc} ...")
        sc_data = {
            "durations": [],
            "label_counts": {i: [] for i in range(6)},
            "runaway_onsets": [],
            "c4_windows": [],
            "max_temps": [],
        }
        
        for seed in range(n_seeds):
            res = generate_run(sc, seed=seed)
            labels = res["labels"]
            meta = res["metadata"]

            sc_data["durations"].append(meta["duration_s"])
            sc_data["max_temps"].append(meta["max_t_core"])
            
            # Count labels
            for c in range(6):
                sc_data["label_counts"][c].append(int(np.sum(labels == c)))

            if meta["t_runaway_onset"] is not None:
                sc_data["runaway_onsets"].append(meta["t_runaway_onset"])
                sc_data["c4_windows"].append(meta["class_4_duration_s"])

        results[sc] = sc_data

    # Generate Markdown Table
    lines = []
    lines.append("# Benchmark Report: 20 Seeds per Scenario (220 Total Runs)")
    lines.append("")
    lines.append("| Scenario | Mean Dur (s) | Label 0 | Label 1 | Label 2 | Label 3 | Label 4 | Label 5 | Runaway Rate | Runaway Onset (min/med/max) | Class 4 Window (min/med/max) | Max Core Temp (°C) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

    for sc in SCENARIO_NAMES:
        d = results[sc]
        mean_dur = f"{np.mean(d['durations']):.0f}"
        
        # Mean label counts
        l_counts = [f"{np.mean(d['label_counts'][c]):.0f}" for c in range(6)]
        
        n_runaway = len(d["runaway_onsets"])
        rate_str = f"{n_runaway}/{n_seeds} ({n_runaway/n_seeds*100:.0f}%)"
        
        if n_runaway > 0:
            onset_str = f"{min(d['runaway_onsets']):.0f} / {np.median(d['runaway_onsets']):.0f} / {max(d['runaway_onsets']):.0f} s"
            c4_str = f"{min(d['c4_windows']):.0f} / {np.median(d['c4_windows']):.0f} / {max(d['c4_windows']):.0f} s"
        else:
            onset_str = "N/A"
            c4_str = "N/A"
            
        t_max_str = f"{min(d['max_temps']):.1f} / {np.median(d['max_temps']):.1f} / {max(d['max_temps']):.1f} °C"

        row = f"| **{sc}** | {mean_dur} | {l_counts[0]} | {l_counts[1]} | {l_counts[2]} | {l_counts[3]} | {l_counts[4]} | {l_counts[5]} | {rate_str} | {onset_str} | {c4_str} | {t_max_str} |"
        lines.append(row)

    md_content = "\n".join(lines)
    with open("BENCHMARK_20_SEEDS.md", "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print("\nBenchmark completed. Summary:\n")
    print(md_content)
    return md_content


if __name__ == "__main__":
    run_benchmark(n_seeds=20)

