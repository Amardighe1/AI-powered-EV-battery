import sys
sys.path.insert(0, ".")
from src.simulation.runner import generate_run
import numpy as np

for sc in ["overcharge_runaway", "internal_short_runaway"]:
    windows = []
    runaways = 0
    onsets = []
    for s in range(20):
        res = generate_run(sc, seed=s)
        m = res["metadata"]
        if m["t_runaway_onset"] is not None:
            runaways += 1
            windows.append(m["class_4_duration_s"])
            onsets.append(m["t_runaway_onset"])
        else:
            print(f"[{sc}] Seed {s}: NO RUNAWAY, maxTc={m['max_t_core']:.1f}C")
            
    print(f"Scenario '{sc}': {runaways}/20 runaway!")
    print(f"  Class 4 Window: min={min(windows):.1f}s, max={max(windows):.1f}s, median={np.median(windows):.1f}s")
    print(f"  Onset time: min={min(onsets):.1f}s, max={max(onsets):.1f}s, median={np.median(onsets):.1f}s")

