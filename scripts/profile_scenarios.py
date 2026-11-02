"""Time one run per scenario to measure baseline simulation performance."""

import sys
import time
import json
sys.path.insert(0, ".")

from src.simulation.runner import generate_run
from src.simulation.scenarios import SCENARIO_NAMES

runtimes = {}
print("Timing one run per scenario (11 scenarios)...")

for sc in SCENARIO_NAMES:
    t0 = time.perf_counter()
    res = generate_run(sc, seed=42)
    elapsed = time.perf_counter() - t0
    n_samples = res["metadata"]["n_samples"]
    runtimes[sc] = {
        "elapsed_s": round(elapsed, 3),
        "n_samples": n_samples,
        "speed_samples_per_s": round(n_samples / max(1e-6, elapsed), 1),
    }
    print(f"Scenario '{sc:24s}': {elapsed:6.3f} s ({n_samples:4d} samples, {runtimes[sc]['speed_samples_per_s']:6.1f} smp/s)")

with open("data_profile_times.json", "w") as f:
    json.dump(runtimes, f, indent=2)

total_single_threaded = sum(d["elapsed_s"] for d in runtimes.values())
print(f"\nTotal time for 1 run of all 11 scenarios: {total_single_threaded:.2f} s")

