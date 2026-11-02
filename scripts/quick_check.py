import sys
sys.path.insert(0, ".")
from src.simulation.runner import generate_run
from src.simulation.scenarios import SCENARIO_NAMES

print("Testing generate_run on all scenarios with seed 42...")
for sc in SCENARIO_NAMES:
    res = generate_run(sc, seed=42)
    meta = res["metadata"]
    labels = res["labels"]
    import numpy as np
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique_labels, counts))
    runaway_t = meta["t_runaway_onset"]
    c4_len = meta["class_4_duration_s"]
    max_tc = meta["max_t_core"]
    print(f"{sc:24s} | N={meta['n_samples']:4d} | runaway={str(runaway_t):8s} | C4_len={c4_len:6.1f}s | maxTc={max_tc:6.1f}C | labels={label_dist}")

