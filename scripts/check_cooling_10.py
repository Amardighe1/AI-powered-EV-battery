import sys
sys.path.insert(0, ".")
from src.simulation.runner import generate_run
import numpy as np

for s in range(10):
    res = generate_run("cooling_failure", seed=s)
    labels = res["labels"]
    u, c = np.unique(labels, return_counts=True)
    m = res["metadata"]
    print(f"Seed {s:2d}: maxTc={m['max_t_core']:.1f}C, starts_c0={labels[0]==0}, has_c3={3 in labels}, labels={dict(zip(u, c))}")

