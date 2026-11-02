import sys
sys.path.insert(0, ".")
from src.simulation.runner import generate_run
import numpy as np

for s in range(5):
    res = generate_run("cooling_failure", seed=s)
    labels = res["labels"]
    u, c = np.unique(labels, return_counts=True)
    m = res["metadata"]
    print(f"Seed {s}: maxTc={m['max_t_core']:.1f}C, labels={dict(zip(u, c))}")

