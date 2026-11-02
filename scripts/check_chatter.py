import sys
sys.path.insert(0, ".")
from src.simulation.runner import generate_run
import numpy as np

res = generate_run("overcharge_runaway", seed=42)
t = res["time"]
v = res["true_states"]["v_terminal"]
dv = np.diff(v, axis=0)

# Check max frame-to-frame jump
print("Max frame-to-frame |dV| in V1..4:", np.max(np.abs(dv), axis=0))
print("Overcharge runaway duration:", len(t), "s")
print("Runaway onset:", res["metadata"]["t_runaway_onset"], "s")
print("Class 4 window:", res["metadata"]["class_4_duration_s"], "s")

