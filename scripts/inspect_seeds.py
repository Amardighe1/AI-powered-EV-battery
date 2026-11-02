import sys
sys.path.insert(0, ".")
from src.simulation.runner import generate_run
import numpy as np

for s in range(20):
    res = generate_run("internal_short_runaway", seed=s)
    m = res["metadata"]
    if m["t_runaway_onset"] is None:
        soc_end = res["true_states"]["soc"][-1]
        v_end = res["true_states"]["v_terminal"][-1]
        t_core_max = res["true_states"]["t_core"].max(axis=0)
        print(f"Seed {s:2d}: maxTc={m['max_t_core']:.1f}C, Group max Tcs={t_core_max}, SOC={soc_end}")

