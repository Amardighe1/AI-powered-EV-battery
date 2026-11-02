import sys
sys.path.insert(0, ".")
from src.simulation.runner import generate_run

# 1. Inspect normal drive
res_norm = generate_run("normal_drive", seed=42)
vt = res_norm["true_states"]["v_terminal"]
imb = np_max = vt.max(axis=1) - vt.min(axis=1)
print(f"Normal drive max imbalance: {imb.max()*1000:.1f} mV, mean: {imb.mean()*1000:.1f} mV")

# 2. Inspect overcharge runaway
res_ovc = generate_run("overcharge_runaway", seed=42, duration_s=3000)
meta_ovc = res_ovc["metadata"]
print(f"Overcharge runaway at 3000s: runaway={meta_ovc['t_runaway_onset']}, maxTc={meta_ovc['max_t_core']:.1f}C")

