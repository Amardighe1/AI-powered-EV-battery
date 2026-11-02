import sys
sys.path.insert(0, ".")
from src.simulation.runner import generate_run
import numpy as np

res = generate_run("internal_short_runaway", seed=42)
t = res["time"]
labels = res["labels"]
co2 = res["true_states"]["co2"]
co = res["true_states"]["co"]
smoke = res["true_states"]["smoke"]
tc = res["true_states"]["t_core"]
sh = res["true_states"]["self_heating_rate"]
t_runaway = res["metadata"]["t_runaway_onset"]

# Find index just before runaway
runaway_idx = int(np.where(t == t_runaway)[0][0])
print(f"Runaway onset at t={t_runaway}s (idx={runaway_idx})")
print(f"At t = {t_runaway - 60:.0f}s: CO2={co2[runaway_idx-60]:.1f} ppm, CO={co[runaway_idx-60]:.1f} ppm, maxTc={tc[runaway_idx-60].max():.1f}C")
print(f"At t = {t_runaway - 10:.0f}s: CO2={co2[runaway_idx-10]:.1f} ppm, CO={co[runaway_idx-10]:.1f} ppm, maxTc={tc[runaway_idx-10].max():.1f}C")
print(f"At t = {t_runaway}: CO2={co2[runaway_idx]:.1f} ppm, CO={co[runaway_idx]:.1f} ppm, maxTc={tc[runaway_idx].max():.1f}C")
print(f"Post-runaway max: CO2={co2.max():.1f} ppm, CO={co.max():.1f} ppm, Smoke={smoke.max():.2f}")

