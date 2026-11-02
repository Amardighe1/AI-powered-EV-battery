import sys
sys.path.insert(0, ".")
from src.simulation.runner import generate_run
import src.simulation.scenarios as sc_mod
import numpy as np

for r_low, r_high in [(0.12, 0.22), (0.15, 0.25)]:
    orig_get = sc_mod.get_scenario_config
    def make_patched(low, high):
        def patched_get(scenario, s, duration_s=1800):
            cfg = orig_get(scenario, s, duration_s)
            if scenario == "internal_short_runaway":
                cfg["r_short_profiles"][:] = np.inf
                val = cfg["rng"].uniform(low, high)
                short_start = cfg["rng"].randint(50, 100)
                cfg["r_short_profiles"][short_start:, 1] = val
            return cfg
        return patched_get

    sc_mod.get_scenario_config = make_patched(r_low, r_high)

    windows = []
    runaways = 0
    for s in range(20):
        res = generate_run("internal_short_runaway", seed=s)
        m = res["metadata"]
        if m["t_runaway_onset"] is not None:
            runaways += 1
            windows.append(m["class_4_duration_s"])

    sc_mod.get_scenario_config = orig_get
    print(f"R_short [{r_low}, {r_high}]: {runaways}/20 runaway. Min={min(windows):.1f}s, Max={max(windows):.1f}s, Median={np.median(windows):.1f}s")

