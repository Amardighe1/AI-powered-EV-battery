import sys
sys.path.insert(0, ".")
from src.simulation.runner import generate_run
import src.simulation.scenarios as sc_mod

for r_min, r_max in [(0.20, 0.35), (0.22, 0.38), (0.25, 0.40)]:
    windows = []
    runaways = []
    for seed in range(10):
        # Temporarily monkey patch r_short range
        orig_get = sc_mod.get_scenario_config
        def patched_get(scenario, s, duration_s=1800):
            cfg = orig_get(scenario, s, duration_s)
            if scenario == "internal_short_runaway":
                val = cfg["rng"].uniform(r_min, r_max)
                short_start = cfg["rng"].randint(60, 140)
                cfg["r_short_profiles"][:] = np.inf
                cfg["r_short_profiles"][short_start:, 1] = val
            return cfg
        import numpy as np
        sc_mod.get_scenario_config = patched_get
        
        res = generate_run("internal_short_runaway", seed=seed)
        sc_mod.get_scenario_config = orig_get
        meta = res["metadata"]
        if meta["t_runaway_onset"] is not None:
            runaways.append(meta["t_runaway_onset"])
            windows.append(meta["class_4_duration_s"])
    print(f"R_short [{r_min:.2f}, {r_max:.2f}]: {len(runaways)}/10 runaways, windows: min={min(windows) if windows else None:.1f}s, max={max(windows) if windows else None:.1f}s, median={np.median(windows) if windows else None:.1f}s")

