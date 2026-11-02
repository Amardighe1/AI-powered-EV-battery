import sys
sys.path.insert(0, ".")
from src.simulation.runner import generate_run
from src.simulation.scenarios import get_scenario_config

cfg = get_scenario_config("internal_short_runaway", seed=3)
res = generate_run("internal_short_runaway", seed=3)
t = res["time"]
tc = res["true_states"]["t_core"]
sh = res["true_states"]["self_heating_rate"]
qc = res["true_states"]["q_chem"]
v = res["true_states"]["v_terminal"]
soc = res["true_states"]["soc"]

print("Initial soc:", cfg["initial_soc"])
print("R_short profile at start:", cfg["r_short_profiles"][100:105, 1])
print(f"Max Tc: {tc.max(axis=0)}")
print(f"Max Q_chem: {qc.max(axis=0)}")
print(f"Final SOC: {soc[-1]}")
print(f"Final V: {v[-1]}")

