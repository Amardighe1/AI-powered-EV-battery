"""Generate publication-grade diagnostic plots for Scenarios 1, 4, 8, and 9.

Shows:
 - Cell Voltages (true & sensor)
 - Pack Current (true & sensor)
 - Temperatures: T_core (true), T_surf (true), DS18B20 T1..4 (sensor), K-type T_hs (sensor)
 - Gases: CO2 (true & NDIR sensor), CO (true & MQ-7 sensor), Smoke (true & MQ-2 sensor)
 - Ground Truth Class Timeline (0..5) with highlighted early warning (4) and fire (5) zones.
"""

import os
import sys
import numpy as np

# Set writable MPLCONFIGDIR before importing matplotlib
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "."), "mpl_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from src.simulation.runner import generate_run


def plot_scenario(scenario: str, seed: int = 42, output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)
    res = generate_run(scenario, seed=seed)
    
    t = res["time"]
    sensors = res["sensors"] # (N, 14)
    true_states = res["true_states"]
    labels = res["labels"]
    meta = res["metadata"]

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True, gridspec_kw={"height_ratios": [1.2, 0.9, 1.3, 1.2, 0.8]})
    fig.suptitle(f"EV Battery Fault & Fire Detection - Scenario: {scenario.upper()} (Seed={seed})", fontsize=14, fontweight="bold", y=0.99)

    # 1. Voltages
    ax0 = axes[0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i in range(4):
        ax0.plot(t, true_states["v_terminal"][:, i], label=f"V{i+1} True", color=colors[i], lw=1.6)
        ax0.plot(t, sensors[:, i], label=f"V{i+1} Sensor", color=colors[i], lw=0.7, ls="--", alpha=0.7)
    ax0.axhline(4.25, color="red", ls=":", lw=1.2, label="V_max Fault (4.25V)")
    ax0.axhline(2.80, color="darkred", ls=":", lw=1.2, label="V_min Fault (2.80V)")
    ax0.set_ylabel("Voltage (V)", fontweight="bold")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", ncol=3, fontsize=8)

    # 2. Current
    ax1 = axes[1]
    ax1.plot(t, true_states["i_pack"], label="Pack Current True", color="#9467bd", lw=1.5)
    ax1.plot(t, sensors[:, 4], label="Current Sensor", color="#8c564b", lw=0.8, ls="--", alpha=0.7)
    ax1.axhline(15.0, color="orange", ls=":", lw=1.0, label="15A Overcurrent Threshold")
    ax1.axhline(-15.0, color="orange", ls=":", lw=1.0)
    ax1.set_ylabel("Current (A)", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    # 3. Temperatures
    ax2 = axes[2]
    # True core max
    ax2.plot(t, np.max(true_states["t_core"], axis=1), label="Max T_core (True)", color="black", lw=2.0)
    for i in range(4):
        ax2.plot(t, true_states["t_surf"][:, i], label=f"T{i+1}_surf (True)", color=colors[i], lw=1.2, alpha=0.6)
        ax2.plot(t, sensors[:, 5 + i], label=f"T{i+1} DS18B20", color=colors[i], lw=0.8, ls=":")
    # K-type hot-spot sensor
    ax2.plot(t, sensors[:, 9], label="T_hs K-type (Sensor)", color="crimson", lw=1.8, ls="-.")
    # Ambient sensor
    ax2.plot(t, sensors[:, 10], label="T_amb (Sensor)", color="gray", lw=1.0, ls="--")
    ax2.axhline(60.0, color="magenta", ls=":", lw=1.2, label="60°C Thermal Fault")
    ax2.set_ylabel("Temp (°C)", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", ncol=3, fontsize=8)

    # 4. Gases
    ax3 = axes[3]
    ax3_twin = ax3.twinx()
    
    # CO2 on left axis
    l1 = ax3.plot(t, true_states["co2"], label="CO2 True (ppm)", color="forestgreen", lw=1.8)
    l2 = ax3.plot(t, sensors[:, 13], label="CO2 NDIR (Sensor)", color="darkgreen", lw=1.0, ls="--")
    ax3.set_ylabel("CO2 (ppm)", color="darkgreen", fontweight="bold")
    ax3.tick_params(axis="y", labelcolor="darkgreen")
    ax3.grid(True, alpha=0.3)

    # CO and Smoke on right axis
    l3 = ax3_twin.plot(t, true_states["co"], label="CO True (ppm)", color="darkorange", lw=1.8)
    l4 = ax3_twin.plot(t, sensors[:, 12], label="CO MQ-7 (Sensor)", color="chocolate", lw=1.0, ls="--")
    l5 = ax3_twin.plot(t, sensors[:, 11], label="Smoke MQ-2 (Sensor)", color="purple", lw=1.0, ls=":")
    ax3_twin.set_ylabel("CO (ppm) & Smoke (%/m)", color="chocolate", fontweight="bold")
    ax3_twin.tick_params(axis="y", labelcolor="chocolate")
    
    lines = l1 + l2 + l3 + l4 + l5
    labels_g = [l.get_label() for l in lines]
    ax3.legend(lines, labels_g, loc="upper left", ncol=3, fontsize=8)

    # 5. Class Timeline
    ax4 = axes[4]
    class_colors = {0: "green", 1: "gold", 2: "darkorange", 3: "red", 4: "magenta", 5: "black"}
    ax4.step(t, labels, where="post", color="blue", lw=1.8, label="Physical State Ground Truth Label")
    ax4.set_yticks([0, 1, 2, 3, 4, 5])
    ax4.set_yticklabels(["0:Normal", "1:Abnormal", "2:Elec Fault", "3:Therm Fault", "4:Early Warn", "5:Runaway/Fire"])
    ax4.set_ylabel("Class", fontweight="bold")
    ax4.set_xlabel("Time (s)", fontweight="bold")
    ax4.set_ylim(-0.5, 5.5)
    ax4.grid(True, alpha=0.3)

    # Shade Class 4 and Class 5 regions
    is_c4 = (labels == 4)
    is_c5 = (labels == 5)
    if np.any(is_c4):
        ax4.fill_between(t, -0.5, 5.5, where=is_c4, color="magenta", alpha=0.25, step="post", label="Class 4 Early Warning Active")
    if np.any(is_c5):
        ax4.fill_between(t, -0.5, 5.5, where=is_c5, color="black", alpha=0.35, step="post", label="Class 5 Critical Runaway Active")
    ax4.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"scenario_{scenario}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Generated plot: {out_path}")
    return out_path


if __name__ == "__main__":
    for sc in ["normal_drive", "external_gas", "overcharge_runaway", "internal_short_runaway"]:
        plot_scenario(sc, seed=42)

