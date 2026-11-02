"""Simulation runner and ground-truth labeler.

Pure Python and NumPy.
"""

from typing import Dict, Any
import numpy as np

from ..models.ecm import PackECM
from ..models.thermal import PackThermal
from ..models.gas import EnclosureGas
from ..models.sensors import SensorSuite, CHANNEL_NAMES
from .scenarios import get_scenario_config


def generate_run(
    scenario: str,
    seed: int = 42,
    duration_s: int = 1800,
    noise_scale: float = 1.0,
) -> Dict[str, Any]:
    """Execute a single simulation run for a given scenario and seed.
    
    Returns:
        dict containing:
            - 'time': np.ndarray of shape (N,)
            - 'sensors': np.ndarray of shape (N, 14)
            - 'sensor_names': list of 14 channel names
            - 'true_states': dict of ground-truth physical quantities
            - 'estimated_soc': np.ndarray of shape (N, 4) with coulomb-counting error and drift
            - 'labels': np.ndarray of shape (N,) with integers 0..5
            - 'time_to_runaway_s': np.ndarray of shape (N,) with seconds to runaway (or NaN)
            - 'metadata': dict with scenario details, onset times, duration, max_temp
    """
    config = get_scenario_config(scenario, seed, duration_s=duration_s)
    rng = config["rng"]

    ecm = PackECM(
        q_nom_ah=config["q_nom_ah"],
        r0_ohm=config["r0_ohm"],
        r1_ohm=config["r1_ohm"],
        c1_farad=config["c1_farad"],
        initial_soc=config["initial_soc"],
    )

    thermal = PackThermal(
        t_amb=config["t_amb"],
        fan_failed=config["fan_failed"],
        initial_temp=config.get("initial_temp", None),
    )

    gas = EnclosureGas(
        rng=rng,
        co2_bg=420.0,
        air_exchange_rate=1.0 / 400.0,
    )

    sensor_suite = SensorSuite(
        rng=rng,
        noise_scale=noise_scale,
    )

    # Coulomb-counting estimator with initial SOC error (+/- 4%) and current sensor drift
    soc_coulomb_error = rng.uniform(-0.04, 0.04, size=4)
    est_soc = np.clip(config["initial_soc"] + soc_coulomb_error, 0.0, 1.2)
    q_nom_as = config["q_nom_ah"] * 3600.0

    # Overcharge dendrite short latching (prevents limit-cycle voltage chatter)
    has_dendrite_short = np.zeros(4, dtype=bool)

    # Storage arrays
    times = []
    sensors_list = []
    est_soc_list = []
    
    # Ground truth records
    v_terminal_list = []
    i_pack_list = []
    t_core_list = []
    t_surf_list = []
    soc_list = []
    q_chem_list = []
    self_heating_rate_list = []
    co2_true_list = []
    co_true_list = []
    smoke_true_list = []
    has_true_ovc_gas_list = []

    t_runaway_onset = None
    stop_step = None
    step_dt = 1.0  # 1 Hz sampling step

    current_profile = config["current"]
    r_short_profiles = config["r_short_profiles"]
    ext_co2 = config["ext_co2"]
    ext_co = config["ext_co"]
    ext_smoke = config["ext_smoke"]
    soc_cutoff = config.get("soc_cutoff", None)
    fan_fail_time = config.get("fan_fail_time", None)
    soc_gas_onset = config.get("soc_gas_onset", 1.08)
    soc_dendrite_trigger = config.get("soc_dendrite_trigger", 1.13)
    r_short_dendrite = config.get("r_short_dendrite", 0.22)

    for step_idx in range(duration_s):
        t_sec = float(step_idx)
        
        # 1. Update dynamic scenario transitions (e.g. fan failure at fan_fail_time)
        if fan_fail_time is not None and t_sec >= fan_fail_time:
            thermal.set_fan_failed(True)

        # 2. Determine demand current with scenario-specific dynamic BMS protections
        i_cmd = current_profile[step_idx]

        if scenario == "normal_charge":
            # BMS CC-CV cut-off when any cell reaches 100% SOC
            if np.max(ecm.soc) >= 1.0:
                i_cmd = 0.0

        elif scenario == "overcharge_aborted":
            # BMS intervenes and cuts charging at randomized soc_cutoff (100% - 112%)
            if soc_cutoff is not None and np.max(ecm.soc) >= soc_cutoff:
                i_cmd = 0.0

        elif scenario == "cooling_failure":
            # Vehicle BMS idles current if SOC drops low to avoid over-discharge conflation
            if np.min(ecm.soc) <= 0.12:
                i_cmd = 0.0

        elif scenario == "over_discharge":
            # Once deeply discharged below 1.5 V, current stops
            if np.min(ecm.soc) < 0.02:
                i_cmd = 0.0

        # Update short circuit resistance from scenario schedule
        ecm.r_short = r_short_profiles[step_idx].copy()

        # For overcharge runaway: separator breakdown / lithium dendrite short
        # triggers when cell SOC exceeds soc_dendrite_trigger (111-115%) or core temp > 110 C.
        # Once triggered, the physical short is permanently LATCHED to avoid voltage chatter.
        if scenario == "overcharge_runaway":
            for grp_idx in range(4):
                if ecm.soc[grp_idx] >= soc_dendrite_trigger or thermal.t_core[grp_idx] >= 110.0:
                    has_dendrite_short[grp_idx] = True
                if has_dendrite_short[grp_idx]:
                    ecm.r_short[grp_idx] = r_short_dendrite

        # 3. Advance electrical model
        elec_out = ecm.step(pack_current=i_cmd, dt=step_dt, t_core=thermal.t_core)
        v_term = elec_out["v_terminal"]
        soc = elec_out["soc"]
        q_electrical = elec_out["q_joule"] + elec_out["q_short"] + elec_out["q_overcharge"]

        # 4. Advance thermal model (with internal sub-stepping)
        therm_out = thermal.step(q_electrical=q_electrical, dt=step_dt)
        t_core = therm_out["t_core"]
        t_surf = therm_out["t_surf"]
        q_chem = therm_out["q_chem"]
        self_heating_rate = therm_out["self_heating_rate"]

        # 5. Advance gas model with SEI and Main decomposition off-gas
        gas_out = gas.step(
            dt=step_dt,
            soc=soc,
            v_side=elec_out["v_side"],
            i_group=elec_out["i_group"],
            soc_gas_onset=soc_gas_onset,
            t_core=t_core,
            k1_sei=therm_out["k1"],
            k2_main=therm_out["k2"],
            self_heating_rate=self_heating_rate,
            ext_co2=ext_co2[step_idx],
            ext_co=ext_co[step_idx],
            ext_smoke=ext_smoke[step_idx],
        )

        # 6. Sample sensors
        raw_sensors = sensor_suite.step(
            t_sec=t_sec,
            v_true=v_term,
            i_true=i_cmd,
            t_surf_true=t_surf,
            t_amb_true=thermal.t_amb,
            smoke_true=gas_out["smoke"],
            co_true=gas_out["co"],
            co2_true=gas_out["co2"],
            dt=step_dt,
        )
        sensors_list.append(raw_sensors)

        # Update Coulomb-counting estimated SOC with measured current (includes sensor offset + noise)
        measured_i = raw_sensors[4]
        est_soc -= (measured_i * step_dt) / q_nom_as
        est_soc_list.append(est_soc.copy())

        # 7. Record true states
        times.append(t_sec)
        v_terminal_list.append(v_term.copy())
        i_pack_list.append(i_cmd)
        t_core_list.append(t_core.copy())
        t_surf_list.append(t_surf.copy())
        soc_list.append(soc.copy())
        q_chem_list.append(q_chem.copy())
        self_heating_rate_list.append(self_heating_rate.copy())
        co2_true_list.append(gas_out["co2"])
        co_true_list.append(gas_out["co"])
        smoke_true_list.append(gas_out["smoke"])
        has_true_ovc_gas_list.append(gas_out["has_true_ovc_gas"])

        # 8. Check for thermal runaway onset:
        # Self-heating rate >= 1.0 C/s in any group
        if np.max(self_heating_rate) >= 1.0 and t_runaway_onset is None:
            t_runaway_onset = t_sec
            # Truncate run about 120 s after runaway
            stop_step = step_idx + 120

        if stop_step is not None and step_idx >= stop_step:
            break

    # Convert recorded lists to NumPy arrays
    n_steps = len(times)
    time_arr = np.array(times, dtype=float)
    sensors_arr = np.array(sensors_list, dtype=float)        # (N, 14)
    est_soc_arr = np.array(est_soc_list, dtype=float)        # (N, 4)
    v_term_arr = np.array(v_terminal_list, dtype=float)      # (N, 4)
    i_pack_arr = np.array(i_pack_list, dtype=float)          # (N,)
    t_core_arr = np.array(t_core_list, dtype=float)          # (N, 4)
    t_surf_arr = np.array(t_surf_list, dtype=float)          # (N, 4)
    soc_arr = np.array(soc_list, dtype=float)                # (N, 4)
    q_chem_arr = np.array(q_chem_list, dtype=float)          # (N, 4)
    sh_rate_arr = np.array(self_heating_rate_list, dtype=float) # (N, 4)
    co2_true_arr = np.array(co2_true_list, dtype=float)      # (N,)
    co_true_arr = np.array(co_true_list, dtype=float)        # (N,)
    smoke_true_arr = np.array(smoke_true_list, dtype=float)  # (N,)
    has_true_ovc_gas_arr = np.array(has_true_ovc_gas_list, dtype=bool)

    # --- Ground Truth Label Computation ---
    # Priority: 5 > 4 > 2 > 3 > 1 > 0
    labels = np.zeros(n_steps, dtype=int)
    time_to_runaway_s = np.full(n_steps, np.nan, dtype=float)

    if t_runaway_onset is not None:
        for k in range(n_steps):
            if time_arr[k] <= t_runaway_onset:
                time_to_runaway_s[k] = t_runaway_onset - time_arr[k]
            else:
                time_to_runaway_s[k] = 0.0

    # Determine precursor active window for Class 4:
    # "Precursor = decomposition heat >= 0.3 W in any group, OR true gas generation from overcharge is active.
    # From the first time the precursor condition is true until runaway onset."
    precursor_condition = (np.max(q_chem_arr, axis=1) >= 0.3) | has_true_ovc_gas_arr
    
    precursor_onset_idx = None
    for k in range(n_steps):
        if precursor_condition[k]:
            precursor_onset_idx = k
            break

    runaway_latched = False
    runaway_onset_idx = None
    if t_runaway_onset is not None:
        runaway_onset_idx = int(np.where(time_arr == t_runaway_onset)[0][0])

    for k in range(n_steps):
        # 1. Class 5 check (latched once runaway occurs)
        if runaway_onset_idx is not None and k >= runaway_onset_idx:
            runaway_latched = True
        elif np.max(sh_rate_arr[k]) >= 1.0:
            runaway_latched = True

        if runaway_latched:
            labels[k] = 5
            continue

        # 2. Class 4 check (from first precursor time until runaway onset)
        if precursor_onset_idx is not None and k >= precursor_onset_idx:
            labels[k] = 4
            continue

        # 3. Class 2 check (Electrical fault)
        # Any cell V >= 4.25 or <= 2.8; |I| >= 15 A averaged over 5s, or >= 30 A instantaneous
        v_k = v_term_arr[k]
        i_inst = abs(i_pack_arr[k])
        i_5s_avg = np.mean(np.abs(i_pack_arr[max(0, k - 4) : k + 1]))
        
        is_elec_fault = (
            np.any(v_k >= 4.25)
            or np.any(v_k <= 2.8)
            or (i_5s_avg >= 15.0)
            or (i_inst >= 30.0)
        )
        if is_elec_fault:
            labels[k] = 2
            continue

        # 4. Class 3 check (Thermal fault)
        # Core T >= 60 C (electrical state normal)
        is_thermal_fault = np.max(t_core_arr[k]) >= 60.0
        if is_thermal_fault:
            labels[k] = 3
            continue

        # 5. Class 1 check (Abnormal)
        # Core T 45-60 C, or surface spread >= 8 C, or cell imbalance >= 60 mV,
        # or external gas (CO > 30 ppm or CO2 > 1200 ppm) with a healthy battery
        max_core_t = np.max(t_core_arr[k])
        surf_spread = np.max(t_surf_arr[k]) - np.min(t_surf_arr[k])
        cell_imbalance = np.max(v_k) - np.min(v_k)
        has_ext_gas = (co_true_arr[k] > 30.0) or (co2_true_arr[k] > 1200.0)

        is_abnormal = (
            (45.0 <= max_core_t < 60.0)
            or (surf_spread >= 8.0)
            or (cell_imbalance >= 0.060)
            or has_ext_gas
        )
        if is_abnormal:
            labels[k] = 1
            continue

        # 6. Class 0 (Normal)
        labels[k] = 0

    class_4_len = 0.0
    if precursor_onset_idx is not None and runaway_onset_idx is not None:
        class_4_len = time_arr[runaway_onset_idx] - time_arr[precursor_onset_idx]

    return {
        "time": time_arr,
        "sensors": sensors_arr,
        "sensor_names": list(CHANNEL_NAMES),
        "estimated_soc": est_soc_arr,
        "true_states": {
            "v_terminal": v_term_arr,
            "i_pack": i_pack_arr,
            "t_core": t_core_arr,
            "t_surf": t_surf_arr,
            "soc": soc_arr,
            "q_chem": q_chem_arr,
            "self_heating_rate": sh_rate_arr,
            "co2": co2_true_arr,
            "co": co_true_arr,
            "smoke": smoke_true_arr,
        },
        "labels": labels,
        "time_to_runaway_s": time_to_runaway_s,
        "metadata": {
            "scenario": scenario,
            "seed": seed,
            "duration_s": time_arr[-1],
            "n_samples": n_steps,
            "t_runaway_onset": t_runaway_onset,
            "t_precursor_onset": time_arr[precursor_onset_idx] if precursor_onset_idx is not None else None,
            "class_4_duration_s": class_4_len,
            "max_t_core": float(np.max(t_core_arr)),
            "max_t_surf": float(np.max(t_surf_arr)),
            "max_co_ppm": float(np.max(co_true_arr)),
            "soc_gas_onset": soc_gas_onset,
            "soc_dendrite_trigger": soc_dendrite_trigger,
        },
    }
