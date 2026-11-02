"""Scenario builders for EV battery fault and thermal runaway simulation.

Scenarios:
 1. Normal drive (WLTP-like dynamic current, regenerative braking)
 2. Normal charge (CC-CV profile)
 3. Hot ambient (starts in Class 0 at 20-25C, transitions to Class 1 as pack warms in 46-50C ambient)
 4. External smoke/CO2 interference (healthy battery, external CO peaks 30-300 ppm)
 5. Overcurrent (sustained 15.5-19.5 A for >= 40 s of Class 2)
 6. Over-discharge (deep discharge below 2.8 V cutoff)
 7. Cooling failure (starts in Class 0, fan fails at t_fail, transitions to Class 1 then Class 3)
 8. Overcharge leading to thermal runaway (dendrite short at 111-115% SOC)
 9. Internal short circuit leading to thermal runaway (R_short ~ 0.18-0.28 Ohm)
 10. Overcharge aborted by BMS at 100-112% SOC (no runaway)
 11. Soft internal short (2-6 Ohm, no runaway)
"""

import numpy as np


SCENARIO_NAMES = [
    "normal_drive",
    "normal_charge",
    "hot_ambient",
    "external_gas",
    "overcurrent",
    "over_discharge",
    "cooling_failure",
    "overcharge_runaway",
    "internal_short_runaway",
    "overcharge_aborted",
    "soft_internal_short",
]


def generate_drive_cycle(
    duration_s: int,
    rng: np.random.RandomState,
    max_disch_c: float = 2.5,
    max_regen_c: float = 1.0,
    mean_load: float = 2.0,
) -> np.ndarray:
    """Generate dynamic synthetic driving profile.
    
    Dissertation constraints: discharge peaks <= 2.5C (12.5 A), regen <= 1.0C (-5.0 A).
    """
    t = np.arange(duration_s)
    current = np.zeros(duration_s, dtype=float)
    
    # Baseline highway/city mix
    current += mean_load + 2.0 * np.sin(2.0 * np.pi * t / 180.0) + 1.5 * np.cos(2.0 * np.pi * t / 60.0)
    
    # Random acceleration bursts and regenerative braking events
    num_events = duration_s // 25
    for _ in range(num_events):
        event_time = rng.randint(0, max(1, duration_s - 15))
        event_len = rng.randint(4, 12)
        if rng.rand() < 0.75:
            amp = rng.uniform(3.0, 8.5)
        else:
            amp = rng.uniform(-4.5, -1.0)
        current[event_time : event_time + event_len] += amp

    # Hard clip strictly to dissertation specifications (<= 12.5 A discharge, <= -5.0 A regen)
    current = np.clip(current, -max_regen_c * 5.0, max_disch_c * 5.0)
    return current


def get_scenario_config(scenario: str, seed: int, duration_s: int = 1800) -> dict:
    """Produce physics configuration, current profile, and fault parameters for a scenario."""
    rng = np.random.RandomState(seed)
    
    # Base cell variations: capacity +/- 2%, R0 +/- 5%
    q_nom_ah = 5.0 * (1.0 + rng.uniform(-0.02, 0.02, size=4))
    r0_ohm = 0.015 * (1.0 + rng.uniform(-0.05, 0.05, size=4))
    r1_ohm = np.full(4, 0.008)
    c1_farad = np.full(4, 3750.0)
    
    # Ambient temperature: randomized 5 - 40 C in all non-hot scenarios
    t_amb = rng.uniform(5.0, 40.0)
    initial_temp = t_amb
    fan_failed = False
    fan_fail_time = None
    
    # Current profile array (1 Hz)
    current = np.zeros(duration_s, dtype=float)
    base_soc = 0.80
    initial_soc = np.full(4, base_soc) + rng.uniform(-0.002, 0.002, size=4)
    
    # Overcharge parameters
    soc_gas_onset = rng.uniform(1.03, 1.10)        # Gas onset SOC (103 - 110%)
    soc_dendrite_trigger = rng.uniform(1.11, 1.15) # Dendrite short trigger SOC (111 - 115%)
    r_short_dendrite = rng.uniform(0.18, 0.28)     # Dendrite short resistance
    soc_cutoff = None

    # Fault arrays
    r_short_profiles = np.full((duration_s, 4), np.inf, dtype=float)
    ext_co2 = np.zeros(duration_s, dtype=float)
    ext_co = np.zeros(duration_s, dtype=float)
    ext_smoke = np.zeros(duration_s, dtype=float)

    if scenario == "normal_drive":
        base_soc = rng.uniform(0.70, 0.85)
        initial_soc = np.full(4, base_soc) + rng.uniform(-0.002, 0.002, size=4)
        current = generate_drive_cycle(duration_s, rng, max_disch_c=2.4, max_regen_c=0.95, mean_load=2.5)

    elif scenario == "normal_charge":
        # CC-CV charging: 0.5C to 0.8C (2.5A to 4.0A charge -> negative current)
        i_chg = -rng.uniform(2.5, 4.0)
        base_soc = rng.uniform(0.25, 0.40)
        initial_soc = np.full(4, base_soc) + rng.uniform(-0.002, 0.002, size=4)
        current[:] = i_chg

    elif scenario == "hot_ambient":
        # Hot ambient: 46 - 50 C
        # Starts in Class 0 (pack at 20-25 C) and transitions to Class 1 as pack warms
        t_amb = rng.uniform(46.0, 50.0)
        initial_temp = rng.uniform(20.0, 25.0)
        base_soc = rng.uniform(0.65, 0.80)
        initial_soc = np.full(4, base_soc) + rng.uniform(-0.002, 0.002, size=4)
        current = generate_drive_cycle(duration_s, rng, max_disch_c=1.5, max_regen_c=0.6, mean_load=1.5)

    elif scenario == "external_gas":
        base_soc = rng.uniform(0.65, 0.80)
        initial_soc = np.full(4, base_soc) + rng.uniform(-0.002, 0.002, size=4)
        current = generate_drive_cycle(duration_s, rng, max_disch_c=1.5, max_regen_c=0.6, mean_load=1.5)
        # External puff around t = 150s - 300s
        puff_start = rng.randint(150, 300)
        puff_len = rng.randint(45, 90)
        # Target peak CO between 30 and 300 ppm (e.g. 50 to 220 ppm)
        target_peak_co = rng.uniform(50.0, 220.0)
        co_rate = (target_peak_co / puff_len) + (target_peak_co * (1.0 / 400.0))
        target_peak_co2 = rng.uniform(1600.0, 3200.0)
        co2_rate = (target_peak_co2 / puff_len) + (target_peak_co2 * (1.0 / 400.0))
        target_peak_smoke = rng.uniform(2.0, 5.0)
        smoke_rate = (target_peak_smoke / puff_len)

        ext_co[puff_start : puff_start + puff_len] = co_rate
        ext_co2[puff_start : puff_start + puff_len] = co2_rate
        ext_smoke[puff_start : puff_start + puff_len] = smoke_rate

    elif scenario == "overcurrent":
        base_soc = rng.uniform(0.65, 0.80)
        initial_soc = np.full(4, base_soc) + rng.uniform(-0.002, 0.002, size=4)
        current = generate_drive_cycle(duration_s, rng, max_disch_c=1.5, max_regen_c=0.6, mean_load=1.5)
        # Sustained 15.5 - 19.5 A pulse for >= 40 s of Class 2 (inject 50 s pulse)
        p_start = rng.randint(120, 220)
        p_len = rng.randint(48, 55)  # 48 - 55 seconds
        current[p_start : p_start + p_len] = rng.uniform(16.0, 19.0)

    elif scenario == "over_discharge":
        # Discharging until cell voltage collapses below 2.8 V
        base_soc = rng.uniform(0.06, 0.09)
        initial_soc = np.full(4, base_soc) + rng.uniform(-0.001, 0.001, size=4)
        current[:] = rng.uniform(4.0, 6.0)

    elif scenario == "cooling_failure":
        # Must start in Class 0 and transition to Class 1, then Class 3!
        # Summer highway cruise: Tamb in 38.5 - 42.5 C (< 45 C, strictly starts in Class 0).
        t_amb = rng.uniform(38.5, 42.5)
        initial_temp = t_amb
        fan_failed = False  # starts working in Class 0
        fan_fail_time = rng.randint(50, 100)  # fan fails at t = 50 - 100 s
        base_soc = rng.uniform(0.92, 0.98)
        initial_soc = np.full(4, base_soc) + rng.uniform(-0.002, 0.002, size=4)
        # Sustained highway drive: 10.5A - 12.4A (<= 2.48C, <= 12.5A dissertation peak)
        current = generate_drive_cycle(duration_s, rng, max_disch_c=2.48, max_regen_c=0.0, mean_load=11.0)

    elif scenario == "overcharge_runaway":
        # Continuous overcharge past 100% SOC leading to thermal runaway
        base_soc = rng.uniform(0.92, 0.96)
        initial_soc = np.full(4, base_soc) + rng.uniform(-0.002, 0.002, size=4)
        # 1.2C - 1.5C continuous charge (6.0A - 7.5A)
        i_chg = -rng.uniform(6.0, 7.5)
        current[:] = i_chg

    elif scenario == "internal_short_runaway":
        base_soc = rng.uniform(0.85, 0.95)
        initial_soc = np.full(4, base_soc) + rng.uniform(-0.002, 0.002, size=4)
        current = generate_drive_cycle(duration_s, rng, max_disch_c=1.2, max_regen_c=0.5, mean_load=1.5)
        # Hard short in group 1 (R_short ~ 0.18 - 0.28 Ohm)
        short_start = rng.randint(60, 140)
        r_short_val = rng.uniform(0.18, 0.28)
        r_short_profiles[short_start:, 1] = r_short_val

    elif scenario == "overcharge_aborted":
        # Overcharge aborted by BMS in 100 - 112% SOC
        base_soc = rng.uniform(0.94, 0.97)
        initial_soc = np.full(4, base_soc) + rng.uniform(-0.001, 0.001, size=4)
        i_chg = -rng.uniform(4.5, 6.5)
        current[:] = i_chg
        soc_cutoff = rng.uniform(1.00, 1.12)

    elif scenario == "soft_internal_short":
        # Soft short R_ISC 2.0 - 6.0 Ohm in group 2
        base_soc = rng.uniform(0.75, 0.85)
        initial_soc = np.full(4, base_soc) + rng.uniform(-0.002, 0.002, size=4)
        current = generate_drive_cycle(duration_s, rng, max_disch_c=1.5, max_regen_c=0.5, mean_load=2.0)
        short_start = rng.randint(60, 150)
        r_isc_val = rng.uniform(2.2, 5.5)
        r_short_profiles[short_start:, 2] = r_isc_val

    else:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {SCENARIO_NAMES}")

    return {
        "scenario": scenario,
        "seed": seed,
        "duration_s": duration_s,
        "rng": rng,
        "q_nom_ah": q_nom_ah,
        "r0_ohm": r0_ohm,
        "r1_ohm": r1_ohm,
        "c1_farad": c1_farad,
        "initial_soc": initial_soc,
        "initial_temp": initial_temp,
        "t_amb": t_amb,
        "fan_failed": fan_failed,
        "fan_fail_time": fan_fail_time,
        "current": current,
        "r_short_profiles": r_short_profiles,
        "ext_co2": ext_co2,
        "ext_co": ext_co,
        "ext_smoke": ext_smoke,
        "soc_gas_onset": soc_gas_onset,
        "soc_dendrite_trigger": soc_dendrite_trigger,
        "r_short_dendrite": r_short_dendrite,
        "soc_cutoff": soc_cutoff,
    }
