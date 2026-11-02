"""Feature extraction pipeline for EV battery fault and fire detection.

Transforms raw sensor windows of shape (60, 14) into 34 engineered features:
 - 10 base signals: T_max, dT_cell, V_max, V_min, V_spread, I, T_amb, Smoke, CO, CO2
 - For each of the 10: last (t=59), mean (60s), least-squares slope (60s) -> 30 features
 - 15 s fast least-squares slopes for T_max, V_min, CO2 -> 3 features
 - Coulomb-counted SOC initialized from first 10 s OCV -> 1 feature
Total: 34 features.

CRITICAL: Strictly uses sensor channels only. Never touches true state.
"""

from typing import List, Optional
import numpy as np

# Precomputed least-squares regression weights for 60s and 15s windows
_T60 = np.arange(60, dtype=float)
_W60 = (_T60 - 29.5) / 17995.0

_T15 = np.arange(15, dtype=float)
_W15 = (_T15 - 7.0) / 280.0

# Empirical NMC OCV-to-SOC lookup table for 21700 NMC cells
def _build_ocv_table():
    soc_grid = np.linspace(0.0, 1.25, 2501)
    # Replicate forward OCV curve
    s = np.clip(soc_grid, 0.0, 1.0)
    ocv_nominal = (
        3.0
        + 0.65 * s
        + 0.45 * (s ** 2)
        - 0.50 * (s ** 3)
        + 0.60 * (s ** 4)
        + 0.03 * np.log(np.maximum(s, 1e-4))
        - 0.08 * np.exp(-35.0 * np.maximum(s, 0.0))
    )
    overcharge_delta = np.maximum(0.0, soc_grid - 1.0)
    ocv_overcharge = 4.20 + 2.5 * overcharge_delta + 1.2 * (overcharge_delta ** 2)
    ocv_curve = np.where(soc_grid > 1.0, ocv_overcharge, ocv_nominal)
    return ocv_curve, soc_grid

_OCV_GRID, _SOC_GRID = _build_ocv_table()


def soc_from_ocv(v_cell: float) -> float:
    """Invert NMC OCV curve to estimate initial SOC from resting cell voltage."""
    return float(np.interp(v_cell, _OCV_GRID, _SOC_GRID))


def compute_run_soc_series(sensors: np.ndarray, q_nom_ah: float = 5.0) -> np.ndarray:
    """Compute running Coulomb-counted SOC for a full sensor run.
    
    1. Uses the average cell voltage over the first 10 seconds to estimate initial SOC from OCV.
    2. Integrates measured sensor current from t=10 onwards.
    """
    n_samples = len(sensors)
    soc_series = np.zeros(n_samples, dtype=float)
    
    # First 10 seconds: initial OCV estimate
    v_init = np.mean(sensors[:10, 0:4])
    initial_soc = soc_from_ocv(v_init)
    soc_series[:10] = initial_soc
    
    q_nom_as = q_nom_ah * 3600.0
    current_soc = initial_soc
    
    for t in range(10, n_samples):
        # Current channel is index 4 (positive = discharge, negative = charge)
        i_sensor = sensors[t, 4]
        current_soc -= (i_sensor * 1.0) / q_nom_as
        soc_series[t] = current_soc
        
    return soc_series


# Feature names in deterministic order (34 total)
FEATURE_NAMES: List[str] = [
    # 10 base signals x 3 statistics (last, mean, slope_60s)
    "T_max_last", "T_max_mean", "T_max_slope_60s",
    "dT_cell_last", "dT_cell_mean", "dT_cell_slope_60s",
    "V_max_last", "V_max_mean", "V_max_slope_60s",
    "V_min_last", "V_min_mean", "V_min_slope_60s",
    "V_spread_last", "V_spread_mean", "V_spread_slope_60s",
    "I_last", "I_mean", "I_slope_60s",
    "T_amb_last", "T_amb_mean", "T_amb_slope_60s",
    "Smoke_last", "Smoke_mean", "Smoke_slope_60s",
    "CO_last", "CO_mean", "CO_slope_60s",
    "CO2_last", "CO2_mean", "CO2_slope_60s",
    # 3 fast 15s slopes
    "T_max_slope_15s",
    "V_min_slope_15s",
    "CO2_slope_15s",
    # 1 Coulomb-counted SOC feature
    "SOC",
]


def compute_features(window: np.ndarray, soc: Optional[float] = None) -> np.ndarray:
    """Compute the 34-dimensional feature vector from a (60, 14) sensor window.
    
    Args:
        window: np.ndarray of shape (60, 14) containing 1 Hz sensor readings.
                Channels: [V1, V2, V3, V4, I, T1, T2, T3, T4, T_hs, T_amb, Smoke, CO, CO2]
        soc: Optional float with running Coulomb-counted SOC at the window end.
             If None, estimated from the window's first 10 s OCV and integrated current.
             
    Returns:
        np.ndarray of shape (34,) with extracted features.
    """
    assert window.shape == (60, 14), f"Expected window shape (60, 14), got {window.shape}"

    # Extract 10 base series across the 60 samples
    # 1. T_max: max across T1..4 (cols 5..8) and T_hs (col 9)
    t_max = np.max(window[:, 5:10], axis=1)

    # 2. dT_cell: surface spread max(T1..4) - min(T1..4)
    dt_cell = np.max(window[:, 5:9], axis=1) - np.min(window[:, 5:9], axis=1)

    # 3. V_max: max(V1..4)
    v_max = np.max(window[:, 0:4], axis=1)

    # 4. V_min: min(V1..4)
    v_min = np.min(window[:, 0:4], axis=1)

    # 5. V_spread: V_max - V_min
    v_spread = v_max - v_min

    # 6. I: Current sensor (col 4)
    i_curr = window[:, 4]

    # 7. T_amb: Ambient temperature (col 10)
    t_amb = window[:, 10]

    # 8. Smoke: Smoke obscuration (col 11)
    smoke = window[:, 11]

    # 9. CO: CO concentration (col 12)
    co = window[:, 12]

    # 10. CO2: CO2 concentration (col 13)
    co2 = window[:, 13]

    signals = [t_max, dt_cell, v_max, v_min, v_spread, i_curr, t_amb, smoke, co, co2]

    feats = []
    # 10 signals x 3 stats
    for sig in signals:
        feats.append(sig[-1])                    # last
        feats.append(float(np.mean(sig)))        # mean
        feats.append(float(np.dot(sig, _W60)))   # slope_60s

    # 3 fast 15s slopes (indices 45..59)
    feats.append(float(np.dot(t_max[-15:], _W15)))
    feats.append(float(np.dot(v_min[-15:], _W15)))
    feats.append(float(np.dot(co2[-15:], _W15)))

    # 1 Coulomb-counted SOC
    if soc is not None:
        feats.append(float(soc))
    else:
        # Fallback: estimate from first 10s of window
        v_init = float(np.mean(window[:10, 0:4]))
        soc_est = soc_from_ocv(v_init)
        # integrate remaining 50s current
        soc_est -= float(np.sum(window[10:, 4])) / (5.0 * 3600.0)
        feats.append(soc_est)

    return np.array(feats, dtype=float)

