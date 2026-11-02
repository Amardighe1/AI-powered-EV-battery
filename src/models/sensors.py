"""14-channel sensor model with physical lag, noise, offsets, quantization, and duty-cycle holds.

Pure Python and NumPy.
"""

import numpy as np


CHANNEL_NAMES = [
    "V1", "V2", "V3", "V4",
    "I",
    "T1", "T2", "T3", "T4",
    "T_hs",
    "T_amb",
    "Smoke",
    "CO",
    "CO2"
]


class SensorSuite:
    """Simulates 14 raw physical sensor channels at 1 Hz with realistic degradation."""

    def __init__(
        self,
        rng: np.random.RandomState = None,
        noise_scale: float = 1.0,
    ):
        if rng is None:
            rng = np.random.RandomState(42)

        self.rng = rng
        self.noise_scale = noise_scale

        # 1. Voltage offsets (+/- 5 mV)
        self.v_offsets = rng.uniform(-0.005, 0.005, size=4)
        
        # 2. Current offset (+/- 0.08 A)
        self.i_offset = rng.uniform(-0.08, 0.08)

        # 3. DS18B20 offsets (+/- 0.4 C) for T1..4 and T_amb
        self.ds_offsets = rng.uniform(-0.4, 0.4, size=5)

        # 4. K-type offset (+/- 1.0 C) for T_hs
        self.k_offset = rng.uniform(-1.0, 1.0)

        # 5. Gas initial offsets / biases
        self.smoke_offset = rng.uniform(-0.1, 0.1)
        self.co_offset = rng.uniform(-1.0, 1.0)
        self.co2_offset = rng.uniform(-15.0, 15.0)

        # Sensor state variables for lag filters (1st-order low-pass)
        self.lag_t_surf = None  # 4 channels, tau = 8 s
        self.lag_t_hs = None    # 1 channel, tau = 2 s
        self.lag_t_amb = None   # 1 channel, tau = 8 s
        self.lag_smoke = None   # 1 channel, tau = 6 s
        self.lag_co = None      # 1 channel, tau = 15 s
        self.lag_co2 = None     # 1 channel, tau = 45 s

        # MQ-7 duty cycle hold state (valid 90s out of every 150s)
        self.mq7_last_valid = 0.0
        
        # Slow random-walk drift for MQ sensors
        self.smoke_drift = 0.0
        self.co_drift = 0.0

    def step(
        self,
        t_sec: float,
        v_true: np.ndarray,      # (4,) Volts
        i_true: float,           # Amperes
        t_surf_true: np.ndarray, # (4,) deg C
        t_amb_true: float,       # deg C
        smoke_true: float,       # %/m
        co_true: float,          # ppm
        co2_true: float,         # ppm
        dt: float = 1.0,         # Sensor sampling interval (1.0 s)
    ) -> np.ndarray:
        """Process true physical values at time t_sec and return 14 raw sensor readings."""
        # Initialize lag states on first call
        if self.lag_t_surf is None:
            self.lag_t_surf = t_surf_true.copy()
            self.lag_t_hs = float(np.max(t_surf_true))
            self.lag_t_amb = float(t_amb_true)
            self.lag_smoke = float(smoke_true)
            self.lag_co = float(co_true)
            self.lag_co2 = float(co2_true)
            self.mq7_last_valid = float(co_true)

        # --- A. Voltages V1..4 (instantaneous, 3 mV noise, 5 mV offset, 1 mV step) ---
        v_noisy = v_true + self.v_offsets + self.rng.normal(0.0, 0.003 * self.noise_scale, size=4)
        # Quantization: 1 mV step (0.001 V)
        v_quant = np.round(v_noisy / 0.001) * 0.001

        # --- B. Current I (instantaneous, 0.06 A noise, 0.08 A offset, clip +/- 20 A) ---
        i_noisy = i_true + self.i_offset + self.rng.normal(0.0, 0.06 * self.noise_scale)
        # Current quantization (10 mA resolution) and clipping
        i_quant = np.round(i_noisy / 0.01) * 0.01
        i_out = np.clip(i_quant, -20.0, 20.0)

        # --- C. DS18B20 Surface Temperatures T1..4 (tau = 8s, offset +/-0.4C, step 0.0625C, clip 125C) ---
        alpha_ds = 1.0 - np.exp(-dt / 8.0)
        self.lag_t_surf += alpha_ds * (t_surf_true - self.lag_t_surf)
        t_surf_noisy = self.lag_t_surf + self.ds_offsets[:4]
        # Quantization: step 0.0625 C, clip 125 C
        t_surf_quant = np.round(t_surf_noisy / 0.0625) * 0.0625
        t_surf_out = np.clip(t_surf_quant, -40.0, 125.0)

        # --- D. K-type Hot-Spot T_hs (derived from hottest group's SURFACE, tau = 2s, offset +/-1C, step 0.25C) ---
        # Note: Must NEVER use core temperature!
        hottest_surface = float(np.max(t_surf_true))
        alpha_k = 1.0 - np.exp(-dt / 2.0)
        self.lag_t_hs += alpha_k * (hottest_surface - self.lag_t_hs)
        t_hs_noisy = self.lag_t_hs + self.k_offset
        t_hs_out = np.round(t_hs_noisy / 0.25) * 0.25

        # --- E. DS18B20 Ambient Temperature T_amb (tau = 8s, offset +/-0.4C, step 0.0625C, clip 125C) ---
        self.lag_t_amb += alpha_ds * (t_amb_true - self.lag_t_amb)
        t_amb_noisy = self.lag_t_amb + self.ds_offsets[4]
        t_amb_out = np.clip(np.round(t_amb_noisy / 0.0625) * 0.0625, -40.0, 125.0)

        # --- F. MQ-2 Smoke (tau = 6s, slow random walk drift, clip >= 0) ---
        alpha_smoke = 1.0 - np.exp(-dt / 6.0)
        self.lag_smoke += alpha_smoke * (smoke_true - self.lag_smoke)
        self.smoke_drift += self.rng.normal(0.0, 0.002 * self.noise_scale)
        smoke_noisy = self.lag_smoke + self.smoke_offset + self.smoke_drift + self.rng.normal(0.0, 0.05 * self.noise_scale)
        smoke_out = max(0.0, smoke_noisy)

        # --- G. MQ-7 CO (tau = 15s, valid 90s of every 150s, hold last value otherwise) ---
        alpha_co = 1.0 - np.exp(-dt / 15.0)
        self.lag_co += alpha_co * (co_true - self.lag_co)
        self.co_drift += self.rng.normal(0.0, 0.01 * self.noise_scale)
        
        # Duty cycle: valid for t % 150 < 90, hold otherwise
        cycle_phase = t_sec % 150.0
        if cycle_phase < 90.0:
            co_measured = self.lag_co + self.co_offset + self.co_drift + self.rng.normal(0.0, 0.5 * self.noise_scale)
            self.mq7_last_valid = max(0.0, co_measured)
            co_out = self.mq7_last_valid
        else:
            # Heater purge cycle: hold last valid value
            co_out = self.mq7_last_valid

        # --- H. NDIR CO2 (tau = 45s, noise 10 ppm, range 0-5000 ppm) ---
        alpha_co2 = 1.0 - np.exp(-dt / 45.0)
        self.lag_co2 += alpha_co2 * (co2_true - self.lag_co2)
        co2_noisy = self.lag_co2 + self.co2_offset + self.rng.normal(0.0, 10.0 * self.noise_scale)
        co2_out = float(np.clip(co2_noisy, 0.0, 5000.0))

        # Pack into (14,) array matching CHANNEL_NAMES
        return np.array([
            v_quant[0], v_quant[1], v_quant[2], v_quant[3],
            i_out,
            t_surf_out[0], t_surf_out[1], t_surf_out[2], t_surf_out[3],
            t_hs_out,
            t_amb_out,
            smoke_out,
            co_out,
            co2_out,
        ], dtype=float)

