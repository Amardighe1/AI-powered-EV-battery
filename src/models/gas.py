"""Enclosure gas mass balance model for CO, CO2, and Smoke.

Includes:
 - Early overcharge gas generation (CO2 first, then CO) ramping with side-reaction rate
 - Reaction 1 (SEI) and Reaction 2 (Main decomposition) gas generation (CO2 then CO)
 - Short venting bursts during rapid self-heating / runaway
 - External gas injection for interference scenarios
 - Ambient air exchange (1/400 s^-1)
"""

import numpy as np


class EnclosureGas:
    """Enclosure gas balance with air exchange, overcharge gas, decomposition gas, and venting bursts."""

    def __init__(
        self,
        rng: np.random.RandomState = None,
        co2_bg: float = 420.0,
        air_exchange_rate: float = 1.0 / 400.0,  # 1/400 1/s
    ):
        if rng is None:
            rng = np.random.RandomState(42)

        self.co2_bg = co2_bg
        self.co_bg = 0.0
        self.smoke_bg = 0.0
        self.lambda_vent = air_exchange_rate

        # Randomized overcharge gas coefficients
        self.k_ovc_co2 = rng.uniform(25.0, 45.0)      # ppm/s at full side-reaction rate
        self.k_ovc_co = rng.uniform(10.0, 22.0)       # ppm/s at severe overcharge

        # Randomized chemical decomposition gas coefficients (SEI and Main)
        # SEI generates CO2 then CO; Main generates heavy CO2 and CO
        self.c_sei_co2 = rng.uniform(1000.0, 2000.0)  # ppm per unit alpha1 rate
        self.c_sei_co = rng.uniform(120.0, 300.0)     # ppm per unit alpha1 rate
        self.c_main_co2 = rng.uniform(3000.0, 6000.0) # ppm per unit alpha2 rate
        self.c_main_co = rng.uniform(1000.0, 2500.0)  # ppm per unit alpha2 rate

        # Venting burst release
        self.k_vent_co2 = rng.uniform(2000.0, 3500.0) # ppm total burst
        self.k_vent_co = rng.uniform(500.0, 1000.0)   # ppm total burst
        self.k_vent_smoke = rng.uniform(25.0, 50.0)   # %/m total burst

        # Current concentrations
        self.co2 = self.co2_bg
        self.co = self.co_bg
        self.smoke = self.smoke_bg

        # Venting burst tracking
        self.vented = np.zeros(4, dtype=bool)
        self.vent_timer = np.zeros(4, dtype=float)

    def step(
        self,
        dt: float,
        soc: np.ndarray,
        v_side: np.ndarray,
        i_group: np.ndarray,
        soc_gas_onset: float,
        t_core: np.ndarray,
        k1_sei: np.ndarray,
        k2_main: np.ndarray,
        self_heating_rate: np.ndarray,
        ext_co2: float = 0.0,
        ext_co: float = 0.0,
        ext_smoke: float = 0.0,
    ) -> dict:
        """Advance enclosure gas concentrations by dt seconds."""
        gen_co2 = 0.0
        gen_co = 0.0
        gen_smoke = 0.0
        
        ovc_gas_rate_co2 = 0.0
        ovc_gas_rate_co = 0.0

        # 1. Overcharge early gas generation (ramped from side-reaction rate)
        # Gas onset SOC is randomized per run (103% - 110%)
        for i in range(4):
            i_chg = max(0.0, -i_group[i])  # positive charging current
            if soc[i] > soc_gas_onset and i_chg > 0.0:
                # Severity factor ramping smoothly beyond onset
                sev = min(1.0, (soc[i] - soc_gas_onset) / 0.06)
                # Gas rate scales with side-reaction voltage and charge current
                side_rate_factor = (v_side[i] / 1.6) * (i_chg / 5.0)
                rate_co2 = self.k_ovc_co2 * sev * max(0.2, side_rate_factor)
                ovc_gas_rate_co2 += rate_co2
                gen_co2 += rate_co2

                # CO starts slightly after CO2 onset (onset + 0.015)
                co_onset = soc_gas_onset + 0.015
                if soc[i] > co_onset:
                    sev_co = min(1.0, (soc[i] - co_onset) / 0.06)
                    rate_co = self.k_ovc_co * sev_co * max(0.2, side_rate_factor)
                    ovc_gas_rate_co += rate_co
                    gen_co += rate_co

            # 2. SEI and Main decomposition gas (CO2 then CO)
            # Reaction rates k1 (SEI) and k2 (Main) generate early off-gas
            decomp_co2 = self.c_sei_co2 * k1_sei[i] + self.c_main_co2 * k2_main[i]
            decomp_co = self.c_sei_co * k1_sei[i] + self.c_main_co * k2_main[i]
            gen_co2 += decomp_co2
            gen_co += decomp_co

            # 3. Venting bursts: triggered when cell reaches 130 C or runaway self-heating > 0.5 C/s
            if not self.vented[i]:
                if t_core[i] >= 130.0 or self_heating_rate[i] >= 0.5:
                    self.vented[i] = True
                    self.vent_timer[i] = 10.0  # burst lasts 10 s

            if self.vent_timer[i] > 0.0:
                burst_frac = dt / 10.0
                gen_co2 += (self.k_vent_co2 / 4.0) * burst_frac
                gen_co += (self.k_vent_co / 4.0) * burst_frac
                gen_smoke += (self.k_vent_smoke / 4.0) * burst_frac
                self.vent_timer[i] = max(0.0, self.vent_timer[i] - dt)

        # 4. External gas sources
        gen_co2 += ext_co2
        gen_co += ext_co
        gen_smoke += ext_smoke

        # Mass balance ODEs: dC/dt = gen - lambda * (C - C_bg)
        decay = np.exp(-self.lambda_vent * dt)
        self.co2 = self.co2_bg + (self.co2 - self.co2_bg) * decay + gen_co2 * dt
        self.co = self.co_bg + (self.co - self.co_bg) * decay + gen_co * dt
        self.smoke = self.smoke_bg + (self.smoke - self.smoke_bg) * decay + gen_smoke * dt

        # Realistic non-negative bounding
        self.co2 = max(0.0, self.co2)
        self.co = max(0.0, self.co)
        self.smoke = max(0.0, self.smoke)

        # True gas-generation condition from overcharge:
        has_true_ovc_gas = (ovc_gas_rate_co2 > 0.1)

        return {
            "co2": self.co2,
            "co": self.co,
            "smoke": self.smoke,
            "ovc_gas_rate_co2": ovc_gas_rate_co2,
            "ovc_gas_rate_co": ovc_gas_rate_co,
            "has_true_ovc_gas": has_true_ovc_gas,
        }
