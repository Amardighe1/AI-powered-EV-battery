"""First-order Thevenin Equivalent Circuit Model (ECM) for 4S2P Li-ion pack.

Pure Python and NumPy.
"""

import numpy as np


def ocv_from_soc(soc: np.ndarray) -> np.ndarray:
    """Empirical Open-Circuit Voltage (OCV) curve for 21700 NMC Li-ion cell.
    
    Valid from 0.0 to 1.3+ SOC with monotonic extrapolation for overcharge.
    """
    # Polynomial + exponential fit typical of NMC chemistry
    # 0% -> ~3.0V, 50% -> ~3.7V, 100% -> 4.2V
    s = np.clip(soc, 0.0, 1.0)
    ocv_nominal = (
        3.0
        + 0.65 * s
        + 0.45 * (s ** 2)
        - 0.50 * (s ** 3)
        + 0.60 * (s ** 4)
        + 0.03 * np.log(np.maximum(s, 1e-4))
        - 0.08 * np.exp(-35.0 * np.maximum(s, 0.0))
    )
    # Extrapolation for overcharge (soc > 1.0)
    overcharge_delta = np.maximum(0.0, soc - 1.0)
    ocv_overcharge = 4.20 + 2.5 * overcharge_delta + 1.2 * (overcharge_delta ** 2)
    
    return np.where(soc > 1.0, ocv_overcharge, ocv_nominal)


class PackECM:
    """4S2P Pack ECM with 4 cell groups in series."""

    def __init__(
        self,
        q_nom_ah: np.ndarray = None,
        r0_ohm: np.ndarray = None,
        r1_ohm: np.ndarray = None,
        c1_farad: np.ndarray = None,
        initial_soc: np.ndarray = None,
    ):
        # 4 groups in series
        if q_nom_ah is None:
            self.q_nom_as = np.full(4, 5.0 * 3600.0)  # 5 Ah in Ampere-seconds
        else:
            self.q_nom_as = q_nom_ah * 3600.0

        self.r0 = np.full(4, 0.015) if r0_ohm is None else np.array(r0_ohm, dtype=float)
        self.r1 = np.full(4, 0.008) if r1_ohm is None else np.array(r1_ohm, dtype=float)
        self.c1 = np.full(4, 3750.0) if c1_farad is None else np.array(c1_farad, dtype=float)

        self.soc = np.full(4, 0.8) if initial_soc is None else np.array(initial_soc, dtype=float)
        self.v_rc = np.zeros(4, dtype=float)
        
        # Internal short resistances (inf by default)
        self.r_short = np.full(4, np.inf, dtype=float)

    def step(self, pack_current: float, dt: float, t_core: np.ndarray) -> dict:
        """Advance electrical model by dt seconds.
        
        pack_current > 0: discharge, pack_current < 0: charge
        """
        # Temperature dependence of R0 (Arrhenius)
        # R0 increases at low temps, decreases slightly at high temps
        t_ref_k = 298.15
        t_k = np.clip(t_core + 273.15, 200.0, 1300.0)
        e_a_r = 15000.0  # J/mol
        r_gas = 8.31446
        r0_temp_factor = np.exp((e_a_r / r_gas) * (1.0 / t_k - 1.0 / t_ref_k))
        # Clamp temperature factor so resistance doesn't vanish or blow up
        r0_temp_factor = np.clip(r0_temp_factor, 0.4, 3.0)
        
        # Lithium plating / swelling resistance growth at severe overcharge (soc > 1.05)
        overcharge_r_factor = 1.0 + 3.0 * np.maximum(0.0, self.soc - 1.05)
        r0_eff = self.r0 * r0_temp_factor * overcharge_r_factor

        # Cell group current includes short circuit current if present
        # V_terminal = OCV - I_tot * R0 - V_RC
        # where I_tot = pack_current + I_short
        # I_short = V_terminal / R_short
        # Solving for V_terminal:
        # V_term = (OCV - V_RC - pack_current * R0_eff) / (1 + R0_eff / R_short)
        ocv = ocv_from_soc(self.soc)
        v_ocv_rc = ocv - self.v_rc - pack_current * r0_eff
        denom = 1.0 + (r0_eff / self.r_short)
        v_terminal = v_ocv_rc / denom
        
        # Individual short circuit currents
        i_short = np.where(np.isneginf(self.r_short) | np.isinf(self.r_short), 0.0, v_terminal / self.r_short)
        i_group = pack_current + i_short

        # Update RC polarization voltage: dV_rc/dt = -V_rc / (R1 * C1) + I_group / C1
        tau_rc = self.r1 * self.c1
        decay = np.exp(-dt / tau_rc)
        self.v_rc = self.v_rc * decay + self.r1 * i_group * (1.0 - decay)

        # Update SOC (Coulomb counting)
        self.soc -= (i_group * dt) / self.q_nom_as

        # Joule heat: I_group^2 * R0_eff + V_rc^2 / R1
        q_joule = (i_group ** 2) * r0_eff + (self.v_rc ** 2) / np.maximum(self.r1, 1e-6)
        
        # Internal short Joule heat: V_terminal * I_short
        q_short = v_terminal * i_short

        # Overcharge side-reaction heat:
        # Up to 1.6 V x charge current, growing beyond 100% SOC
        charge_current = np.maximum(0.0, -i_group)  # positive when charging
        v_side = 1.6 * np.clip((self.soc - 1.0) / 0.08, 0.0, 1.0)
        q_overcharge = v_side * charge_current

        return {
            "v_terminal": v_terminal,
            "soc": self.soc.copy(),
            "i_group": i_group,
            "q_joule": q_joule,
            "q_short": q_short,
            "q_overcharge": q_overcharge,
            "v_side": v_side,
        }

