"""Two-node thermal model with Arrhenius decomposition kinetics for 4S2P pack.

Pure Python and NumPy.
"""

import numpy as np


class PackThermal:
    """Two-node thermal model for 4 cell groups with Arrhenius decomposition."""

    def __init__(
        self,
        t_amb: float = 25.0,
        fan_failed: bool = False,
        initial_temp: float = None,
    ):
        self.t_amb = float(t_amb)
        init_t = float(t_amb) if initial_temp is None else float(initial_temp)
        
        # Thermal capacities and resistances per group
        self.c_c = 70.0    # J/K (Core)
        self.c_s = 22.0    # J/K (Surface)
        self.r_cs = 1.2    # K/W (Core to surface)
        self.ha = 0.06 if fan_failed else 0.24  # W/K (Surface to ambient)
        self.k_neigh = 0.08  # W/K (Neighbor conduction between adjacent surface nodes)

        # Arrhenius parameters
        self.r_gas = 8.314462618  # J/(mol*K)
        # Reaction 1: SEI decomposition
        self.a1 = 1.667e15        # 1/s
        self.ea1 = 135.08e3       # J/mol
        self.h1 = 6000.0          # J
        # Reaction 2: Main decomposition
        self.a2 = 6.0e15          # 1/s
        self.ea2 = 150.0e3        # J/mol
        self.h2 = 40000.0         # J

        # State vectors (4 cell groups)
        self.t_core = np.full(4, init_t, dtype=float)
        self.t_surf = np.full(4, init_t, dtype=float)
        self.alpha1 = np.zeros(4, dtype=float)
        self.alpha2 = np.zeros(4, dtype=float)

    def set_fan_failed(self, failed: bool = True):
        """Update convective heat transfer coefficient when coolant fan fails."""
        self.ha = 0.06 if failed else 0.24

    def compute_rates(self, t_core: np.ndarray, alpha1: np.ndarray, alpha2: np.ndarray):
        """Compute chemical decomposition rates and heat generation."""
        # Temperature in Kelvin clamped to avoid zero or negative
        t_k = np.clip(t_core + 273.15, 200.0, 1500.0)

        # Arrhenius exponent clipping to prevent overflow / underflow
        exp_arg1 = -self.ea1 / (self.r_gas * t_k)
        exp_arg2 = -self.ea2 / (self.r_gas * t_k)
        exp_arg1 = np.clip(exp_arg1, -100.0, 50.0)
        exp_arg2 = np.clip(exp_arg2, -100.0, 50.0)

        rem1 = np.clip(1.0 - alpha1, 0.0, 1.0)
        rem2 = np.clip(1.0 - alpha2, 0.0, 1.0)

        k1 = self.a1 * np.exp(exp_arg1) * rem1
        k2 = self.a2 * np.exp(exp_arg2) * rem2

        q_chem1 = self.h1 * k1
        q_chem2 = self.h2 * k2
        q_chem = q_chem1 + q_chem2
        self_heating_rate = q_chem / self.c_c  # C/s

        return k1, k2, q_chem1, q_chem2, q_chem, self_heating_rate

    def step(
        self,
        q_electrical: np.ndarray,
        dt: float,
        t_amb: float = None,
    ) -> dict:
        """Advance thermal model by dt with sub-stepping for stability.
        
        Sub-stepping: 0.5 s nominal, 0.05 s if any T_core > 110 C.
        """
        if t_amb is not None:
            self.t_amb = float(t_amb)

        # Determine sub-step dt
        max_core = np.max(self.t_core)
        sub_dt = 0.05 if max_core > 110.0 else min(0.5, dt)
        num_substeps = max(1, int(np.ceil(dt / sub_dt)))
        actual_sub_dt = dt / num_substeps

        for _ in range(num_substeps):
            k1, k2, q1, q2, q_chem, sh_rate = self.compute_rates(
                self.t_core, self.alpha1, self.alpha2
            )
            q_gen_core = q_electrical + q_chem

            # Core conduction to surface
            q_core_to_surf = (self.t_core - self.t_surf) / self.r_cs

            # Neighbor conduction between adjacent groups: 0-1, 1-2, 2-3
            q_neigh = np.zeros(4, dtype=float)
            q_neigh[0] = self.k_neigh * (self.t_surf[1] - self.t_surf[0])
            q_neigh[1] = self.k_neigh * (self.t_surf[0] - self.t_surf[1]) + self.k_neigh * (self.t_surf[2] - self.t_surf[1])
            q_neigh[2] = self.k_neigh * (self.t_surf[1] - self.t_surf[2]) + self.k_neigh * (self.t_surf[3] - self.t_surf[2])
            q_neigh[3] = self.k_neigh * (self.t_surf[2] - self.t_surf[3])

            # Surface convection to ambient
            q_surf_to_amb = self.ha * (self.t_surf - self.t_amb)

            # Derivatives
            dt_core = (q_gen_core - q_core_to_surf) / self.c_c
            dt_surf = (q_core_to_surf - q_surf_to_amb + q_neigh) / self.c_s

            # Update states
            self.t_core += dt_core * actual_sub_dt
            self.t_surf += dt_surf * actual_sub_dt
            self.alpha1 += k1 * actual_sub_dt
            self.alpha2 += k2 * actual_sub_dt

            # Clamping to valid physical ranges and avoiding overflow
            self.t_core = np.clip(self.t_core, -40.0, 1000.0)
            self.t_surf = np.clip(self.t_surf, -40.0, 1000.0)
            self.alpha1 = np.clip(self.alpha1, 0.0, 1.0)
            self.alpha2 = np.clip(self.alpha2, 0.0, 1.0)

        # Re-evaluate final instantaneous rates
        k1, k2, q1, q2, q_chem, self_heating_rate = self.compute_rates(
            self.t_core, self.alpha1, self.alpha2
        )

        return {
            "t_core": self.t_core.copy(),
            "t_surf": self.t_surf.copy(),
            "alpha1": self.alpha1.copy(),
            "alpha2": self.alpha2.copy(),
            "k1": k1,
            "k2": k2,
            "q_chem": q_chem,
            "q_chem1": q1,
            "q_chem2": q2,
            "self_heating_rate": self_heating_rate,
        }
