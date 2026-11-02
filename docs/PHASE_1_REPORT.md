# Phase 1.1: Simulator Enhancements & Verification Evidence

## 1. Explanation of the Voltage Chatter in Overcharge after ~530 s

In the initial Phase 1 simulation of Scenario 8 (`overcharge_runaway`), high-frequency voltage oscillation ("chatter") appeared in the cell terminal voltages between $t \approx 530\text{ s}$ and $t \approx 650\text{ s}$.

### Root Cause Analysis:
1. **Unlatched State Transition**: When cell group 1 reached $SoC > 1.125$, the code injected an internal micro-short representing separator puncture by lithium dendrites ($R_{short} \approx 0.35\,\Omega$).
2. **Current Reversal & Depletion**: The active charging current was $I_{pack} = -6.5\text{ A}$ (charge). However, the internal short drew $I_{short} = V_{term} / R_{short} \approx 4.5\text{ V} / 0.35\,\Omega \approx +12.8\text{ A}$ (discharge). The net cell group current became $I_{group} = -6.5 + 12.8 = +6.3\text{ A}$ (net discharging).
3. **Limit-Cycle Toggling**: Because the net current was discharging, at timestep $k+1$, the cell's $SoC$ dropped slightly below $1.125$ (e.g. $1.1249$). In the original code, `r_short_profiles` reset `ecm.r_short` to $\infty$ at every step. Because $SoC < 1.125$ and $T_{core}$ was not yet $> 110^\circ\text{C}$, the short deactivated. The cell charged at $-6.5\text{ A}$, jumping the voltage by $+0.2\text{ V}$. In timestep $k+2$, $SoC$ crossed $1.125$ again, reactivating the short and dropping the voltage by $-0.2\text{ V}$.
4. **Physical Reality & Fix**: In real lithium-ion cells, mechanical separator puncture by lithium dendrites is an **irreversible physical damage mode**—the separator does not self-heal when $SoC$ drops $0.01\%$. In Phase 1.1, the internal short trigger is permanently latched (`has_dendrite_short[i] = True`). Once triggered, $R_{short}$ stays permanently active. As verified in the Phase 1.1 plots, the voltage curve is now completely smooth with zero numerical chatter.

---

## 2. Benchmark Report: 20 Seeds per Scenario (220 Total Runs)

| Scenario | Mean Dur (s) | Label 0 | Label 1 | Label 2 | Label 3 | Label 4 | Label 5 | Runaway Rate | Runaway Onset (min/med/max) | Class 4 Window (min/med/max) | Max Core Temp (°C) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **normal_drive** | 1799 | 1800 | 0 | 0 | 0 | 0 | 0 | 0/20 (0%) | N/A | N/A | 10.6 / 18.6 / 41.4 °C |
| **normal_charge** | 1799 | 1800 | 0 | 0 | 0 | 0 | 0 | 0/20 (0%) | N/A | N/A | 7.8 / 17.1 / 40.2 °C |
| **hot_ambient** | 1799 | 902 | 898 | 0 | 0 | 0 | 0 | 0/20 (0%) | N/A | N/A | 46.6 / 48.1 / 50.2 °C |
| **external_gas** | 1799 | 1143 | 657 | 0 | 0 | 0 | 0 | 0/20 (0%) | N/A | N/A | 8.6 / 17.2 / 40.5 °C |
| **overcurrent** | 1799 | 1753 | 0 | 47 | 0 | 0 | 0 | 0/20 (0%) | N/A | N/A | 11.3 / 19.8 / 42.2 °C |
| **over_discharge** | 1799 | 1705 | 0 | 95 | 0 | 0 | 0 | 0/20 (0%) | N/A | N/A | 7.8 / 16.5 / 40.2 °C |
| **cooling_failure** | 1799 | 205 | 1367 | 0 | 228 | 0 | 0 | 0/20 (0%) | N/A | N/A | 59.6 / 61.3 / 63.7 °C |
| **overcharge_runaway** | 707 | 44 | 0 | 290 | 0 | 253 | 121 | 20/20 (100%) | 519 / 576 / 674 s | **106 / 254 / 381 s** | 996.2 / 999.5 / 1000.0 °C |
| **internal_short_runaway** | 435 | 113 | 57 | 0 | 50 | 95 | 121 | 20/20 (100%) | 230 / 310 / 412 s | **65 / 94 / 137 s** | 883.3 / 997.3 / 1000.0 °C |
| **overcharge_aborted** | 1799 | 199 | 0 | 997 | 0 | 604 | 0 | 0/20 (0%) | N/A | N/A | 13.2 / 26.4 / 57.0 °C |
| **soft_internal_short** | 1799 | 708 | 1092 | 0 | 0 | 0 | 0 | 0/20 (0%) | N/A | N/A | 20.7 / 29.3 / 57.3 °C |

> **Key Observations**:
> - **Lead-Time Window**:
>   - `overcharge_runaway`: min **106 s**, median **254 s**, max **381 s** (1.8 to 6.4 minutes).
>   - `internal_short_runaway`: min **65 s**, median **94 s**, max **137 s** (1.1 to 2.3 minutes).
>   - **100% of runaway runs fall strictly inside [60 s, 900 s]**.
> - **Class Transitions**:
>   - `hot_ambient`: strictly starts in Class 0 (mean 902 s), transitions to Class 1 as pack warms in hot ambient.
>   - `cooling_failure`: strictly starts in Class 0 (mean 205 s), transitions to Class 1 (mean 1367 s), then transitions to Class 3 (mean 228 s, max core temp 59.6–63.7°C).
> - **Overcurrent Duration**: Mean duration in Class 2 is **47 seconds** ($\ge 40\text{ s}$ guaranteed).
> - **Exclusivity**: Class 5 occurs in 100% of scenarios 8 and 9, and **0%** in scenarios 1–7, 10, 11.

---

## 3. Raw Pytest Output

```
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.0.2, pluggy-1.6.0 -- C:\Users\amard\AppData\Local\Programs\Python\Python310\python.exe
cachedir: .pytest_cache
rootdir: D:\AI-powered EV battery
plugins: anyio-4.12.1, cov-7.0.0, typeguard-4.4.4
collecting ... collected 12 items

tests/test_simulator.py::test_scenario_names_completeness PASSED         [  8%]
tests/test_simulator.py::test_runaway_exclusivity_and_class4_precursor_20_seeds PASSED [ 16%]
tests/test_simulator.py::test_bitwise_reproducibility PASSED             [ 25%]
tests/test_simulator.py::test_no_nan_and_physical_bounds PASSED          [ 33%]
tests/test_simulator.py::test_class_5_latched_and_class_4_timing PASSED  [ 41%]
tests/test_simulator.py::test_labels_independent_of_sensor_noise PASSED  [ 50%]
tests/test_simulator.py::test_sensor_lag_time_constants PASSED           [ 58%]
tests/test_simulator.py::test_sensor_noise_variance PASSED               [ 66%]
tests/test_simulator.py::test_t_hs_derived_from_surface_not_core PASSED  [ 75%]
tests/test_simulator.py::test_label_priority_order PASSED                [ 83%]
tests/test_simulator.py::test_sensor_quantisation_and_clipping PASSED    [ 91%]
tests/test_simulator.py::test_sensor_mq7_duty_cycle_hold PASSED          [100%]

======================= 12 passed in 108.36s (0:01:48) ========================
```

---

## 4. Parameter Tuning Log

| Parameter | Initial Value | Final Value | Reason for Tuning |
|---|---|---|---|
| **Dendrite Short Latching** | Unlatched (re-evaluated each step) | **Permanently Latched** (`has_dendrite_short = True`) | In Phase 1, when $I_{short}$ discharged the cell, SOC momentarily fell below $1.125$, resetting $R_{short} \rightarrow \infty$, causing a non-physical limit-cycle chatter at 530 s. Physical separator puncture is irreversible; latching eliminates chatter. |
| **Dendrite Trigger SOC** | Fixed $1.125$ | **Randomized $U(1.11, 1.15)$** | Matches physical variability in separator puncture threshold across cell manufacturing lots. |
| **Dendrite Short Resistance** | Fixed $0.35\,\Omega$ | **Randomized $U(0.18, 0.28)\,\Omega$** | Guarantees consistent runaway transition within target Class 4 window across 100% of randomized seeds. |
| **Overcharge Gas Onset SOC** | Fixed $1.08$ | **Randomized $U(1.03, 1.10)$** | Literature shows early electrolyte oxidation off-gassing begins variably between 103% and 110% SOC. |
| **SEI / Decomp. Gas Coefficients** | $c_{sei, CO2} = 300\text{ ppm}$, $c_{sei, CO} = 0\text{ ppm}$ | **$c_{sei, CO2} = 1000\text{--}2000$, $c_{sei, CO} = 120\text{--}300\text{ ppm}$** | Allows `internal_short_runaway` to exhibit noticeable CO2 and CO emissions before runaway onset. |
| **Overcurrent Pulse Duration** | 8 s pulse | **48–55 s sustained pulse** ($16.0\text{--}19.0\text{ A}$) | Satisfies requirement that sustained 15.5–19.5 A current must give $\ge 40\text{ s}$ of Class 2 label. |
| **Non-Hot Ambient Temperature** | $20.0\text{--}26.0^\circ\text{C}$ | **$5.0\text{--}40.0^\circ\text{C}$** | Covers the complete operational non-hot thermal envelope per user requirement. |
| **Cooling Failure Initial State** | Pre-warmed at $55^\circ\text{C}$ (started in Class 1) | **$T_{init} = T_{amb} \in [38.5, 42.5]^\circ\text{C}$**, fan initially active | Ensures `cooling_failure` strictly starts in Class 0 ($T < 45^\circ\text{C}$), then fan fails at $t=50\text{--}100\text{ s}$ and transitions to Class 1, then Class 3. |
| **Hot Ambient Initial State** | $T_{init} = T_{amb} = 48^\circ\text{C}$ (started in Class 1) | **$T_{init} \in [20, 25]^\circ\text{C}$**, $T_{amb} \in [46, 50]^\circ\text{C}$ | Ensures `hot_ambient` strictly starts in Class 0, warming over ~900s to transition into Class 1. |
| **External Gas Peak CO** | $\sim 15\text{--}20\text{ ppm}$ | **$50\text{--}220\text{ ppm}$** ($30\text{--}300\text{ ppm}$ range) | Delivers realistic external exhaust plume spike triggering Class 1 without battery damage. |
| **Overcharge Aborted Cutoff** | Fixed $102\text{--}106.5\%$ | **Randomized $100\text{--}112\%$** | Allows BMS to intervene across broad overcharge window, generating brief precursor gas when cutoff exceeds onset. |

---

## 5. Diagnostic Plots (Scenarios 1, 4, 8, 9)

````carousel
![Scenario 1: Normal Drive](C:/Users/amard/.gemini/antigravity/brain/c89fc909-89ad-46fd-90e4-bf78aa2764c4/scenario_normal_drive.png)
<!-- slide -->
![Scenario 4: External Gas Interference](C:/Users/amard/.gemini/antigravity/brain/c89fc909-89ad-46fd-90e4-bf78aa2764c4/scenario_external_gas.png)
<!-- slide -->
![Scenario 8: Overcharge Runaway](C:/Users/amard/.gemini/antigravity/brain/c89fc909-89ad-46fd-90e4-bf78aa2764c4/scenario_overcharge_runaway.png)
<!-- slide -->
![Scenario 9: Internal Short Runaway](C:/Users/amard/.gemini/antigravity/brain/c89fc909-89ad-46fd-90e4-bf78aa2764c4/scenario_internal_short_runaway.png)
````
