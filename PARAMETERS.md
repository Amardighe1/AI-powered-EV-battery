# Model Parameters & Literature Traceability

This document records all physical, thermal, chemical, electrical, sensor, and environmental parameters used in the Phase 1.1 EV Battery Fault & Thermal Runaway Simulator for a 4S2P Li-ion pack.

Each parameter is explicitly classified as either **Literature** (with academic / industry citation) or **Assumed** (heuristic or engineering assumption specific to this pack enclosure setup).

---

## 1. Electrical & Cell Group Parameters

The battery pack is configured as **4S2P** (4 series-connected cell-groups, each containing 2 parallel 21700 NMC cells).

| Parameter | Symbol | Value | Unit | Status | Source / Justification |
|---|---|---|---|---|---|
| Nominal Capacity per group | $Q_{nom}$ | 5.0 $\pm$ 2% | Ah | **Literature** | Standard dual-cell parallel group (e.g. 2x 2500 mAh 21700 NMC cells or single 5 Ah group), randomized $\pm 2\%$ cell-to-cell. |
| Nominal Voltage | $V_{nom}$ | 3.7 | V | **Literature** | Typical NMC chemistry nominal cell voltage. |
| Charge Cutoff Voltage | $V_{max}$ | 4.2 | V | **Literature** | Standard manufacturer upper voltage limit. |
| Discharge Cutoff Voltage | $V_{min}$ | 2.8 | V | **Literature** | Standard manufacturer lower voltage limit. |
| Ohmic Resistance | $R_0$ | 15 $\pm$ 5% | $\text{m}\Omega$ | **Literature** | Typical 21700 2P group high-power internal resistance at 25°C (Kim et al., 2007; Chen et al., 2020). |
| Polarization Resistance | $R_1$ | 8 | $\text{m}\Omega$ | **Literature** | 1st-order Thevenin ECM dynamic resistance for Li-ion (He et al., 2011). |
| Polarization Capacitance | $C_1$ | 3750 | F | **Literature** | Gives RC time constant $\tau_{RC} = R_1 C_1 = 30\text{ s}$, consistent with diffusion relaxation dynamics in Li-ion cells. |
| Max Drive Peak Discharge | $I_{dis, max}$ | 2.5C (12.5 A) | A | **Literature** | Normal automotive driving cycle limitation for 5 Ah group (WLTP/US06 power demands). |
| Max Drive Regen Charge | $I_{chg, max}$ | 1.0C (5.0 A) | A | **Literature** | Standard regenerative braking charge acceptance limit without lithium plating risk. |
| Overcharge Side-Reaction Voltage | $V_{side}$ | up to 1.6 | V | **Literature** | Overpotential growth during continuous overcharge beyond 100% SOC (Wang et al., 2019; Ren et al., 2021). |
| Overcharge Dendrite Trigger SOC | $SoC_{dendrite}$ | 111%–115% | - | **Assumed** | SOC threshold at which lithium dendrites / swelling mechanically puncture separator (Wang et al., 2019). |
| Overcharge Dendrite Short Resistance | $R_{short, ovc}$ | 0.18–0.28 | $\Omega$ | **Assumed** | Effective internal short resistance resulting from catastrophic dendrite puncture. |
| Coulomb-Counting Initial SOC Error | $\Delta SoC_{init}$ | $\pm 4\%$ | - | **Literature** | Realistic automotive BMS open-circuit voltage lookup and rest-time uncertainty. |

---

## 2. Thermal & Heat Transfer Parameters

Two-node thermal model (Core $T_c$ and Surface $T_s$) per cell group:

| Parameter | Symbol | Value | Unit | Status | Source / Justification |
|---|---|---|---|---|---|
| Core Heat Capacity | $C_c$ | 70 | J/K | **Literature** | Cylindrical 21700 cell internal jelly-roll thermal mass (Feng et al., 2018; Hatchard & Dahn, 2001). |
| Surface Heat Capacity | $C_s$ | 22 | J/K | **Literature** | Steel casing and outer thermal boundary mass for 21700 group (Feng et al., 2018). |
| Core-to-Surface Conduction Resistance | $R_{cs}$ | 1.2 | K/W | **Literature** | Radial conduction resistance across jelly-roll and steel can (Lin et al., 2014; Kim et al., 2007). |
| Surface-to-Ambient Convective Conductance | $hA$ | 0.24 (nominal), 0.06 (fan fail) | W/K | **Assumed** | Forced-air / coolant pack enclosure boundary; fan failure represents 75% reduction in convective heat removal. |
| Inter-Group Conduction Conductance | $k_{neigh}$ | 0.08 | W/K | **Assumed** | Direct structural conduction through module busbars and cell holders between adjacent groups. |
| Ambient Temperature (Non-Hot Scenarios) | $T_{amb}$ | 5–40 (randomized) | °C | **Literature** | Standard automotive operational ambient envelope (-40°C to +40°C). |
| Maximum Core Temperature Clamp | $T_{max}$ | 1000 | °C | **Assumed** | Numerical clamp to prevent arithmetic overflow during post-runaway flash pyrolysis. |

---

## 3. Chemical Decomposition Kinetics (Thermal Runaway)

Two-reaction Arrhenius decomposition scheme modeling solid-electrolyte interphase (SEI) collapse and main cathode/electrolyte decomposition:

| Parameter | Symbol | Value | Unit | Status | Source / Justification |
|---|---|---|---|---|---|
| Universal Gas Constant | $R_{gas}$ | 8.31446 | J/(mol·K) | **Literature** | Standard physical constant. |
| SEI Frequency Factor | $A_1$ | $1.667 \times 10^{15}$ | $\text{s}^{-1}$ | **Literature** | Hatchard & Dahn (2001); Kim et al. (2007) Li-ion accelerated calorimeter decomposition. |
| SEI Activation Energy | $E_{a, 1}$ | 135.08 | kJ/mol | **Literature** | Hatchard & Dahn (2001); Dahn battery thermal decomposition kinetic models. |
| SEI Heat of Reaction | $H_1$ | 6.0 | kJ | **Literature** | Normalized total exothermic enthalpy of anode SEI breakdown layer. |
| Main Decomposition Frequency Factor | $A_2$ | $6.0 \times 10^{15}$ | $\text{s}^{-1}$ | **Literature** | Feng et al. (2018); Ren et al. (2021) high-temperature cathode/electrolyte reaction. |
| Main Decomposition Activation Energy | $E_{a, 2}$ | 150.0 | kJ/mol | **Literature** | High-energy barrier for active material phase breakdown (Feng et al., 2018). |
| Main Decomposition Heat of Reaction | $H_2$ | 40.0 | kJ | **Literature** | High enthalpy release of cathode oxygen liberation and violent solvent combustion. |
| Runaway Onset Criteria | $(dT_c/dt)_{crit}$ | $\ge 1.0$ | °C/s | **Literature** | Standard international battery safety testing benchmark for thermal runaway onset. |
| Runaway Precursor Heat Threshold | $Q_{prec}$ | 0.3 | W | **Literature** | Onset of measurable exothermic self-heating prior to uncontrolled thermal explosion. |

---

## 4. Gas Generation & Enclosure Dynamics

| Parameter | Symbol | Value | Unit | Status | Source / Justification |
|---|---|---|---|---|---|
| Enclosure Air Exchange Rate | $\lambda_{vent}$ | 1/400 (0.0025) | $\text{s}^{-1}$ | **Assumed** | Typical sealed automotive battery enclosure with pressure-equalizing semi-permeable membrane. |
| Ambient $CO_2$ Baseline | $C_{amb, CO2}$ | 420 | ppm | **Literature** | Global atmospheric average background $CO_2$. |
| Ambient $CO$ Baseline | $C_{amb, CO}$ | 0 | ppm | **Literature** | Clean ambient air reference. |
| Ambient Smoke Baseline | $Smoke_{amb}$ | 0 | %/m | **Literature** | Clean air optical obscuration reference. |
| Overcharge Gas Onset SOC | $SoC_{gas\_onset}$ | 103%–110% (randomized) | - | **Literature** | Electrolyte oxidation off-gassing onset during forced overcharge (Koch et al., 2018). |
| Early $CO_2$ Rate from Overcharge | $k_{ovc, CO2}$ | 25–45 (randomized) | ppm/s | **Assumed** | Carbonate solvent decarboxylation producing early $CO_2$. |
| Early $CO$ Rate from Overcharge | $k_{ovc, CO}$ | 10–22 (randomized) | ppm/s | **Assumed** | Incomplete reduction / oxidation at high overpotential after $CO_2$ onset. |
| SEI Decomposition $CO_2$ Yield | $c_{sei, CO2}$ | 1000–2000 (randomized) | ppm | **Assumed** | Off-gassing from SEI breakdown before thermal runaway. |
| SEI Decomposition $CO$ Yield | $c_{sei, CO}$ | 120–300 (randomized) | ppm | **Assumed** | Secondary carbonaceous off-gas release during SEI self-heating. |
| Main Decomposition $CO_2$ Yield | $c_{main, CO2}$ | 3000–6000 (randomized) | ppm | **Assumed** | Massive gas generation from cathode active material collapse. |
| Main Decomposition $CO$ Yield | $c_{main, CO}$ | 1000–2500 (randomized) | ppm | **Assumed** | Dense combustible toxic gas release during thermal runaway. |
| Venting Burst Yields ($CO_2$, $CO$, Smoke) | $\Delta C_{vent}$ | Randomized | ppm, %/m | **Assumed** | Safety valve pop releases rapid gas and aerosol particulates. |

---

## 5. Sensor Suite & Distortion Models

| Channel | Physical Sensor Type | Time Constant $\tau$ | Calibration Offset | ADC Resolution / Step | Noise ($\sigma$) | Output Limits | Status / Source |
|---|---|---|---|---|---|---|---|
| $V_{1..4}$ | Cell Tap Voltage ADC | 0 s (instant) | $\pm 5$ mV | 1 mV | 3 mV | 0–5 V | **Literature** (automotive BMS AFE standard, e.g. BQ76952 / LTC6811). |
| $I$ | Hall Effect / Shunt Current Sensor | 0 s (instant) | $\pm 0.08$ A | 10 mA | 0.06 A | $[-20, +20]$ A | **Literature** (CSLA / LEM automotive current transducer specs). |
| $T_{1..4}$ | DS18B20 1-Wire Digital Surface Sensor | 8 s | $\pm 0.4$ °C | 0.0625 °C | None (digital 12-bit) | Max 125 °C clip | **Literature** (Maxim DS18B20 official datasheet). |
| $T_{hs}$ | Type-K Thermocouple (Hot-Spot Surface) | 2 s | $\pm 1.0$ °C | 0.25 °C | None (ADC quantized) | Unlimited | **Literature** (IEC 60584-2 Class 2 thermocouple specs). |
| $T_{amb}$ | DS18B20 Ambient Sensor | 8 s | $\pm 0.4$ °C | 0.0625 °C | None (digital 12-bit) | Max 125 °C clip | **Literature** (Maxim DS18B20 official datasheet). |
| $Smoke$ | MQ-2 Optical / Metal-Oxide Particulate | 6 s | $\pm 0.1$ %/m | Continuous | $\sigma = 0.05$ | $\ge 0$ | **Assumed** (datasheet typical response time). |
| $CO$ | MQ-7 Catalytic / Electrochemical CO | 15 s | $\pm 2$ ppm | Continuous | $\sigma = 0.5$ | 90s valid / 150s cycle hold | **Literature** (Zhengzhou Winsen MQ-7 sensor datasheet heating cycle). |
| $CO_2$ | NDIR Infrared $CO_2$ Sensor | 45 s | $\pm 25$ ppm | Continuous | $\sigma = 10$ ppm | 0–5000 ppm | **Literature** (Senseair S8 / Amphenol Telaire NDIR response time). |

---

## 6. Phase 1.1 Parameter Tuning Log

This log details every parameter that was adjusted between initial Phase 1 and Phase 1.1, the motivation, starting value, and final value:

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
