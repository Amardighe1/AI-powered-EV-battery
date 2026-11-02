# Phase 2 Report: AI-Powered EV Battery Fault and Fire Detection

This report details the execution, methodology, evaluation, and benchmark findings of Phase 2.

---

## 1. Dataset Generation & Class Distribution

390 runs were synthesized using the multi-process batch engine across disjoint random seeds, stratified per scenario with a 60% Train (234 runs), 20% Validation (78 runs), and 20% Test (78 runs) split:
- **Runaway Scenarios (60 runs each)**: 36 train, 12 val, 12 test for `overcharge_runaway` and `internal_short_runaway`.
- **Non-Runaway Scenarios (30 runs each)**: 18 train, 6 val, 6 test across all 9 non-runaway scenarios (`normal_drive`, `normal_charge`, `hot_ambient`, `external_gas`, `overcurrent`, `over_discharge`, `cooling_failure`, `overcharge_aborted`, `soft_internal_short`).

### Window Extraction
- **Train / Validation**: Stride = 5 s (window length = 60 s, 1 Hz).
- **Test**: Stride = 1 s (window length = 60 s, 1 Hz) for true 1 Hz temporal resolution in alarm evaluation.
- **Window Label**: Assigned as the true physical ground-truth label at the last sample of each 60 s window ($t=59$).

### Class Counts per Split

| Class ID | Class Description | Train Count (stride=5s) | Train % | Val Count (stride=5s) | Val % | Test Count (stride=1s) | Test % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | Normal | 34,799 | 54.46% | 12,180 | 57.13% | 59,803 | 56.36% |
| **1** | Abnormal | 14,727 | 23.05% | 5,179 | 24.29% | 22,870 | 21.55% |
| **2** | Electrical Fault | 5,013 | 7.85% | 1,051 | 4.93% | 8,608 | 8.11% |
| **3** | Thermal Fault | 1,639 | 2.56% | 272 | 1.28% | 1,942 | 1.83% |
| **4** | Early Warning | 5,979 | 9.36% | 2,059 | 9.66% | 9,992 | 9.42% |
| **5** | Fire / Critical Runaway | 1,742 | 2.73% | 578 | 2.71% | 2,904 | 2.74% |
| **Total** | **All Classes** | **63,899** | **100.0%** | **21,319** | **100.0%** | **106,119** | **100.0%** |

---

## 2. Dataset Leakage & Data Hygiene Verification

A rigorous audit was enforced across the dataset pipeline to guarantee zero data leakage:
1. **Split by RUN, Not by Window**: Partitioning was performed strictly at the whole-run level. No sliding windows from the same run ever cross between train, validation, and test splits.
2. **Disjoint Seed Partitions**:
   - `train_seeds ∩ val_seeds = ∅`
   - `train_seeds ∩ test_seeds = ∅`
   - `val_seeds ∩ test_seeds = ∅`
   All 390 runs were assigned globally unique seeds based on scenario offset partitioning.
3. **Strict Feature Scaling Boundary**: `StandardScaler` (for the 34 tabular features) and the raw sensor normalization statistics (`channel_mean`, `channel_std` for the 14 raw channels) were computed **strictly on the Train split**. Validation and test matrices were transformed using frozen Train parameters.
4. **Sensor-Only Channel Boundary**: `compute_features(window[60,14])` uses **only** the 14 raw 1 Hz sensor channels (`V1-4, I, T1-4, T_hs, T_amb, Smoke, CO, CO2`). True physical state variables (internal core temperatures, Arrhenius reaction states, true SOC, or chemical self-heating rates) are strictly inaccessible to the features and models.
5. **Causal Running SOC**: The SOC feature is estimated purely from sensor voltage ($V_{cell}$ average over the initial 10 s inverted through an NMC OCV lookup curve), followed by forward Coulomb counting from noisy sensor current $I_{meas}$.

---

## 3. Model Comparison Table

Evaluated on the full Test split (**106,119 windows** at 1 Hz resolution):

| Model | Val Macro-F1 | Test Macro-F1 | Test Accuracy | Test Weighted-F1 | F1 (0) Normal | F1 (1) Abn | F1 (2) Elec | F1 (3) Therm | F1 (4) Early | F1 (5) Fire | Train Time | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Threshold Baseline** | 0.7249 | 0.7646 | 88.24% | 0.8829 | 0.9608 | 0.8915 | 0.7630 | 0.6412 | 0.5879 | 0.7435 | 0.00 s | **0.017 ms** |
| **Balanced Random Forest** | **0.9295** | **0.9228** | **95.76%** | **0.9573** | **0.9896** | **0.9664** | **0.8543** | **0.8989** | **0.8319** | **0.9955** | **2.22 s** | 22.28 ms |
| **1D-CNN (CPU)** | 0.8619 | 0.8905 | 91.65% | 0.9177 | 0.9564 | 0.9019 | 0.8083 | 0.8809 | 0.8015 | 0.9941 | 90.63 s | 0.59 ms |

> [!IMPORTANT]
> **Model Selection**: **Balanced Random Forest** achieved the highest Validation Macro-F1 (**0.9295** vs 0.8619 for 1D-CNN and 0.7249 for Threshold) and dominant per-class F1 across both early warning (0.8319) and critical fire (0.9955). It is selected as the primary detection model and saved to `artifacts/best_model.joblib`.

---

## 4. Test Set Confusion Matrices (106,119 Windows)

### A. Balanced Random Forest (Best Model)
```
True \ Pred       0 (Norm)    1 (Abn)    2 (Elec)   3 (Therm)    4 (Early)    5 (Crit)      Total
0 (Normal)          59,278        502          23           0            0           0     59,803
1 (Abnormal)           692     22,093           0          85            0           0     22,870
2 (Elec Fault)          30          0       8,508           0           70           0      8,608
3 (Therm Fault)          0        255           0       1,659           28           0      1,942
4 (Early Warning)        0          0       2,779           5        7,195          13      9,992
5 (Critical Fire)        0          0           0           0           13       2,891      2,904
```

### B. Threshold Baseline Detector
```
True \ Pred       0 (Norm)    1 (Abn)    2 (Elec)   3 (Therm)    4 (Early)    5 (Crit)      Total
0 (Normal)          55,694      3,288          73           0          748           0     59,803
1 (Abnormal)           161     21,037           0         923          480         269     22,870
2 (Elec Fault)         280          0       7,645           0          683           0      8,608
3 (Therm Fault)          0          0           0       1,352          129         461      1,942
4 (Early Warning)        0          0       3,713           0        5,010       1,269      9,992
5 (Critical Fire)        0          0           0           0            3       2,901      2,904
```

### C. 1D-CNN (CPU)
```
True \ Pred       0 (Norm)    1 (Abn)    2 (Elec)   3 (Therm)    4 (Early)    5 (Crit)      Total
0 (Normal)          55,323      3,792         688           0            0           0     59,803
1 (Abnormal)           551     21,965           0         354            0           0     22,870
2 (Elec Fault)          11          0       8,395           0          202           0      8,608
3 (Therm Fault)          0         80           0       1,853            9           0      1,942
4 (Early Warning)        0          0       3,082          58        6,834          18      9,992
5 (Critical Fire)        0          0           0           0           16       2,888      2,904
```

---

## 5. Run-Level Alarm Performance

Evaluated on all 78 Test runs (12 Overcharge Runaway, 12 Internal Short Runaway, 54 Non-Runaway runs spanning 26.1 hours of testing).  
**Critical Alarm Trigger Rule**: Predicted class $\ge 4$ confirmed for **3 consecutive seconds**.

| Metric | Threshold Baseline | Balanced Random Forest (Best) | 1D-CNN |
| :--- | :---: | :---: | :---: |
| **Overcharge Runaway Detection Rate** | 100.0% (12/12) | **100.0% (12/12)** | 100.0% (12/12) |
| • Lead Time (Min) | 224.0 s | **176.0 s** | 72.0 s |
| • Lead Time (Median) | 286.0 s | **269.0 s** | 260.0 s |
| • Lead Time (Max) | 357.0 s | **368.0 s** | 345.0 s |
| **Internal Short Runaway Detection Rate** | 100.0% (12/12) | **100.0% (12/12)** | 100.0% (12/12) |
| • Lead Time (Min) | 107.0 s | **65.0 s** | 55.0 s |
| • Lead Time (Median) | 181.0 s | **97.0 s** | 93.5 s |
| • Lead Time (Max) | 252.0 s | **130.0 s** | 124.0 s |
| **Non-Runaway Test Duration** | 26.10 hours | 26.10 hours | 26.10 hours |
| **False Critical Alarm Events** | 17 events | **13 events** | **3 events** |
| **False Alarms per Hour** | 0.651 / h | **0.498 / h** | **0.115 / h** |

---

## 6. Gas Sensor Ablation Study

Retrained the best model (**Random Forest**) with all gas-related sensor features removed (**Smoke, CO, and CO2 dropped**; 24 non-gas features retained vs 34 original).

| Evaluation Metric | With Gas Sensors (Full 34 Features) | Without Gas Sensors (24 Features) | Delta / Impact of Gas Removal |
| :--- | :---: | :---: | :---: |
| **Validation Macro-F1** | **0.9295** | 0.8665 | -0.0630 (-6.8%) |
| **Test Macro-F1** | **0.9228** | 0.8766 | -0.0462 (-5.0%) |
| **Class 0 (Normal) F1** | 0.9896 | 0.9508 | -0.0388 |
| **Class 1 (Abnormal) F1** | **0.9664** | **0.8487** | **-0.1177 (-12.2%)** |
| **Class 2 (Electrical) F1** | 0.8543 | 0.7667 | -0.0876 |
| **Class 3 (Thermal) F1** | 0.8989 | 0.8919 | -0.0070 |
| **Class 4 (Early Warning) F1** | 0.8319 | 0.8073 | -0.0246 |
| **Class 5 (Critical) F1** | 0.9955 | 0.9943 | -0.0012 |
| **Overcharge Lead Time (Median)** | 269.0 s | 272.0 s | +3.0 s |
| **Overcharge Lead Time (Max)** | **368.0 s** | **330.0 s** | **-38.0 s** |
| **Internal Short Lead Time (Median)**| 97.0 s | 96.5 s | -0.5 s |
| **False Critical Alarm Events** | **13 events** | **19 events** | **+6 events (+46.2%)** |
| **False Critical Alarms per Hour** | **0.498 / h** | **0.728 / h** | **+0.230 / h (+46.2%)** |

### Key Takeaways from the Ablation
1. **False Alarm Suppression**: Removing gas sensors causes a **+46.2% surge in false alarms** (0.498 to 0.728 alarms/hour). Without gas sensors, external environmental fluctuations and thermal noise are frequently misinterpreted as early thermal precursors.
2. **Abnormal Classification Accuracy**: Class 1 F1 drops precipitously from **0.9664 to 0.8487** because external gas interference (`external_gas` scenario) cannot be classified without direct smoke/CO/CO2 discrimination.
3. **Early Warning Horizon**: In overcharge runaway, maximum early warning lead time contracts by **38 seconds** (from 368 s down to 330 s) because off-gas venting occurs dozens of seconds before significant bulk cell surface temperature rise.

---

## 7. Runtimes

| Pipeline Phase | Workers / Method | Total Duration | Per-Unit Throughput |
| :--- | :---: | :---: | :---: |
| **Scenario Profiling** | 1 CPU Worker | 5.84 s | ~0.53 s / run |
| **Dataset Generation (390 Runs)** | 12 CPU Workers (`multiprocessing`) | 24.25 s | **16.1 runs / s** |
| **Window Extraction & Feat Eng (191k windows)** | Vectorized NumPy | 42.10 s | 4,545 windows / s |
| **Threshold Baseline Training & Eval** | Pure Python/NumPy | 0.00 s (training) / 1.82 s (eval) | **0.017 ms / window** |
| **Random Forest Training (100 trees)** | 12 CPU Workers (`n_jobs=-1`) | 2.22 s | 28,783 windows / s |
| **Random Forest Inference Latency** | Single CPU Core | - | 22.28 ms / window |
| **1D-CNN Training (20 epochs)** | CPU (`batch_size=512`) | 90.63 s | 4.53 s / epoch |
| **1D-CNN Inference Latency** | Single CPU Core | - | **0.59 ms / window** |
| **Unit Test Suite (12 tests, 220 sim runs)** | `pytest` (20 seeds per scenario) | 85.46 s | 100% Passing (12/12) |

---

## 8. Assumptions & Modeling Decisions

1. **Self-Contained SOC Estimation**: Coulomb-counting requires an initial reference SOC. The feature extractor takes the mean cell voltage over the first 10 seconds of resting sensor data, maps it through an inverted NMC open-circuit voltage (OCV) empirical table, and integrates noisy current sensor readings ($I_{meas} \cdot \Delta t / Q_{nom}$) thereafter.
2. **Debounced Alarm Filter**: Critical alarms (Class $\ge 4$) require 3 consecutive seconds of positive prediction before activating. This debounces transient sensor noise while introducing a deterministic 2-second confirmation latency.
3. **False Alarm Event Definition**: In non-runaway scenarios, false alarms are counted on the rising edge of critical alarm confirmation. Persistent alarm states count as 1 continuous alarm event, and rate is normalized per test hour.
4. **Sensor Noise Robustness**: Feature engineering utilizes 60-second least-squares linear slopes calculated analytically via orthogonal dot products ($\vec{w}_{60} \cdot \vec{x}$). This acts as an optimal Wiener-like low-pass filter against high-frequency sensor noise and ADC quantization steps.
5. **No Simulator Modifications**: In strict adherence to Phase 2 guidelines, `src/simulation` remained completely untouched and frozen. All 12 regression tests verify full behavioral consistency.
