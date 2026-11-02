# Benchmark Report: 20 Seeds per Scenario (220 Total Runs)

| Scenario | Mean Dur (s) | Label 0 | Label 1 | Label 2 | Label 3 | Label 4 | Label 5 | Runaway Rate | Runaway Onset (min/med/max) | Class 4 Window (min/med/max) | Max Core Temp (°C) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **normal_drive** | 1799 | 1800 | 0 | 0 | 0 | 0 | 0 | 0/20 (0%) | N/A | N/A | 10.6 / 18.6 / 41.4 °C |
| **normal_charge** | 1799 | 1800 | 0 | 0 | 0 | 0 | 0 | 0/20 (0%) | N/A | N/A | 7.8 / 17.1 / 40.2 °C |
| **hot_ambient** | 1799 | 902 | 898 | 0 | 0 | 0 | 0 | 0/20 (0%) | N/A | N/A | 46.6 / 48.1 / 50.2 °C |
| **external_gas** | 1799 | 1143 | 657 | 0 | 0 | 0 | 0 | 0/20 (0%) | N/A | N/A | 8.6 / 17.2 / 40.5 °C |
| **overcurrent** | 1799 | 1753 | 0 | 47 | 0 | 0 | 0 | 0/20 (0%) | N/A | N/A | 11.3 / 19.8 / 42.2 °C |
| **over_discharge** | 1799 | 1705 | 0 | 95 | 0 | 0 | 0 | 0/20 (0%) | N/A | N/A | 7.8 / 16.5 / 40.2 °C |
| **cooling_failure** | 1799 | 205 | 1367 | 0 | 228 | 0 | 0 | 0/20 (0%) | N/A | N/A | 59.6 / 61.3 / 63.7 °C |
| **overcharge_runaway** | 707 | 44 | 0 | 290 | 0 | 253 | 121 | 20/20 (100%) | 519 / 576 / 674 s | 106 / 254 / 381 s | 996.2 / 999.5 / 1000.0 °C |
| **internal_short_runaway** | 435 | 113 | 57 | 0 | 50 | 95 | 121 | 20/20 (100%) | 230 / 310 / 412 s | 65 / 94 / 137 s | 883.3 / 997.3 / 1000.0 °C |
| **overcharge_aborted** | 1799 | 199 | 0 | 997 | 0 | 604 | 0 | 0/20 (0%) | N/A | N/A | 13.2 / 26.4 / 57.0 °C |
| **soft_internal_short** | 1799 | 708 | 1092 | 0 | 0 | 0 | 0 | 0/20 (0%) | N/A | N/A | 20.7 / 29.3 / 57.3 °C |