"""Comprehensive evaluation, alarm metrics, and ablation engine.

Metrics:
 - Window-level Classification: Macro-F1, Precision, Recall, Accuracy, Confusion Matrix
 - Run-level Alarm Metrics (stride=1s on Test runs):
   - Critical alarm = predicted class >= 4 for 3 consecutive seconds
   - Lead time (runaway onset minus alarm): min, median, max, % detected before onset
   - False critical alarms per hour on non-runaway runs
 - Gas Sensor Ablation: evaluates model performance without Smoke, CO, and CO2
"""

import json
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)

from .features import FEATURE_NAMES

NON_RUNAWAY_SCENARIOS = {
    "normal_drive", "normal_charge", "hot_ambient", "external_gas",
    "overcurrent", "over_discharge", "cooling_failure",
    "overcharge_aborted", "soft_internal_short"
}


def compute_alarm_metrics_for_runs(
    run_predictions: list,  # list of dicts: {"scenario": sc, "times": [...], "preds": [...], "t_runaway_onset": float/None}
):
    """Compute run-level alarm metrics on test runs with stride=1s."""
    sc_stats = {
        "overcharge_runaway": {"lead_times": [], "detected_before_onset": 0, "total_runs": 0},
        "internal_short_runaway": {"lead_times": [], "detected_before_onset": 0, "total_runs": 0},
    }

    false_alarm_events = 0
    total_non_runaway_seconds = 0.0

    for run in run_predictions:
        sc = run["scenario"]
        times = np.array(run["times"])
        preds = np.array(run["preds"])
        t_onset = run["t_runaway_onset"]
        n_windows = len(preds)

        # Detect critical alarms: predicted class >= 4 for 3 consecutive seconds
        is_crit = (preds >= 4)
        crit_3s = np.zeros(n_windows, dtype=bool)
        for i in range(2, n_windows):
            if is_crit[i] and is_crit[i - 1] and is_crit[i - 2]:
                crit_3s[i] = True

        first_alarm_time = None
        if np.any(crit_3s):
            first_alarm_idx = int(np.where(crit_3s)[0][0])
            first_alarm_time = float(times[first_alarm_idx])

        # A. Runaway scenarios: lead time
        if sc in sc_stats:
            sc_stats[sc]["total_runs"] += 1
            if t_onset is not None and first_alarm_time is not None:
                lead_time = t_onset - first_alarm_time
                sc_stats[sc]["lead_times"].append(lead_time)
                if lead_time >= 0.0:
                    sc_stats[sc]["detected_before_onset"] += 1
            elif t_onset is not None and first_alarm_time is None:
                # Undetected runaway
                sc_stats[sc]["lead_times"].append(-999.0)

        # B. Non-runaway scenarios: count false critical alarm events
        elif sc in NON_RUNAWAY_SCENARIOS:
            run_duration = times[-1] - times[0] if len(times) > 1 else 0.0
            total_non_runaway_seconds += run_duration
            
            # Count rising edges of critical alarm
            in_alarm = False
            for active in crit_3s:
                if active and not in_alarm:
                    false_alarm_events += 1
                    in_alarm = True
                elif not active and in_alarm:
                    in_alarm = False

    # Aggregate runaway metrics
    summary = {}
    for sc, data in sc_stats.items():
        lts = [lt for lt in data["lead_times"] if lt != -999.0]
        n_runs = data["total_runs"]
        det_pct = (data["detected_before_onset"] / max(1, n_runs)) * 100.0
        if len(lts) > 0:
            summary[sc] = {
                "detected_pct": det_pct,
                "lead_time_min_s": float(np.min(lts)),
                "lead_time_med_s": float(np.median(lts)),
                "lead_time_max_s": float(np.max(lts)),
                "total_runs": n_runs,
            }
        else:
            summary[sc] = {
                "detected_pct": 0.0,
                "lead_time_min_s": np.nan,
                "lead_time_med_s": np.nan,
                "lead_time_max_s": np.nan,
                "total_runs": n_runs,
            }

    # False alarm rate per hour
    total_hours = total_non_runaway_seconds / 3600.0
    far_per_hour = (false_alarm_events / max(1e-4, total_hours)) if total_hours > 0 else 0.0
    summary["false_alarms"] = {
        "false_alarm_events": false_alarm_events,
        "total_test_hours": round(total_hours, 2),
        "false_alarms_per_hour": round(far_per_hour, 3),
    }

    return summary

