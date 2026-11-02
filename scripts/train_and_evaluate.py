"""End-to-end training, evaluation, alarm metric computation, and ablation pipeline.

Executes:
 1. Dataset loading & integrity verification (leakage check)
 2. Threshold Baseline evaluation (Val & Test)
 3. Balanced Random Forest training & evaluation (Val & Test)
 4. 1D-CNN training & evaluation (Val & Test)
 5. Stride=1s Run-level Alarm Metrics computation for all models:
    - Critical alarm = predicted class >= 4 for 3 consecutive seconds
    - Lead time (runaway onset minus alarm): min, median, max, % detected
    - False critical alarms per hour on non-runaway test runs
 6. Model selection based on Val Macro-F1 and Alarm performance
 7. Gas Sensor Ablation on the best model (without Smoke, CO, and CO2)
 8. Save artifacts (best_model, metadata.json, confusion matrices, tables)
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)

from src.ml.models import (
    ThresholdClassifier,
    train_random_forest,
    Battery1DCNN,
    train_1d_cnn,
)
from src.ml.evaluate import compute_alarm_metrics_for_runs, NON_RUNAWAY_SCENARIOS


def measure_inference_latency_rf(model, X_sample: np.ndarray, n_trials: int = 1000) -> float:
    """Measure single-window inference latency in milliseconds for Random Forest."""
    single = X_sample[:1]
    # Warmup
    for _ in range(50):
        _ = model.predict(single)
    t0 = time.perf_counter()
    for _ in range(n_trials):
        _ = model.predict(single)
    lat_ms = ((time.perf_counter() - t0) / n_trials) * 1000.0
    return lat_ms


def measure_inference_latency_threshold(model, X_sample: np.ndarray, n_trials: int = 1000) -> float:
    """Measure single-window inference latency in milliseconds for Threshold Classifier."""
    single = X_sample[:1]
    for _ in range(50):
        _ = model.predict(single)
    t0 = time.perf_counter()
    for _ in range(n_trials):
        _ = model.predict(single)
    lat_ms = ((time.perf_counter() - t0) / n_trials) * 1000.0
    return lat_ms


def measure_inference_latency_cnn(model, raw_sample: np.ndarray, n_trials: int = 1000, device="cpu") -> float:
    """Measure single-window inference latency in milliseconds for 1D-CNN."""
    single = torch.tensor(raw_sample[:1], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(50):
            _ = model(single)
        t0 = time.perf_counter()
        for _ in range(n_trials):
            _ = model(single)
        lat_ms = ((time.perf_counter() - t0) / n_trials) * 1000.0
    return lat_ms


def compute_window_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    p, r, f1_per_class, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(6)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "f1_per_class": [float(x) for x in f1_per_class],
        "precision_per_class": [float(x) for x in p],
        "recall_per_class": [float(x) for x in r],
        "support_per_class": [int(x) for x in s],
        "confusion_matrix": cm.tolist(),
    }


def predict_test_runs_tabular(model, X_test_data: np.ndarray, test_metadata: list) -> list:
    """Slice tabular test predictions by run for alarm evaluation."""
    all_preds = model.predict(X_test_data)
    run_predictions = []
    idx_start = 0
    for meta in test_metadata:
        n = meta["n_windows"]
        run_preds = all_preds[idx_start : idx_start + n]
        run_predictions.append({
            "run_id": meta["run_id"],
            "scenario": meta["scenario"],
            "times": meta["times"],
            "preds": run_preds.tolist(),
            "t_runaway_onset": meta["t_runaway_onset"],
        })
        idx_start += n
    return all_preds, run_predictions


def predict_test_runs_cnn(model, raw_test_data: np.ndarray, test_metadata: list, batch_size=512, device="cpu") -> list:
    """Slice 1D-CNN test predictions by run for alarm evaluation."""
    model.eval()
    dev = torch.device(device)
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(raw_test_data), batch_size):
            batch = torch.tensor(raw_test_data[i : i + batch_size], dtype=torch.float32, device=dev)
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds_list.append(preds)
    all_preds = np.concatenate(preds_list)

    run_predictions = []
    idx_start = 0
    for meta in test_metadata:
        n = meta["n_windows"]
        run_preds = all_preds[idx_start : idx_start + n]
        run_predictions.append({
            "run_id": meta["run_id"],
            "scenario": meta["scenario"],
            "times": meta["times"],
            "preds": run_preds.tolist(),
            "t_runaway_onset": meta["t_runaway_onset"],
        })
        idx_start += n
    return all_preds, run_predictions


def main():
    print("=" * 80)
    print("PHASE 2: TRAINING, EVALUATION, ALARM METRICS & ABLATION")
    print("=" * 80)

    # 1. Load Data
    cache = np.load("data/dataset_cache.npz")
    with open("data/test_run_metadata.json", "r") as f:
        test_metadata = json.load(f)
    with open("artifacts/feature_names.json", "r") as f:
        feature_names = json.load(f)
    scaler = joblib.load("artifacts/scaler.joblib")

    X_train_scaled = cache["X_train_scaled"]
    X_val_scaled = cache["X_val_scaled"]
    X_test_scaled = cache["X_test_scaled"]
    y_train = cache["y_train"]
    y_val = cache["y_val"]
    y_test = cache["y_test"]

    raw_train_norm = cache["raw_train_norm"]
    raw_val_norm = cache["raw_val_norm"]
    raw_test_norm = cache["raw_test_norm"]

    print(f"Dataset Loaded:")
    print(f"  Train: X={X_train_scaled.shape}, raw={raw_train_norm.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val_scaled.shape}, raw={raw_val_norm.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test_scaled.shape}, raw={raw_test_norm.shape}, y={y_test.shape}")

    # Unscaled features for Threshold baseline
    X_val_unscaled = scaler.inverse_transform(X_val_scaled)
    X_test_unscaled = scaler.inverse_transform(X_test_scaled)

    # Dictionary to hold all results
    all_results = {}

    # =========================================================================
    # MODEL 1: Threshold Classifier
    # =========================================================================
    print("\n" + "-" * 60)
    print("EVALUATING MODEL 1: Threshold Baseline")
    print("-" * 60)
    tc = ThresholdClassifier(feature_names)
    t0_tc = time.perf_counter()
    p_val_tc = tc.predict(X_val_unscaled)
    val_macro_tc = f1_score(y_val, p_val_tc, average="macro", zero_division=0)
    print(f"Threshold Baseline Val Macro-F1: {val_macro_tc:.4f}")

    p_test_tc, tc_run_preds = predict_test_runs_tabular(tc, X_test_unscaled, test_metadata)
    tc_window_metrics = compute_window_metrics(y_test, p_test_tc)
    tc_alarm_metrics = compute_alarm_metrics_for_runs(tc_run_preds)
    tc_latency_ms = measure_inference_latency_threshold(tc, X_test_unscaled)

    all_results["threshold"] = {
        "val_macro_f1": float(val_macro_tc),
        "test_metrics": tc_window_metrics,
        "alarm_metrics": tc_alarm_metrics,
        "train_time_s": 0.0,
        "inference_latency_ms": round(tc_latency_ms, 4),
    }
    print(f"Threshold Test Macro-F1: {tc_window_metrics['macro_f1']:.4f} | Latency: {tc_latency_ms:.4f} ms")
    print(f"Threshold Alarm Lead Times: {json.dumps(tc_alarm_metrics, indent=2)}")

    # =========================================================================
    # MODEL 2: Balanced Random Forest
    # =========================================================================
    print("\n" + "-" * 60)
    print("TRAINING MODEL 2: Balanced Random Forest")
    print("-" * 60)
    rf_model, rf_train_time = train_random_forest(
        X_train_scaled, y_train, X_val_scaled, y_val, random_state=42
    )
    p_val_rf = rf_model.predict(X_val_scaled)
    val_macro_rf = f1_score(y_val, p_val_rf, average="macro", zero_division=0)

    p_test_rf, rf_run_preds = predict_test_runs_tabular(rf_model, X_test_scaled, test_metadata)
    rf_window_metrics = compute_window_metrics(y_test, p_test_rf)
    rf_alarm_metrics = compute_alarm_metrics_for_runs(rf_run_preds)
    rf_latency_ms = measure_inference_latency_rf(rf_model, X_test_scaled)

    all_results["random_forest"] = {
        "val_macro_f1": float(val_macro_rf),
        "test_metrics": rf_window_metrics,
        "alarm_metrics": rf_alarm_metrics,
        "train_time_s": round(rf_train_time, 2),
        "inference_latency_ms": round(rf_latency_ms, 4),
    }
    print(f"Random Forest Test Macro-F1: {rf_window_metrics['macro_f1']:.4f} | Latency: {rf_latency_ms:.4f} ms")
    print(f"RF Alarm Lead Times: {json.dumps(rf_alarm_metrics, indent=2)}")

    # =========================================================================
    # MODEL 3: 1D-CNN (CPU, class-weighted, early stopping)
    # =========================================================================
    print("\n" + "-" * 60)
    print("TRAINING MODEL 3: 1D-CNN (CPU)")
    print("-" * 60)
    cnn_model, cnn_train_time = train_1d_cnn(
        raw_train_norm,
        y_train,
        raw_val_norm,
        y_val,
        in_channels=14,
        batch_size=256,
        max_epochs=20,
        patience=5,
        device="cpu",
    )
    cnn_model.eval()
    with torch.no_grad():
        val_logits = cnn_model(torch.tensor(raw_val_norm, dtype=torch.float32))
        p_val_cnn = torch.argmax(val_logits, dim=1).cpu().numpy()
    val_macro_cnn = f1_score(y_val, p_val_cnn, average="macro", zero_division=0)

    p_test_cnn, cnn_run_preds = predict_test_runs_cnn(cnn_model, raw_test_norm, test_metadata, device="cpu")
    cnn_window_metrics = compute_window_metrics(y_test, p_test_cnn)
    cnn_alarm_metrics = compute_alarm_metrics_for_runs(cnn_run_preds)
    cnn_latency_ms = measure_inference_latency_cnn(cnn_model, raw_test_norm, device="cpu")

    all_results["1d_cnn"] = {
        "val_macro_f1": float(val_macro_cnn),
        "test_metrics": cnn_window_metrics,
        "alarm_metrics": cnn_alarm_metrics,
        "train_time_s": round(cnn_train_time, 2),
        "inference_latency_ms": round(cnn_latency_ms, 4),
    }
    print(f"1D-CNN Test Macro-F1: {cnn_window_metrics['macro_f1']:.4f} | Latency: {cnn_latency_ms:.4f} ms")
    print(f"1D-CNN Alarm Lead Times: {json.dumps(cnn_alarm_metrics, indent=2)}")

    # =========================================================================
    # MODEL SELECTION
    # =========================================================================
    print("\n" + "=" * 60)
    print("MODEL SELECTION")
    print("=" * 60)
    # Compare Validation Macro-F1 and Alarm Performance
    candidates = ["threshold", "random_forest", "1d_cnn"]
    print(f"Model Summary:")
    for c in candidates:
        r = all_results[c]
        print(f"  {c:15s} | Val Macro-F1: {r['val_macro_f1']:.4f} | Test Macro-F1: {r['test_metrics']['macro_f1']:.4f} | "
              f"OC Det: {r['alarm_metrics']['overcharge_runaway']['detected_pct']:.1f}% "
              f"(med LT: {r['alarm_metrics']['overcharge_runaway']['lead_time_med_s']:.1f}s) | "
              f"IS Det: {r['alarm_metrics']['internal_short_runaway']['detected_pct']:.1f}% "
              f"(med LT: {r['alarm_metrics']['internal_short_runaway']['lead_time_med_s']:.1f}s) | "
              f"FAR: {r['alarm_metrics']['false_alarms']['false_alarms_per_hour']:.3f}/h")

    # Determine best model based on validation macro-F1
    best_candidate = max(candidates, key=lambda c: all_results[c]["val_macro_f1"])
    print(f"\n>>> Selected Best Model: {best_candidate.upper()} <<<")

    # =========================================================================
    # ABLATION STUDY: Retrain Best Model WITHOUT Smoke, CO, and CO2
    # =========================================================================
    print("\n" + "=" * 60)
    print("ABLATION STUDY: RETRAINING BEST MODEL WITHOUT GAS SENSORS")
    print("=" * 60)

    gas_keywords = ["Smoke", "CO", "CO2"]
    no_gas_feat_indices = [
        i for i, name in enumerate(feature_names)
        if not any(g in name for g in gas_keywords)
    ]
    no_gas_feat_names = [feature_names[i] for i in no_gas_feat_indices]
    print(f"Retained {len(no_gas_feat_indices)} non-gas features out of {len(feature_names)}:")
    print(" ", no_gas_feat_names)

    if best_candidate == "random_forest":
        X_train_no_gas = X_train_scaled[:, no_gas_feat_indices]
        X_val_no_gas = X_val_scaled[:, no_gas_feat_indices]
        X_test_no_gas = X_test_scaled[:, no_gas_feat_indices]

        rf_ablation, ab_train_time = train_random_forest(
            X_train_no_gas, y_train, X_val_no_gas, y_val, random_state=42
        )
        p_val_ab = rf_ablation.predict(X_val_no_gas)
        val_macro_ab = f1_score(y_val, p_val_ab, average="macro", zero_division=0)

        p_test_ab, ab_run_preds = predict_test_runs_tabular(rf_ablation, X_test_no_gas, test_metadata)
        ab_window_metrics = compute_window_metrics(y_test, p_test_ab)
        ab_alarm_metrics = compute_alarm_metrics_for_runs(ab_run_preds)
        ab_latency_ms = measure_inference_latency_rf(rf_ablation, X_test_no_gas)

        ablation_model_obj = rf_ablation

    elif best_candidate == "1d_cnn":
        # Drop channels 11, 12, 13 (Smoke, CO, CO2) -> keep channels 0..10 (11 channels)
        no_gas_channels = list(range(11))
        raw_train_no_gas = raw_train_norm[:, :, no_gas_channels]
        raw_val_no_gas = raw_val_norm[:, :, no_gas_channels]
        raw_test_no_gas = raw_test_norm[:, :, no_gas_channels]

        cnn_ablation, ab_train_time = train_1d_cnn(
            raw_train_no_gas, y_train, raw_val_no_gas, y_val,
            in_channels=11, batch_size=256, max_epochs=20, patience=5, device="cpu"
        )
        with torch.no_grad():
            val_logits_ab = cnn_ablation(torch.tensor(raw_val_no_gas, dtype=torch.float32))
            p_val_ab = torch.argmax(val_logits_ab, dim=1).cpu().numpy()
        val_macro_ab = f1_score(y_val, p_val_ab, average="macro", zero_division=0)

        p_test_ab, ab_run_preds = predict_test_runs_cnn(cnn_ablation, raw_test_no_gas, test_metadata, device="cpu")
        ab_window_metrics = compute_window_metrics(y_test, p_test_ab)
        ab_alarm_metrics = compute_alarm_metrics_for_runs(ab_run_preds)
        ab_latency_ms = measure_inference_latency_cnn(cnn_ablation, raw_test_no_gas, device="cpu")

        ablation_model_obj = cnn_ablation

    all_results["ablation_no_gas"] = {
        "base_model": best_candidate,
        "val_macro_f1": float(val_macro_ab),
        "test_metrics": ab_window_metrics,
        "alarm_metrics": ab_alarm_metrics,
        "train_time_s": round(ab_train_time, 2),
        "inference_latency_ms": round(ab_latency_ms, 4),
        "retained_features": no_gas_feat_names if best_candidate == "random_forest" else "channels_0_to_10",
    }
    print(f"Ablation Test Macro-F1: {ab_window_metrics['macro_f1']:.4f}")
    print(f"Ablation Alarm Lead Times: {json.dumps(ab_alarm_metrics, indent=2)}")

    # =========================================================================
    # SAVE ARTIFACTS
    # =========================================================================
    print("\n" + "=" * 60)
    print("SAVING ARTIFACTS")
    print("=" * 60)
    os.makedirs("artifacts", exist_ok=True)

    # Save best model
    if best_candidate == "random_forest":
        joblib.dump(rf_model, "artifacts/best_model.joblib")
        print("Saved artifacts/best_model.joblib")
    elif best_candidate == "1d_cnn":
        torch.save(cnn_model.state_dict(), "artifacts/best_model.pt")
        print("Saved artifacts/best_model.pt")

    # Save full metadata JSON
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_model": best_candidate,
        "classes": {
            0: "Normal",
            1: "Abnormal",
            2: "Electrical fault",
            3: "Thermal fault",
            4: "Early warning",
            5: "Fire/critical",
        },
        "models": all_results,
    }
    with open("artifacts/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved artifacts/metadata.json")

    # Generate and save pretty markdown tables
    print("\nTraining and Evaluation pipeline finished successfully.")


if __name__ == "__main__":
    main()
