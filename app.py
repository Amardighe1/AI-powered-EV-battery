"""AI-Powered EV Battery Fault and Fire Detection - Demo Web Application Backend.

FastAPI application providing:
 - GET /scenarios: lists all 11 simulation scenarios
 - POST /run: executes physics simulation, computes 1 Hz features, runs ML inference + 
              hard-limit safety layer and 3-sample persistence filter
 - POST /predict_csv: runs the same inference pipeline on an uploaded 14-column CSV
 - GET /sample_csv: downloads a sample 14-column CSV for testing upload
 - GET /: serves the static frontend dashboard (index.html)
"""

import os
import sys
import io
import json
import time
from typing import Optional, List, Dict, Any
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.simulation.runner import generate_run
from src.ml.features import compute_run_soc_series, compute_features, FEATURE_NAMES

app = FastAPI(
    title="AI-Powered EV Battery Fault & Fire Detection",
    description="Software-only prototype for 4S2P Li-ion pack fault and early thermal runaway detection",
    version="1.0.0",
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SCENARIO_NAMES = [
    "normal_drive",
    "normal_charge",
    "hot_ambient",
    "external_gas",
    "overcurrent",
    "over_discharge",
    "cooling_failure",
    "overcharge_runaway",
    "internal_short_runaway",
    "overcharge_aborted",
    "soft_internal_short",
]

# Global artifacts loaded once at startup
MODEL = None
SCALER = None
SAVED_FEATURE_NAMES = None


def load_artifacts():
    global MODEL, SCALER, SAVED_FEATURE_NAMES
    if MODEL is not None:
        return
    model_path = os.path.join(REPO_ROOT, "artifacts", "best_model.joblib")
    scaler_path = os.path.join(REPO_ROOT, "artifacts", "scaler.joblib")
    feat_path = os.path.join(REPO_ROOT, "artifacts", "feature_names.json")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise RuntimeError("Artifacts not found! Ensure Phase 2 training has been executed.")

    MODEL = joblib.load(model_path)
    SCALER = joblib.load(scaler_path)
    with open(feat_path, "r") as f:
        SAVED_FEATURE_NAMES = json.load(f)
    print(f"[FastAPI] Loaded Model: {type(MODEL).__name__} | Features: {len(SAVED_FEATURE_NAMES)}")

# Eagerly load on startup
load_artifacts()


class RunRequest(BaseModel):
    scenario: str
    seed: int = 42


def apply_safety_pipeline(sensors: np.ndarray, raw_probs: np.ndarray) -> tuple:
    """Applies the physical hard-limit safety layer and 3-sample persistence filter.
    
    Args:
        sensors: (N, 14) array of sensor channels:
                 [V1, V2, V3, V4, I, T1, T2, T3, T4, T_hs, T_amb, Smoke, CO, CO2]
        raw_probs: (N, 6) array of ML class probabilities
        
    Returns:
        filtered_class: (N,) int array of debounced/filtered classes
        alarm_state: (N,) int array (1 if critical alarm active, else 0)
    """
    n = len(sensors)
    raw_preds = np.argmax(raw_probs, axis=1)
    filtered_class = np.zeros(n, dtype=int)
    alarm_state = np.zeros(n, dtype=int)

    crit_consecutive = 0
    latched_class5 = False

    for t in range(n):
        # 1. Physical hard-limit safety overrides
        t_max = float(np.max(sensors[t, 5:10]))
        v_max = float(np.max(sensors[t, 0:4]))
        v_min = float(np.min(sensors[t, 0:4]))
        i_curr = float(abs(sensors[t, 4]))
        co2_val = float(sensors[t, 13])

        candidate = int(raw_preds[t])

        # Hard-limit Layer
        if latched_class5 or t_max >= 115.0 or (t_max >= 75.0 and co2_val >= 2200.0):
            candidate = 5
            latched_class5 = True
        elif candidate < 3 and t_max >= 58.0:
            candidate = 3
        elif candidate in (0, 1) and (v_max >= 4.28 or v_min <= 2.70 or i_curr >= 25.0):
            candidate = 2

        # 2. 3-Sample Persistence Filter for Early Warning (4) & Critical (5)
        if candidate >= 4:
            crit_consecutive += 1
            if crit_consecutive >= 3:
                alarm_state[t] = 1
                filtered_class[t] = candidate
            else:
                # Hold previous confirmed class during 1-2 sample transition
                filtered_class[t] = filtered_class[max(0, t - 1)]
        else:
            crit_consecutive = 0
            if not latched_class5:
                alarm_state[t] = 0
                filtered_class[t] = candidate
            else:
                filtered_class[t] = 5
                alarm_state[t] = 1

    return filtered_class, alarm_state


def run_sensor_inference(sensors: np.ndarray) -> tuple:
    """Computes features per 1 Hz step and evaluates ML model."""
    n = len(sensors)
    soc_series = compute_run_soc_series(sensors)

    # Extract 60s windows with edge padding for t < 59
    feats_list = []
    for t in range(n):
        if t >= 59:
            win = sensors[t - 59 : t + 1]
        else:
            win = np.pad(sensors[: t + 1], ((59 - t, 0), (0, 0)), mode="edge")
        feats_list.append(compute_features(win, soc=soc_series[t]))

    X = np.array(feats_list, dtype=np.float32)
    X_scaled = SCALER.transform(X)
    probs = MODEL.predict_proba(X_scaled)
    filtered_class, alarm_state = apply_safety_pipeline(sensors, probs)

    return filtered_class, probs, alarm_state


@app.get("/", response_class=FileResponse)
def get_dashboard():
    """Serves the static single-file frontend."""
    index_path = os.path.join(REPO_ROOT, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.get("/scenarios")
def get_scenarios():
    """Returns the list of 11 scenario names."""
    return {"scenarios": SCENARIO_NAMES}


@app.post("/run")
def execute_scenario_run(req: RunRequest):
    """Executes physics simulation and returns per-second ML predictions and sensor signals."""
    if req.scenario not in SCENARIO_NAMES:
        raise HTTPException(status_code=400, detail=f"Invalid scenario '{req.scenario}'. Must be one of {SCENARIO_NAMES}")

    t0 = time.perf_counter()
    run_data = generate_run(req.scenario, seed=req.seed)
    sensors = run_data["sensors"]
    n_steps = len(sensors)

    filtered_class, probs, alarm_state = run_sensor_inference(sensors)
    proc_time = time.perf_counter() - t0

    # True labels (sent strictly for faint dashed ground-truth line comparison)
    true_labels = run_data["labels"].tolist()
    onset_idx = int(np.where(run_data["labels"] == 5)[0][0]) if np.any(run_data["labels"] == 5) else None

    return {
        "scenario": req.scenario,
        "seed": req.seed,
        "n_steps": n_steps,
        "time": [float(t) for t in run_data["time"]],
        "sensors": sensors.tolist(),
        "predicted_class": filtered_class.tolist(),
        "probs": probs.tolist(),
        "alarm_state": alarm_state.tolist(),
        "true_label": true_labels,
        "t_runaway_onset": onset_idx,
        "compute_time_s": round(proc_time, 3),
    }


@app.post("/predict_csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """Accepts a 14-column CSV (>=60 rows), runs inference, and returns per-second telemetry."""
    content = await file.read()
    try:
        # Load CSV using numpy
        text = content.decode("utf-8")
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if not lines:
            raise ValueError("CSV file is empty.")

        # Check for header
        first_line = lines[0].split(",")
        has_header = False
        try:
            float(first_line[0])
        except ValueError:
            has_header = True

        data_rows = lines[1:] if has_header else lines
        raw_data = []
        for line in data_rows:
            parts = [float(x.strip()) for x in line.split(",") if x.strip()]
            if len(parts) < 14:
                raise ValueError(f"Each row must contain at least 14 columns. Found {len(parts)} columns.")
            raw_data.append(parts[:14])

        sensors = np.array(raw_data, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")

    if len(sensors) < 60:
        raise HTTPException(status_code=400, detail=f"CSV must contain at least 60 rows for 60s windowing (got {len(sensors)}).")

    t0 = time.perf_counter()
    filtered_class, probs, alarm_state = run_sensor_inference(sensors)
    proc_time = time.perf_counter() - t0

    return {
        "scenario": "uploaded_csv",
        "seed": None,
        "n_steps": len(sensors),
        "time": list(range(len(sensors))),
        "sensors": sensors.tolist(),
        "predicted_class": filtered_class.tolist(),
        "probs": probs.tolist(),
        "alarm_state": alarm_state.tolist(),
        "true_label": None,
        "t_runaway_onset": None,
        "compute_time_s": round(proc_time, 3),
    }


@app.get("/sample_csv")
def download_sample_csv(scenario: str = "overcharge_runaway", seed: int = 42):
    """Generates and downloads a sample 14-column CSV file for testing CSV uploads."""
    sc = scenario if scenario in SCENARIO_NAMES else "overcharge_runaway"
    run_data = generate_run(sc, seed=seed)
    sensors = run_data["sensors"]

    header = "V1,V2,V3,V4,I,T1,T2,T3,T4,T_hs,T_amb,Smoke,CO,CO2\n"
    csv_rows = [header]
    for row in sensors:
        csv_rows.append(",".join(f"{val:.4f}" for val in row) + "\n")

    csv_content = "".join(csv_rows)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=sample_{sc}_seed{seed}.csv"},
    )


if __name__ == "__main__":
    import uvicorn
    # Make sure artifacts exist
    load_artifacts()
    print("[FastAPI] Starting EV Battery Demo Server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
