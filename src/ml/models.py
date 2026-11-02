"""Model definitions: Threshold Baseline, Balanced Random Forest, and 1D-CNN.

Includes training pipelines with early stopping on validation macro-F1.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.utils.class_weight import compute_class_weight


class ThresholdClassifier:
    """Standard BMS rule-based threshold baseline detector.
    
    Priority: 5 > 4 > 2 > 3 > 1 > 0
    Operates on unscaled engineered features or raw window features.
    """

    def __init__(self, feature_names: list):
        self.feature_names = feature_names
        self.f_idx = {name: i for i, name in enumerate(feature_names)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes 0..5 based on physical threshold hierarchy."""
        n_samples = len(X)
        preds = np.zeros(n_samples, dtype=int)

        t_max_last = X[:, self.f_idx["T_max_last"]]
        t_max_slope_15s = X[:, self.f_idx["T_max_slope_15s"]]
        t_max_slope_60s = X[:, self.f_idx["T_max_slope_60s"]]
        dt_cell = X[:, self.f_idx["dT_cell_last"]]
        v_max = X[:, self.f_idx["V_max_last"]]
        v_min = X[:, self.f_idx["V_min_last"]]
        v_spread = X[:, self.f_idx["V_spread_last"]]
        i_curr = np.abs(X[:, self.f_idx["I_mean"]])
        i_inst = np.abs(X[:, self.f_idx["I_last"]])
        co = X[:, self.f_idx["CO_last"]]
        co2 = X[:, self.f_idx["CO2_last"]]
        soc = X[:, self.f_idx["SOC"]]

        for i in range(n_samples):
            # 1. Class 5: Critical thermal runaway / fire
            if t_max_last[i] >= 115.0 or t_max_slope_15s[i] >= 0.35 or (co2[i] >= 2200.0 and t_max_last[i] >= 75.0):
                preds[i] = 5
                continue

            # 2. Class 4: Early precursor warning
            if t_max_slope_60s[i] >= 0.04 or (co[i] >= 20.0 and co2[i] >= 900.0 and t_max_last[i] >= 35.0) or (soc[i] >= 1.06 and co2[i] >= 700.0):
                preds[i] = 4
                continue

            # 3. Class 2: Electrical fault
            if v_max[i] >= 4.25 or v_min[i] <= 2.80 or i_curr[i] >= 15.0 or i_inst[i] >= 30.0:
                preds[i] = 2
                continue

            # 4. Class 3: Thermal fault
            if t_max_last[i] >= 58.0:
                preds[i] = 3
                continue

            # 5. Class 1: Abnormal
            if (44.0 <= t_max_last[i] < 58.0) or dt_cell[i] >= 7.0 or v_spread[i] >= 0.055 or co[i] >= 25.0 or co2[i] >= 1100.0:
                preds[i] = 1
                continue

            # 6. Class 0: Normal
            preds[i] = 0

        return preds


class Battery1DCNN(nn.Module):
    """1D-CNN operating on raw (14, 60) sensor windows.
    
    Lightweight CPU-optimized architecture with residual depth and batch norm.
    """

    def __init__(self, in_channels: int = 14, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 14 -> 32 channels, length 60 -> 30
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2: 32 -> 64 channels, length 30 -> 15
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3: 64 -> 64 channels, length 15 -> 1
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, 60, in_channels) -> transpose to (B, in_channels, 60)
        if x.dim() == 3 and x.shape[1] == 60:
            x = x.transpose(1, 2)
        out = self.features(x)
        logits = self.classifier(out)
        return logits


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train Balanced Random Forest on engineered features."""
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    rf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    preds_val = rf.predict(X_val)
    val_macro_f1 = f1_score(y_val, preds_val, average="macro")
    print(f"Random Forest trained in {train_time:.2f} s | Val Macro-F1: {val_macro_f1:.4f}")
    return rf, train_time


def train_1d_cnn(
    raw_train: np.ndarray, # (N, 60, C)
    y_train: np.ndarray,
    raw_val: np.ndarray,   # (M, 60, C)
    y_val: np.ndarray,
    in_channels: int = 14,
    batch_size: int = 256,
    max_epochs: int = 25,
    patience: int = 5,
    device: str = "cpu",
):
    """Train 1D-CNN on normalized raw sensor windows using PyTorch on CPU."""
    torch.manual_seed(42)
    dev = torch.device(device)

    # Compute balanced class weights for loss
    classes = np.arange(6)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = torch.tensor(cw, dtype=torch.float32, device=dev)

    train_ds = TensorDataset(torch.tensor(raw_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(raw_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = Battery1DCNN(in_channels=in_channels, num_classes=6).to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_macro_f1 = -1.0
    best_weights = None
    no_improve_epochs = 0
    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)

        # Validation evaluation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(dev)
                logits = model(bx)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)

        val_preds = np.array(val_preds)
        val_macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        scheduler.step(val_macro_f1)

        print(f"  Epoch {epoch:2d}/{max_epochs:2d}: Val Macro-F1 = {val_macro_f1:.4f} (best={max(best_macro_f1, val_macro_f1):.4f})")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"  Early stopping triggered at epoch {epoch} (patience={patience})")
                break

    train_time = time.perf_counter() - t0
    model.load_state_dict(best_weights)
    model.eval()
    print(f"1D-CNN training completed in {train_time:.2f} s | Best Val Macro-F1: {best_macro_f1:.4f}")
    return model, train_time

