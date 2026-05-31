"""
scripts/train_lstm.py
═══════════════════════════════════════════════════════════
Complete LSTM Training Pipeline for AQI Prediction
Nepal / Kathmandu Air Quality Dataset
═══════════════════════════════════════════════════════════

Usage:
    python scripts/train_lstm.py --city Kathmandu --epochs 150 --horizon 24

Requirements:
    pip install tensorflow scikit-learn pandas numpy matplotlib joblib
"""

import argparse
import os
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
FEATURES = ["pm25", "pm10", "co", "no2", "so2", "o3", "temperature", "humidity"]
TARGET    = "aqi"
SEQ_LEN   = 24      # 24 time-steps as input window
FORECAST  = 1       # single-step output (multi-step loops this)
BATCH     = 32
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────

def load_data(path: str, city: str) -> pd.DataFrame:
    """
    Load AQI dataset.
    Expects CSV with columns: timestamp, city, pm25, pm10, co, no2, so2, o3,
                               temperature, humidity, aqi
    """
    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[df["city"] == city].copy()
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Loaded {len(df)} records for {city}")
    return df


def generate_synthetic_data(city: str, n_rows: int = 26280) -> pd.DataFrame:
    """
    Generate synthetic training data when real dataset unavailable.
    Simulates 3 years of hourly Kathmandu AQI readings.
    """
    logger.info(f"Generating {n_rows} synthetic records for {city}...")
    np.random.seed(42)
    t = np.arange(n_rows)

    # Diurnal pattern: peaks at 7am and 7pm
    diurnal = np.sin(2 * np.pi * (t % 24 - 14) / 24)
    # Seasonal pattern: worse in winter (Oct–Jan)
    seasonal = 0.3 * np.sin(2 * np.pi * t / (365 * 24) + np.pi)
    # Base AQI values for different cities
    base_aqi = {
        "Kathmandu": 145, "Pokhara": 90, "Lalitpur": 138,
        "Bhaktapur": 130, "Biratnagar": 120, "Bharatpur": 100,
    }.get(city, 130)

    noise = np.random.normal(0, 8, n_rows)
    aqi   = np.clip(base_aqi + 30 * diurnal + 20 * seasonal + noise, 10, 400)

    timestamps = pd.date_range("2022-01-01", periods=n_rows, freq="1h")
    df = pd.DataFrame({
        "timestamp":   timestamps,
        "city":        city,
        "aqi":         aqi.round(1),
        "pm25":        (aqi * 0.45 + np.random.normal(0, 3, n_rows)).clip(0).round(1),
        "pm10":        (aqi * 0.82 + np.random.normal(0, 5, n_rows)).clip(0).round(1),
        "co":          (1.8 + 0.3 * diurnal + np.random.normal(0, 0.2, n_rows)).clip(0).round(2),
        "no2":         (48  + 10  * diurnal + np.random.normal(0, 4, n_rows)).clip(0).round(1),
        "so2":         (22  + 5   * diurnal + np.random.normal(0, 2, n_rows)).clip(0).round(1),
        "o3":          (38  + 8   * diurnal + np.random.normal(0, 3, n_rows)).clip(0).round(1),
        "temperature": (22  + 5   * np.sin(2*np.pi*t/8760) + 3*diurnal + np.random.normal(0,1,n_rows)).round(1),
        "humidity":    (62  - 8   * diurnal + np.random.normal(0, 4, n_rows)).clip(10, 100).round(1),
    })
    return df


# ──────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ──────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and feature-engineer the dataset."""
    logger.info("Preprocessing data...")

    # Missing value imputation: linear interpolation + KNN-style fill
    df[FEATURES + [TARGET]] = df[FEATURES + [TARGET]].interpolate(method="linear")
    df.dropna(inplace=True)

    # Feature engineering
    df["hour"]         = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["month"]        = df["timestamp"].dt.month
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Lag features (past 1h, 3h, 6h, 24h AQI)
    for lag in [1, 3, 6, 24]:
        df[f"aqi_lag_{lag}"] = df[TARGET].shift(lag)
    df.dropna(inplace=True)

    # Rolling statistics
    df["aqi_roll_mean_3h"]  = df[TARGET].rolling(3).mean()
    df["aqi_roll_std_6h"]   = df[TARGET].rolling(6).std()
    df.dropna(inplace=True)

    logger.info(f"After preprocessing: {len(df)} records, {df.shape[1]} features")
    return df


def create_sequences(
    data:     np.ndarray,
    targets:  np.ndarray,
    seq_len:  int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert time-series arrays into (X, y) supervised sequences."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(targets[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ──────────────────────────────────────────────
# 3. MODEL ARCHITECTURE
# ──────────────────────────────────────────────

def build_model(seq_len: int, n_features: int) -> Sequential:
    """
    Stacked LSTM with Dropout and BatchNormalization.
    Input:  (seq_len, n_features)
    Output: (1,) — predicted AQI
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, n_features),
             recurrent_dropout=0.1),
        Dropout(0.2),
        BatchNormalization(),

        LSTM(64, return_sequences=False, recurrent_dropout=0.1),
        Dropout(0.2),

        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1),
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()
    return model


# ──────────────────────────────────────────────
# 4. TRAINING
# ──────────────────────────────────────────────

def train(model, X_train, y_train, X_val, y_val, epochs: int, city: str):
    """Train with callbacks: EarlyStopping, ReduceLR, Checkpoint."""
    ckpt_path = MODELS_DIR / f"lstm_best_{city}.h5"
    log_dir   = MODELS_DIR / "logs"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True, verbose=1),
        TensorBoard(log_dir=str(log_dir), histogram_freq=1),
    ]

    logger.info(f"Training for up to {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ──────────────────────────────────────────────
# 5. EVALUATION
# ──────────────────────────────────────────────

def evaluate(model, X_test, y_test, scaler) -> dict:
    """Compute RMSE, MAE on inverse-scaled predictions."""
    preds_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse transform (AQI is feature index 0)
    def inverse(arr):
        dummy = np.zeros((len(arr), scaler.n_features_in_))
        dummy[:, 0] = arr
        return scaler.inverse_transform(dummy)[:, 0]

    preds = inverse(preds_scaled)
    actual = inverse(y_test)

    rmse = float(np.sqrt(mean_squared_error(actual, preds)))
    mae  = float(mean_absolute_error(actual, preds))
    r2   = float(1 - np.sum((actual - preds)**2) / np.sum((actual - actual.mean())**2))

    logger.info(f"Test RMSE: {rmse:.2f}  MAE: {mae:.2f}  R²: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2, "preds": preds, "actual": actual}


# ──────────────────────────────────────────────
# 6. PLOTTING
# ──────────────────────────────────────────────

def plot_results(history, eval_result: dict, city: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"LSTM AQI Model — {city}", fontsize=14, fontweight="bold")

    # Loss curves
    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Actual vs Predicted
    n = min(200, len(eval_result["actual"]))
    axes[1].plot(eval_result["actual"][:n],  label="Actual AQI",    alpha=0.8)
    axes[1].plot(eval_result["preds"][:n],   label="Predicted AQI", alpha=0.8, linestyle="--")
    axes[1].set_title(f"Actual vs Predicted (RMSE={eval_result['rmse']:.2f})")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("AQI")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = MODELS_DIR / f"training_results_{city}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved to {out_path}")
    plt.close()


# ──────────────────────────────────────────────
# 7. MAIN PIPELINE
# ──────────────────────────────────────────────

def main(args):
    # ── Load or generate data ──
    if args.data and os.path.exists(args.data):
        df = load_data(args.data, args.city)
    else:
        logger.warning("No dataset path provided — using synthetic data.")
        df = generate_synthetic_data(args.city)

    df = preprocess(df)

    # ── Feature selection ──
    feature_cols = FEATURES + [f"aqi_lag_{l}" for l in [1, 3, 6, 24]] + \
                   ["aqi_roll_mean_3h", "aqi_roll_std_6h",
                    "hour", "day_of_week", "month", "is_peak_hour"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X_raw = df[feature_cols].values
    y_raw = df[TARGET].values

    # ── Scale ──
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X_raw)

    # Scale target separately for inverse transform
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y_raw.reshape(-1, 1)).flatten()

    # Save scaler
    scaler_path = MODELS_DIR / f"scaler_{args.city}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")

    # ── Create sequences ──
    X, y = create_sequences(X_scaled, y_scaled, SEQ_LEN)
    logger.info(f"Sequences: X={X.shape}  y={y.shape}")

    # ── Train/val/test split (70/15/15) ──
    n = len(X)
    X_train, X_temp  = X[:int(n*0.7)],  X[int(n*0.7):]
    y_train, y_temp  = y[:int(n*0.7)],  y[int(n*0.7):]
    X_val,   X_test  = X_temp[:len(X_temp)//2], X_temp[len(X_temp)//2:]
    y_val,   y_test  = y_temp[:len(y_temp)//2], y_temp[len(y_temp)//2:]

    # ── Build & train ──
    model = build_model(SEQ_LEN, X.shape[2])
    history = train(model, X_train, y_train, X_val, y_val, args.epochs, args.city)

    # ── Evaluate ──
    eval_result = evaluate(model, X_test, y_test, target_scaler)

    # ── Save final model ──
    final_path = MODELS_DIR / f"lstm_aqi_{args.city.lower()}.h5"
    model.save(str(final_path))
    logger.info(f"Final model saved to {final_path}")

    # ── Plot ──
    plot_results(history, eval_result, args.city)

    print("\n" + "="*50)
    print(f"  TRAINING COMPLETE — {args.city}")
    print(f"  RMSE : {eval_result['rmse']:.2f} AQI units")
    print(f"  MAE  : {eval_result['mae']:.2f} AQI units")
    print(f"  R²   : {eval_result['r2']:.4f}")
    print(f"  Model: {final_path}")
    print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM AQI model")
    parser.add_argument("--city",   type=str, default="Kathmandu")
    parser.add_argument("--data",   type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    main(args)