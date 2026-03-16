"""
sales_forecasting_lstm.py

Complete Sales Forecasting project using LSTM.
Dataset format required: CSV with columns -> Date, Sales

Example:
    python sales_forecasting_lstm.py --data_path data/sales_data.csv
"""

import argparse
import os
import random
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, LSTM
from keras.models import Sequential


# Reproducibility for consistent training behavior.
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
keras.utils.set_random_seed(SEED)


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load CSV, validate schema, parse datetime, and sort by date."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(path)

    required_cols = {"Date", "Sales"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Date", "Sales"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < 30:
        raise ValueError("Dataset is too small. Please provide at least 30 rows.")

    return df


def plot_original_sales(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot original sales over time."""
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Sales"], color="steelblue", linewidth=2)
    plt.title("Original Sales Data")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "original_sales.png", dpi=150)
    plt.close()


def create_sequences(
    scaled_sales: np.ndarray, dates: np.ndarray, window_size: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows:
    X = previous `window_size` values, y = next value.
    """
    x_values, y_values, target_dates = [], [], []

    for i in range(window_size, len(scaled_sales)):
        x_values.append(scaled_sales[i - window_size : i, 0])
        y_values.append(scaled_sales[i, 0])
        target_dates.append(dates[i])

    x_arr = np.array(x_values)
    y_arr = np.array(y_values)
    dates_arr = np.array(target_dates)

    # LSTM expects shape: (samples, timesteps, features)
    x_arr = x_arr.reshape((x_arr.shape[0], x_arr.shape[1], 1))
    return x_arr, y_arr, dates_arr


def build_model(window_size: int = 10) -> Sequential:
    """Create and compile LSTM model (1 LSTM + 1 Dense)."""
    model = Sequential(
        [
            Input(shape=(window_size, 1)),
            LSTM(64),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def plot_actual_vs_predicted(
    test_dates: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot actual and predicted test values."""
    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, actual, label="Actual Sales", linewidth=2)
    plt.plot(test_dates, predicted, label="Predicted Sales", linewidth=2)
    plt.title("Actual vs Predicted Sales (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()


def forecast_next_days(
    model: Sequential,
    scaler: MinMaxScaler,
    full_scaled_series: np.ndarray,
    last_date: pd.Timestamp,
    window_size: int = 10,
    days_ahead: int = 7,
) -> pd.DataFrame:
    """
    Forecast future points recursively:
    1st prediction uses last real window.
    Each next prediction uses previous predictions in the window.
    """
    window = full_scaled_series[-window_size:].reshape(1, window_size, 1)
    future_scaled = []

    for _ in range(days_ahead):
        next_scaled = model.predict(window, verbose=0)[0, 0]
        future_scaled.append(next_scaled)
        new_window = np.append(window[0, 1:, 0], next_scaled)
        window = new_window.reshape(1, window_size, 1)

    future_values = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days_ahead + 1)]

    return pd.DataFrame({"Date": future_dates, "Forecast_Sales": future_values})


def main(data_path: str, output_dir: str) -> None:
    # 1) Load and clean data
    df = load_and_prepare_data(data_path)
    out_path = Path(output_dir)

    # 2) Plot original series
    plot_original_sales(df, out_path)

    # 3) Scale sales values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_sales = scaler.fit_transform(df[["Sales"]].values)

    # 4) Create sliding-window sequences (10 -> 1)
    window_size = 10
    x_all, y_all, y_dates = create_sequences(scaled_sales, df["Date"].values, window_size=window_size)

    # 5) Train/test split (80/20) preserving temporal order
    split_idx = int(len(x_all) * 0.8)
    x_train, x_test = x_all[:split_idx], x_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]
    test_dates = y_dates[split_idx:]

    # 6) Build model
    model = build_model(window_size=window_size)

    # 7) Train model
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=16,
        shuffle=False,  # Keep sequence order for time-series training
        callbacks=[early_stop],
        verbose=1,
    )

    # 8) Predict test set
    y_pred_scaled = model.predict(x_test, verbose=0)

    # 9) Inverse transform values back to original scale
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_actual = scaler.inverse_transform(y_pred_scaled).flatten()

    # 10) Plot actual vs predicted
    plot_actual_vs_predicted(test_dates, y_test_actual, y_pred_actual, out_path)

    # 11) Predict next 7 days
    future_df = forecast_next_days(
        model=model,
        scaler=scaler,
        full_scaled_series=scaled_sales[:, 0],
        last_date=df["Date"].iloc[-1],
        window_size=window_size,
        days_ahead=7,
    )

    # 12) Print 7-day forecast clearly
    print("\n7-Day Sales Forecast")
    print("-" * 30)
    for _, row in future_df.iterrows():
        print(f"{row['Date'].date()} -> {row['Forecast_Sales']:.2f}")

    # Save forecast table
    future_df.to_csv(out_path / "future_7_day_forecast.csv", index=False)
    print(f"\nSaved plots and forecast CSV to: {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sales Forecasting using LSTM")
    parser.add_argument("--data_path", type=str, required=True, help="CSV path with Date,Sales columns")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save plots/results")
    args = parser.parse_args()

    main(data_path=args.data_path, output_dir=args.output_dir)
