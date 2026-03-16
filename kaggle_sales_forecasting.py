"""
Kaggle Sales Forecasting with Automatic Dataset Download
Downloads Store Item Demand dataset from Kaggle and runs LSTM forecasting
"""

import os
import random
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, LSTM
from keras.models import Sequential

# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
keras.utils.set_random_seed(SEED)


def download_kaggle_dataset():
    """Download Store Item Demand dataset from Kaggle"""
    print("=" * 60)
    print("DOWNLOADING KAGGLE DATASET")
    print("=" * 60)
    
    dataset_path = Path("demand-forecasting-kernels-only")
    
    if dataset_path.exists():
        print(f"✓ Dataset already exists at: {dataset_path}")
        return dataset_path / "train.csv"
    
    try:
        import kaggle
        print("Downloading 'demand-forecasting-kernels-only' dataset...")
        print("This is a 5-year sales dataset with 913,000 records")
        
        kaggle.api.dataset_download_files(
            'chakradharmattapalli/demand-forecasting-kernels-only',
            path='.',
            unzip=True
        )
        
        print(f"✓ Dataset downloaded successfully!")
        return dataset_path / "train.csv"
        
    except Exception as e:
        print(f"⚠ Kaggle download failed: {e}")
        print("\nTo enable Kaggle downloads:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section and click 'Create New Token'")
        print("3. Place kaggle.json in: ~/.kaggle/ (Linux/Mac) or C:\\Users\\<You>\\.kaggle\\ (Windows)")
        print("\nUsing fallback: generating sample data...")
        return generate_sample_data()


def generate_sample_data():
    """Generate sample sales data as fallback"""
    print("\nGenerating sample sales data...")
    dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='D')
    
    # Create realistic sales pattern with trend and seasonality
    trend = np.linspace(100, 200, len(dates))
    seasonal = 50 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
    noise = np.random.normal(0, 10, len(dates))
    sales = trend + seasonal + noise
    sales = np.maximum(sales, 0)  # No negative sales
    
    df = pd.DataFrame({'date': dates, 'sales': sales})
    
    sample_path = Path("data")
    sample_path.mkdir(exist_ok=True)
    csv_path = sample_path / "generated_sales_data.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"✓ Sample data created: {csv_path}")
    return csv_path


def prepare_kaggle_data(csv_path):
    """Load and prepare Kaggle dataset"""
    print("\n" + "=" * 60)
    print("PREPARING DATA")
    print("=" * 60)
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded dataset: {len(df):,} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Kaggle dataset has: date, store, item, sales
    # Aggregate by date for overall sales forecasting
    if 'date' in df.columns and 'sales' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df_agg = df.groupby('date')['sales'].sum().reset_index()
        df_agg.columns = ['Date', 'Sales']
    else:
        # Fallback for different column names
        df.columns = ['Date', 'Sales'] if len(df.columns) == 2 else df.columns
        df['Date'] = pd.to_datetime(df['Date'])
        df_agg = df[['Date', 'Sales']].copy()
    
    df_agg = df_agg.sort_values('Date').reset_index(drop=True)
    
    print(f"✓ Aggregated to {len(df_agg):,} daily records")
    print(f"Date range: {df_agg['Date'].min()} to {df_agg['Date'].max()}")
    print(f"Sales range: {df_agg['Sales'].min():.2f} to {df_agg['Sales'].max():.2f}")
    
    return df_agg


def plot_original_sales(df, output_dir):
    """Plot original sales over time"""
    plt.figure(figsize=(14, 6))
    plt.plot(df["Date"], df["Sales"], color="steelblue", linewidth=1.5, alpha=0.8)
    plt.title("Original Sales Data Over Time", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sales", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "01_original_sales.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/01_original_sales.png")


def create_sequences(scaled_sales, dates, window_size=30):
    """Build sliding windows for LSTM"""
    x_values, y_values, target_dates = [], [], []
    
    for i in range(window_size, len(scaled_sales)):
        x_values.append(scaled_sales[i - window_size : i, 0])
        y_values.append(scaled_sales[i, 0])
        target_dates.append(dates[i])
    
    x_arr = np.array(x_values)
    y_arr = np.array(y_values)
    dates_arr = np.array(target_dates)
    
    x_arr = x_arr.reshape((x_arr.shape[0], x_arr.shape[1], 1))
    return x_arr, y_arr, dates_arr


def build_model(window_size=30):
    """Create LSTM model"""
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def plot_training_history(history, output_dir):
    """Plot training and validation loss"""
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Training History', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "02_training_history.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/02_training_history.png")


def plot_actual_vs_predicted(test_dates, actual, predicted, output_dir):
    """Plot actual vs predicted sales"""
    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, actual, label="Actual Sales", linewidth=2, alpha=0.8)
    plt.plot(test_dates, predicted, label="Predicted Sales", linewidth=2, alpha=0.8)
    plt.title("Actual vs Predicted Sales (Test Set)", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sales", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "03_actual_vs_predicted.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/03_actual_vs_predicted.png")


def forecast_future(model, scaler, full_scaled_series, last_date, window_size=30, days_ahead=30):
    """Forecast future sales"""
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


def plot_forecast(df_history, df_forecast, output_dir):
    """Plot historical data with forecast"""
    plt.figure(figsize=(14, 6))
    
    # Plot last 180 days of history
    recent_history = df_history.tail(180)
    plt.plot(recent_history["Date"], recent_history["Sales"], 
             label="Historical Sales", linewidth=2, color='steelblue')
    
    # Plot forecast
    plt.plot(df_forecast["Date"], df_forecast["Forecast_Sales"], 
             label="30-Day Forecast", linewidth=2, color='orange', linestyle='--')
    
    plt.title("Sales Forecast (Next 30 Days)", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sales", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "04_future_forecast.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/04_future_forecast.png")


def main():
    print("\n" + "=" * 60)
    print("KAGGLE SALES FORECASTING WITH LSTM")
    print("=" * 60 + "\n")
    
    # Step 1: Download dataset
    csv_path = download_kaggle_dataset()
    
    # Step 2: Prepare data
    df = prepare_kaggle_data(csv_path)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Step 3: Plot original data
    print("\n" + "=" * 60)
    print("VISUALIZING DATA")
    print("=" * 60)
    plot_original_sales(df, output_dir)
    
    # Step 4: Scale data
    print("\n" + "=" * 60)
    print("PREPARING SEQUENCES")
    print("=" * 60)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_sales = scaler.fit_transform(df[["Sales"]].values)
    
    # Step 5: Create sequences
    window_size = 30
    x_all, y_all, y_dates = create_sequences(scaled_sales, df["Date"].values, window_size)
    print(f"✓ Created {len(x_all):,} sequences (window size: {window_size})")
    
    # Step 6: Train/test split
    split_idx = int(len(x_all) * 0.8)
    x_train, x_test = x_all[:split_idx], x_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]
    test_dates = y_dates[split_idx:]
    print(f"✓ Train: {len(x_train):,} | Test: {len(x_test):,}")
    
    # Step 7: Build and train model
    print("\n" + "=" * 60)
    print("TRAINING LSTM MODEL")
    print("=" * 60)
    model = build_model(window_size)
    
    early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)
    
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop],
        verbose=1
    )
    
    plot_training_history(history, output_dir)
    
    # Step 8: Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    y_pred_scaled = model.predict(x_test, verbose=0)
    
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_actual = scaler.inverse_transform(y_pred_scaled).flatten()
    
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
    
    print(f"✓ Mean Absolute Error (MAE): {mae:.2f}")
    print(f"✓ Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    plot_actual_vs_predicted(test_dates, y_test_actual, y_pred_actual, output_dir)
    
    # Step 9: Forecast future
    print("\n" + "=" * 60)
    print("FORECASTING NEXT 30 DAYS")
    print("=" * 60)
    future_df = forecast_future(
        model, scaler, scaled_sales[:, 0], 
        df["Date"].iloc[-1], window_size, days_ahead=30
    )
    
    # Save forecast
    forecast_path = output_dir / "forecast_30_days.csv"
    future_df.to_csv(forecast_path, index=False)
    print(f"✓ Saved forecast: {forecast_path}")
    
    plot_forecast(df, future_df, output_dir)
    
    # Display forecast summary
    print("\n" + "=" * 60)
    print("FORECAST SUMMARY (Next 30 Days)")
    print("=" * 60)
    print(f"{'Date':<15} {'Forecasted Sales':>20}")
    print("-" * 40)
    for _, row in future_df.head(10).iterrows():
        print(f"{str(row['Date'].date()):<15} {row['Forecast_Sales']:>20,.2f}")
    print("...")
    print(f"\nAverage forecasted sales: {future_df['Forecast_Sales'].mean():,.2f}")
    print(f"Total forecasted sales (30 days): {future_df['Forecast_Sales'].sum():,.2f}")
    
    print("\n" + "=" * 60)
    print("✓ COMPLETE! All outputs saved to 'outputs/' folder")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
