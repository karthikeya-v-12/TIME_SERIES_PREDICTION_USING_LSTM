# Kaggle Sales Forecasting Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Kaggle API (Optional but Recommended)

To automatically download datasets from Kaggle:

#### Windows:
1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token" - this downloads `kaggle.json`
4. Create folder: `C:\Users\<YourUsername>\.kaggle\`
5. Move `kaggle.json` to that folder

#### Linux/Mac:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Run the Script
```bash
python kaggle_sales_forecasting.py
```

## What Happens When You Run

1. **Auto-downloads** the "Store Item Demand Forecasting" dataset (913K records, 5 years)
2. **Prepares data** - aggregates daily sales across all stores
3. **Trains LSTM model** - uses 30-day windows to predict next day
4. **Generates outputs**:
   - `outputs/01_original_sales.png` - Historical sales visualization
   - `outputs/02_training_history.png` - Model training progress
   - `outputs/03_actual_vs_predicted.png` - Model accuracy on test data
   - `outputs/04_future_forecast.png` - 30-day forecast visualization
   - `outputs/forecast_30_days.csv` - Detailed forecast data

5. **Prints results** - Shows forecast summary in terminal

## No Kaggle API?

If you don't setup Kaggle API, the script automatically generates realistic sample data with:
- 6 years of daily sales data
- Seasonal patterns
- Growth trends
- Random variations

You'll still get all outputs and forecasts!

## Dataset Details

**Primary Dataset**: Store Item Demand Forecasting Challenge
- 913,000 records
- 5 years of data (2013-2017)
- 10 stores, 50 items
- Perfect for time series forecasting

**Fallback**: Auto-generated synthetic data
- 6 years of daily data
- Realistic patterns
- No setup required

## Output Guarantee

Every run produces:
- 4 visualization plots
- 1 CSV with 30-day forecast
- Terminal output with forecast summary
- Model performance metrics (MAE, MAPE)

## Troubleshooting

**Issue**: Kaggle download fails
- **Solution**: Script automatically uses sample data

**Issue**: Out of memory
- **Solution**: Script uses efficient batching (32 samples)

**Issue**: No output folder
- **Solution**: Script creates it automatically

## Customization

Edit these variables in `kaggle_sales_forecasting.py`:
- `window_size = 30` - Days of history to use for prediction
- `days_ahead = 30` - How many days to forecast
- `epochs = 50` - Training iterations
- `batch_size = 32` - Training batch size
