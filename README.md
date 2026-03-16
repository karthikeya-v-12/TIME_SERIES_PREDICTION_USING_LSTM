# Time Series Prediction Using LSTM

This project demonstrates sales forecasting using Long Short-Term Memory (LSTM) neural networks. It includes data preprocessing, model training, and prediction visualization for time series data.

## Features

- LSTM-based forecasting model
- Data visualization (original sales, actual vs predicted)
- 7-day future sales prediction
- Kaggle integration for data handling

## Files

- `sales_forecasting_lstm.py`: Main LSTM model implementation
- `kaggle_sales_forecasting.py`: Kaggle-specific forecasting script
- `setup_kaggle.md`: Instructions for setting up Kaggle API
- `requirements.txt`: Python dependencies
- `data/sales_data.csv`: Sample sales data
- `outputs/`: Generated plots and forecasts

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/karthikeya-v-12/TIME_SERIES_PREDICTION_USING_LSTM.git
   cd TIME_SERIES_PREDICTION_USING_LSTM
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Kaggle API (if using Kaggle data):
   Follow the instructions in `setup_kaggle.md`

## Usage

Run the main forecasting script:
```bash
python sales_forecasting_lstm.py
```

For Kaggle integration:
```bash
python kaggle_sales_forecasting.py
```

## Results

The model generates:
- Original sales data plot
- Actual vs predicted sales comparison
- 7-day future sales forecast (saved to `outputs/future_7_day_forecast.csv`)

## Requirements

- Python 3.7+
- TensorFlow/Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## License

This project is open-source. Feel free to use and modify.