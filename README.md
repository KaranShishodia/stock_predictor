# 📈 Stock Predictor Pro

A fully modular, production-quality stock price analysis and deep-learning forecasting system built with **Keras (LSTM)**, **yfinance**, and **Streamlit**.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Data** | Live OHLCV data via `yfinance` |
| **Indicators** | SMA, EMA, RSI, MACD, Bollinger Bands, Volume MA |
| **Model** | Stacked LSTM (128 → 64) + Dropout + Dense, saved as `.keras` |
| **Training** | Early stopping, LR reduction, model checkpointing |
| **Evaluation** | RMSE, MAE, MAPE, R² — printed and plotted |
| **Forecasting** | N-day rolling forward forecast |
| **Dashboard** | Streamlit app with interactive Plotly charts |
| **CLI** | Every module runnable as a standalone script |

---

## 🗂 Project Structure

```
stock_predictor/
├── config.py          # Central configuration (hyperparams, paths, etc.)
├── data.py            # Data fetching + feature engineering
├── model.py           # Keras LSTM architecture + save/load helpers
├── train.py           # Training pipeline
├── predict.py         # Back-test + future forecasting pipeline
├── evaluate.py        # Metrics (RMSE/MAE/MAPE/R²) + loss/prediction plots
├── visualize.py       # All matplotlib/seaborn charts
├── app.py             # Streamlit dashboard
├── requirements.txt
├── models/            # Saved .keras model + scaler.pkl (auto-created)
└── outputs/           # Generated PNG charts (auto-created)
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
# Default: GOOG, 2015-2024
python train.py

# Custom ticker and range
python train.py --ticker AAPL --start 2018-01-01 --end 2024-01-01 --epochs 100
```

Training will:
- Download data automatically
- Compute all technical indicators
- Train the LSTM with early stopping
- Save the model to `models/stock_lstm_model.keras`
- Save the scaler to `models/scaler.pkl`
- Print RMSE / MAE / MAPE / R² on the test set

### 3. Run predictions

```bash
# Back-test on held-out test data
python predict.py --ticker GOOG

# Back-test + 30-day future forecast
python predict.py --ticker GOOG --future_days 30
```

### 4. Launch the Streamlit dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ⚙️ Configuration

Edit `config.py` to change any global settings without touching logic:

```python
DEFAULT_TICKER    = "GOOG"
DEFAULT_START_DATE = "2015-01-01"
LOOKBACK_WINDOW   = 100   # past days fed into LSTM
LSTM_UNITS_1      = 128
LSTM_UNITS_2      = 64
EPOCHS            = 50
BATCH_SIZE        = 32
```

---

## 🧠 Model Architecture

```
Input  →  (LOOKBACK_WINDOW, 1)
           ↓
LSTM(128, return_sequences=True)  →  Dropout(0.2)
           ↓
LSTM(64,  return_sequences=False) →  Dropout(0.2)
           ↓
Dense(25, relu)
           ↓
Dense(1)       ← predicted next-day Close price (scaled)
```

**Loss:** Mean Squared Error (MSE)  
**Optimizer:** Adam (lr=0.001)  
**Callbacks:** EarlyStopping · ReduceLROnPlateau · ModelCheckpoint

---

## 📊 Technical Indicators

| Indicator | Parameters |
|---|---|
| SMA | 20, 50, 100, 200 days |
| EMA | 12, 26 days |
| RSI | 14-period |
| MACD | 12/26/9 |
| Bollinger Bands | 20-period, 2σ |
| Volume MA | 20-period |

---

## 📉 CLI Reference

```bash
# Data check
python data.py --ticker MSFT --start 2018-01-01 --end 2024-01-01

# Model summary
python model.py

# Train
python train.py --ticker TSLA --epochs 75

# Predict / forecast
python predict.py --ticker TSLA --future_days 20

# Streamlit app
streamlit run app.py
```

---

## ⚠️ Disclaimer

This project is **for educational purposes only**.  
Stock price predictions should not be used for real investment decisions.  
Past model performance does not guarantee future accuracy.

---

## 📦 Tech Stack

- [TensorFlow / Keras](https://keras.io/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/)
