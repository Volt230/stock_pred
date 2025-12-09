# src/config.py

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Training settings
SEQ_LEN = 48         # last 48 hours
EPOCHS = 20
BATCH_SIZE = 32

# Stocks to train and predict
STOCKS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]

# Full feature list (MUST BE SAME in train & predict)
FEATURE_COLS = [
    "Open","High","Low","Close","Volume",
    "SMA_14","EMA_14","RSI_14",
    "MACD","MACD_Signal","MACD_Hist",
    "BB_Upper","BB_Middle","BB_Lower",
    "sentiment_mean","sentiment_count"
]
