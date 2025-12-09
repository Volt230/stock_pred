# src/train.py

import numpy as np
import pandas as pd
import tensorflow as tf
import json
import os

from .config import (
    STOCKS, DATA_DIR, MODELS_DIR,
    SEQ_LEN, EPOCHS, BATCH_SIZE, FEATURE_COLS
)
from .fetch_data import fetch_price
from .indicators import add_technical_indicators
from .fetch_news import fetch_news_yahoo
from .sentiment import aggregate_headlines_to_hourly
from .preprocessing import scale_data, save_scaler
from .utils import ticker_to_safe


# ======================================================
# MODEL ARCHITECTURE
# ======================================================
def build_model(seq_len, n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    return model


# ======================================================
# PREPARE DF FOR ONE TICKER
# ======================================================
def prepare_training_df(ticker):

    print(f"\n[TRAIN] Fetching price data for {ticker}")

    # More training data (180 days)
    df = fetch_price(ticker, period="180d", interval="60m")

    if df is None or df.empty:
        raise ValueError("Price data empty")

    # Indicators
    df = add_technical_indicators(df)

    # Sentiment
    try:
        news = fetch_news_yahoo(ticker)
        sent = aggregate_headlines_to_hourly(news, "title", "published", "1h")
        df = df.join(sent, how="left")
    except Exception as e:
        print(f"[WARN] Sentiment error for {ticker}: {e}")

    # Fill missing sentiment
    df["sentiment_mean"] = df["sentiment_mean"].fillna(0)
    df["sentiment_count"] = df["sentiment_count"].fillna(0)

    df = df.ffill().bfill()

    # Only keep required features
    df = df[[c for c in FEATURE_COLS if c in df.columns]].copy()

    # Validate feature count
    if len(df.columns) != len(FEATURE_COLS):
        raise ValueError(f"Feature mismatch! Expected {len(FEATURE_COLS)}, got {len(df.columns)}")

    return df


# ======================================================
# CREATE TRAINING WINDOWS
# ======================================================
def create_training_sequences(df):

    X, y = [], []

    values = df.values
    close_prices = df["Close"].values

    for i in range(len(df) - SEQ_LEN - 1):
        X.append(values[i:i+SEQ_LEN])
        y.append(close_prices[i+SEQ_LEN])

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    return X, y


# ======================================================
# MAIN TRAINING LOOP
# ======================================================
if __name__ == "__main__":

    print("\n=========== TRAINING STARTED ===========\n")

    for ticker in STOCKS:

        try:
            base = ticker_to_safe(ticker)

            df = prepare_training_df(ticker)
            X, y = create_training_sequences(df)

            print(f"[{ticker}] X: {X.shape}, y: {y.shape}, features: {len(FEATURE_COLS)}")

            # Scaling
            X_scaled, y_scaled, scaler_X, scaler_y = scale_data(X, y)

            save_scaler(scaler_X, f"{MODELS_DIR}/{base}_scaler_X.pkl")
            save_scaler(scaler_y, f"{MODELS_DIR}/{base}_scaler_y.pkl")

            # Train model
            model = build_model(SEQ_LEN, len(FEATURE_COLS))
            model.fit(
                X_scaled, y_scaled,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=1
            )

            # Save model
            model.save(f"{MODELS_DIR}/{base}_lstm.h5")

            # Save features used
            json.dump(FEATURE_COLS, open(f"{MODELS_DIR}/{base}_features.json", "w"))

            print(f"[SUCCESS] Training completed for {ticker}\n")

        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")

    print("\n=========== TRAINING FINISHED ===========\n")
