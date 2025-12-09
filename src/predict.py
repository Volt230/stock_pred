# src/predict.py
import numpy as np
import pandas as pd
import tensorflow as tf
import os

from .config import DATA_DIR, MODELS_DIR, SEQ_LEN, STOCKS
from .fetch_data import load_price
from .indicators import add_technical_indicators
from .sentiment import aggregate_headlines_to_hourly
from .fetch_news import fetch_news_yahoo
from .preprocessing import load_scaler
from .utils import ticker_to_safe, save_df


# =====================================================================
#  MAIN FUTURE PREDICTION FUNCTION
# =====================================================================
def predict_future_series(ticker, steps=24):

    base = ticker_to_safe(ticker)

    # ----------------------------------------------------------
    # LOAD PRICE DATA
    # ----------------------------------------------------------
    df = load_price(ticker)
    if df is None or df.empty:
        raise ValueError(f"No price data found for {ticker}")

    # ----------------------------------------------------------
    # ADD INDICATORS
    # ----------------------------------------------------------
    df = add_technical_indicators(df)

    # ----------------------------------------------------------
    # SENTIMENT (safe fallback)
    # ----------------------------------------------------------
    try:
        news = fetch_news_yahoo(ticker)
        sent = aggregate_headlines_to_hourly(news, "title", "published", "1h")
        df = df.join(sent, how="left")
    except Exception as e:
        print("[WARN] Sentiment unavailable:", e)
        df["sentiment_mean"] = 0
        df["sentiment_count"] = 0

    # fill missing
    df["sentiment_mean"] = df["sentiment_mean"].fillna(0).infer_objects(copy=False)
    df["sentiment_count"] = df["sentiment_count"].fillna(0).infer_objects(copy=False)

    df = df.ffill().bfill()

    # ----------------------------------------------------------
    # FEATURES
    # ----------------------------------------------------------
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_14", "EMA_14", "RSI_14",
        "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Upper", "BB_Middle", "BB_Lower",
        "sentiment_mean", "sentiment_count"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values

    # ----------------------------------------------------------
    # PAD IF NOT ENOUGH DATA
    # ----------------------------------------------------------
    if len(X) < SEQ_LEN:
        pad_rows = np.tile(X[-1], (SEQ_LEN - len(X), 1))
        X = np.vstack([pad_rows, X])

    window = X[-SEQ_LEN:]

    # ----------------------------------------------------------
    # LOAD SCALERS
    # ----------------------------------------------------------
    scaler_X = load_scaler(f"{MODELS_DIR}/{base}_scaler_X.pkl")
    scaler_y = load_scaler(f"{MODELS_DIR}/{base}_scaler_y.pkl")

    # ----------------------------------------------------------
    # LOAD MODEL
    # ----------------------------------------------------------
    model_path = f"{MODELS_DIR}/{base}_lstm.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found: " + model_path)

    model = tf.keras.models.load_model(model_path)

    # ----------------------------------------------------------
    # FUTURE PREDICTIONS
    # ----------------------------------------------------------
    preds = []
    cur = window.copy()

    for _ in range(steps):

        # scale
        scaled = scaler_X.transform(cur.reshape(1, -1))
        scaled = scaled.reshape(1, SEQ_LEN, len(feature_cols))

        # ---------------------------
        # UNIVERSAL PREDICTION HANDLER
        # handles both 1-output and 2-output models
        # ---------------------------
        out = model.predict(scaled, verbose=0)

        if isinstance(out, np.ndarray):
            # one-output model
            pred_scaled = out
        elif isinstance(out, list) and len(out) >= 1:
            # multiple-output model
            pred_scaled = out[0]
        else:
            raise ValueError("Unrecognized model output format")

        price = scaler_y.inverse_transform(pred_scaled)[0][0]
        preds.append(price)

        # update window (autoregressive)
        new_row = cur[-1].copy()
        close_index = feature_cols.index("Close")
        new_row[close_index] = price

        cur = np.vstack([cur[1:], new_row])

    # ----------------------------------------------------------
    # BUILD FUTURE TIMESTAMP INDEX
    # ----------------------------------------------------------
    last_time = df.index.max()
    idx = [last_time + pd.Timedelta(hours=i + 1) for i in range(steps)]

    out = pd.DataFrame({"pred_price": preds}, index=idx)

    # ----------------------------------------------------------
    # SAVE FOR DASHBOARD
    # ----------------------------------------------------------
    save_path = f"{DATA_DIR}/{base}_future_preds.csv"
    save_df(out, save_path)

    print(f"[OK] Saved future predictions → {save_path}")

    return out


# =====================================================================
#  COMMAND LINE MODE
# =====================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--ticker", type=str)
    parser.add_argument("--all", action="store_true", help="Run predictions for all stocks")

    args = parser.parse_args()

    if args.all:
        for t in STOCKS:
            try:
                print("\nPredicting", t)
                print(predict_future_series(t))
            except Exception as e:
                print("ERR", t, e)

    elif args.ticker:
        print(predict_future_series(args.ticker))

    else:
        print("Use --ticker SYMBOL or --all")
