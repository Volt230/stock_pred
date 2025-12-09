# src/fetch_data.py

import yfinance as yf
import pandas as pd
import os
from .config import DATA_DIR


def flatten_columns(df):
    """
    Flattens MultiIndex columns into simple names.
    Example: ('Close', 'RELIANCE.NS') → 'Close'
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def fetch_price(ticker, period="60d", interval="60m", save=True):

    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if df is None or df.empty:
        print(f"[fetch_price] EMPTY data for {ticker}")
        return None

    # FIX 1: Flatten MultiIndex columns
    df = flatten_columns(df)

    # FIX 2: Normalize names (string only)
    df.columns = [str(c).capitalize() for c in df.columns]

    # FIX 3: Ensure required columns exist
    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            print(f"[fetch_price] Missing {col} for {ticker}, filling with Close")
            df[col] = df["Close"]

    # FIX 4: Clean index and remove timezone
    df.index = pd.to_datetime(df.index, errors="coerce")
    df.index = df.index.tz_localize(None)

    # Save cleaned data
    if save:
        safe = ticker.replace(".", "_")
        df.to_csv(f"{DATA_DIR}/{safe}_price.csv")

    return df


def load_price(ticker):
    safe = ticker.replace(".", "_")
    path = f"{DATA_DIR}/{safe}_price.csv"

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Flatten and normalize again
    df = flatten_columns(df)
    df.columns = [str(c).capitalize() for c in df.columns]

    return df
