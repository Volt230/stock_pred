# src/train_data.py
from .fetch_data import fetch_price, load_price
from .fetch_data import load_price as load_price_local
import pandas as pd

def get_training_data(ticker, period="90d", interval="60m"):
    """
    Use Yahoo historical (fetch_price) and replace the last candle with realtime NSE (if you have nse fetch).
    For now we simply use Yahoo and resample hourly and ffill to fill weekends.
    """
    df = fetch_price(ticker, period=period, interval=interval, save=True)
    if df is None or df.empty:
        df = load_price_local(ticker)
    # ensure tz-naive
    df.index = pd.to_datetime(df.index)
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        try:
            df.index = df.index.tz_convert(None)
        except Exception:
            pass
    # resample hourly and forward fill (fills weekends)
    df = df.resample("1h").ffill()
    return df
