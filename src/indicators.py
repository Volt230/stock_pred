# src/indicators.py
import pandas as pd
import numpy as np

def SMA(series, period=14):
    return series.rolling(window=period, min_periods=1).mean()

def EMA(series, period=14):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def BollingerBands(series, period=20, std_factor=2):
    sma = SMA(series, period)
    std = series.rolling(window=period, min_periods=1).std()
    upper = sma + std_factor * std
    lower = sma - std_factor * std
    return upper, sma, lower

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # flatten columns if multi-level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    # ensure 'Close' present
    if 'Close' not in df.columns:
        close_candidates = [c for c in df.columns if 'Close' in str(c)]
        if close_candidates:
            df['Close'] = df[close_candidates[0]]
        else:
            raise KeyError("No 'Close' column found when adding indicators.")
    df['SMA_14'] = SMA(df['Close'], 14)
    df['EMA_14'] = EMA(df['Close'], 14)
    df['RSI_14'] = RSI(df['Close'], 14)
    macd, macd_signal, macd_hist = MACD(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist
    bb_upper, bb_middle, bb_lower = BollingerBands(df['Close'])
    df['BB_Upper'] = bb_upper
    df['BB_Middle'] = bb_middle
    df['BB_Lower'] = bb_lower
    # fill missing values safely
    df = df.bfill().ffill()
    return df
