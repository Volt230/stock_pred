# src/utils.py
import pickle
import re
import os

def ticker_to_safe(ticker: str) -> str:
    # e.g. "TCS.NS" -> "TCS_NS"
    return re.sub(r"[^0-9A-Za-z]+", "_", ticker).strip("_")

def save_df(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
