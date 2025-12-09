# src/nse_fetch.py
import requests
import pandas as pd
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
}

def fetch_nse_price(ticker):
    """
    Fetch real-time price from NSE India API.
    Works for Indian tickers like RELIANCE.NS, TCS.NS, etc.
    """
    symbol = ticker.replace(".NS", "")

    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"

    session = requests.Session()
    session.get("https://www.nseindia.com", headers=HEADERS)

    r = session.get(url, headers=HEADERS)
    r.raise_for_status()
    data = r.json()

    price_info = data["priceInfo"]
    highlow = price_info["intraDayHighLow"]

    df = pd.DataFrame({
        "Open": [price_info.get("open", None)],
        "High": [highlow.get("max", None)],
        "Low": [highlow.get("min", None)],
        "Close": [price_info.get("lastPrice", None)],
        "Volume": [data["securityInfo"].get("totalTradedVolume", 0)]
    })

    df.index = [pd.Timestamp.now()]
    return df
