# src/fetch_news.py

import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
import urllib.parse
from datetime import datetime

# ===========================================================
#  GOOGLE NEWS FALLBACK
# ===========================================================
def fetch_google_news(ticker, max_items=20):
    try:
        query = ticker.replace(".NS", "") + " stock"
        q = urllib.parse.quote_plus(query)
        url = f"https://news.google.com/rss/search?q={q}"

        feed = feedparser.parse(url)
        rows = []

        for entry in feed.entries[:max_items]:
            title = getattr(entry, "title", "")
            pub = getattr(entry, "published", None)
            if title and pub:
                rows.append({"title": title, "published": pub})

        if not rows:
            return pd.DataFrame(columns=["title", "published"])

        df = pd.DataFrame(rows)
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
        df.dropna(subset=["published"], inplace=True)
        return df

    except Exception as e:
        print("[Google News] ERROR:", e)
        return pd.DataFrame(columns=["title", "published"])


# ===========================================================
#  YAHOO NEWS (PRIMARY SOURCE)
# ===========================================================
def fetch_yahoo_news(ticker, max_items=20):
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        html = requests.get(url, timeout=5).text
        soup = BeautifulSoup(html, "html.parser")

        rows = []
        items = soup.select("h3 a")[:max_items]

        for tag in items:
            title = tag.text.strip()
            if title:
                rows.append({"title": title, "published": datetime.utcnow()})

        if not rows:
            return pd.DataFrame(columns=["title", "published"])

        df = pd.DataFrame(rows)
        return df

    except Exception as e:
        print("[Yahoo News] ERROR:", e)
        return pd.DataFrame(columns=["title", "published"])


# ===========================================================
#  HYBRID NEWS FETCHER (Used by training & prediction)
# ===========================================================
def fetch_news_yahoo(ticker, max_items=30):
    """
    TRAIN & PREDICT both call this function.
    Now it:
    1. Tries Yahoo Finance news 
    2. If Yahoo fails → Falls back to Google News
    3. Combines both sources
    4. Returns clean DataFrame
    """

    print(f"[NEWS] Fetching news for {ticker}")

    df_y = fetch_yahoo_news(ticker, max_items)
    df_g = fetch_google_news(ticker, max_items)

    # Combine & clean
    df = pd.concat([df_y, df_g], ignore_index=True)
    df.drop_duplicates(subset=["title"], inplace=True)
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published"])
    df = df.sort_values("published")

    if df.empty:
        print("[NEWS] No news found for", ticker)

    return df
