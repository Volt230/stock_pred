# src/news_google.py
import feedparser
import pandas as pd
import urllib.parse

def fetch_google_news(ticker, max_items=20):
    """
    Fetch Google News RSS items for ticker (search: '<ticker_without_.NS> stock')
    Returns DataFrame with columns: title, published
    """
    query = ticker.replace(".NS","") + " stock"
    q = urllib.parse.quote_plus(query)
    url = f"https://news.google.com/rss/search?q={q}"
    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries[:max_items]:
        title = getattr(entry, "title", "")
        pub = getattr(entry, "published", None) or getattr(entry, "pubDate", None)
        if title and pub:
            rows.append({"title": title, "published": pub})
    if not rows:
        return pd.DataFrame(columns=["title","published"])
    df = pd.DataFrame(rows)
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published"])
    return df
