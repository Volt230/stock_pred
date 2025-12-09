# src/sentiment.py
import pandas as pd
from textblob import TextBlob

def score_text(text):
    try:
        return float(TextBlob(str(text)).sentiment.polarity)
    except Exception:
        return 0.0

def aggregate_headlines_to_hourly(df, text_col="title", time_col="published", resample_rule="1h"):
    if df is None or df.empty:
        return pd.DataFrame(columns=["sentiment_mean","sentiment_count"])
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    # convert to tz-naive
    try:
        df[time_col] = df[time_col].dt.tz_convert(None)
    except Exception:
        try:
            df[time_col] = df[time_col].dt.tz_localize(None)
        except Exception:
            pass
    df["score"] = df[text_col].apply(score_text)
    df = df.set_index(time_col)
    mean = df["score"].resample(resample_rule).mean().rename("sentiment_mean")
    cnt = df["score"].resample(resample_rule).count().rename("sentiment_count")
    out = pd.concat([mean,cnt], axis=1).fillna(0)
    return out
