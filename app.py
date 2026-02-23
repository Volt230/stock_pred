# dashboard/app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

from src.config import STOCKS, DATA_DIR
from src.predict import predict_future_series
from src.utils import ticker_to_safe


# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")
st.title("📈 Stock Prediction Dashboard")


# ---------------------------------------------------------
# CLEAN CSV LOADER (fixed for future_preds.csv)
# ---------------------------------------------------------
def load_price_csv(path):
    """Loads price CSV even if file contains garbage headers or extra columns."""
    if not os.path.exists(path):
        return None

    try:
        clean_rows = []

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                # Skip garbage header lines
                low = line.lower()
                if any(low.startswith(x) for x in ["price", "ticker", "datetime", "date", "symbol", "generated"]):
                    continue

                # Detect datetime-like rows (start with number + contain ":" and "-")
                if not (line[0].isdigit() and "-" in line and ":" in line):
                    continue

                # Split CSV properly
                parts = [p.strip() for p in line.split(",")]

                # Need at least 6 columns, can have more → keep only first six
                if len(parts) < 6:
                    continue

                clean_rows.append(parts[:6])   # ONLY FIRST SIX COLUMNS

        if not clean_rows:
            return None

        df = pd.DataFrame(clean_rows, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])

        # Parse datetime
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df = df.dropna(subset=["Datetime"])
        df = df.set_index("Datetime")

        # Remove timezone info
        try:
            df.index = df.index.tz_localize(None)
        except:
            try:
                df.index = df.index.tz_convert(None)
            except:
                pass

        # Convert numeric values
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna()

        return df

    except Exception as e:
        st.error(f"❌ Price CSV Load Error: {e}")
        return None



def load_future_csv(path):
    """Loads future prediction CSV (only timestamp + pred_price)"""
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path, index_col=0)

        # INDEX must be datetime
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna()

        # Convert numeric
        df["pred_price"] = pd.to_numeric(df["pred_price"], errors="coerce")
        df = df.dropna()

        return df

    except Exception as e:
        st.error(f"❌ Future CSV Load Error: {e}")
        return None


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
ticker = st.sidebar.selectbox("Select Stock", STOCKS)

if st.sidebar.button("▶ Predict Next 24 Hours"):
    with st.spinner("Generating prediction..."):
        try:
            predict_future_series(ticker, steps=24)
            st.success("✔ Prediction Completed")
        except Exception as e:
            st.error(f"❌ Prediction Failed: {e}")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
safe = ticker_to_safe(ticker)

price_path = f"{DATA_DIR}/{safe}_price.csv"
future_path = f"{DATA_DIR}/{safe}_future_preds.csv"

df = load_price_csv(price_path)
future_df = load_future_csv(future_path)


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
st.subheader("📌 Current vs Next Prediction")

col1, col2 = st.columns(2)

if df is not None and len(df):
    last_close = float(df["Close"].iloc[-1])
    col1.metric("Latest Close", f"{last_close:.2f}")
else:
    last_close = None
    col1.metric("Latest Close", "N/A")

if future_df is not None and len(future_df):
    next_pred = float(future_df["pred_price"].iloc[0])

    if last_close is not None:
        delta = next_pred - last_close
        pct = (delta / last_close) * 100
        col2.metric("Next Prediction", f"{next_pred:.2f}", f"{delta:.2f} ({pct:.2f}%)")
    else:
        col2.metric("Next Prediction", f"{next_pred:.2f}")
else:
    col2.metric("Next Prediction", "N/A")

st.markdown("---")


# ---------------------------------------------------------
# PLOT: LAST 48 HOURS + PREDICTION
# ---------------------------------------------------------
st.subheader("📊 Last 48 Hours + Next 24 Hours Prediction")

fig = go.Figure()

# HISTORY (48 hours)
if df is not None and len(df):
    last_time = df.index.max()
    start_time = last_time - pd.Timedelta(hours=48)

    history = df[df.index >= start_time]

    fig.add_trace(go.Scatter(
        x=history.index,
        y=history["Close"],
        mode="lines",
        name="History (48h)",
        line=dict(color="#1f77b4", width=3)
    ))

# FUTURE (24 hours)
if future_df is not None and len(future_df):
    fig.add_trace(go.Scatter(
        x=future_df.index,
        y=future_df["pred_price"],
        mode="lines+markers",
        name="Prediction (24h)",
        line=dict(color="#FFA500", width=3)
    ))

fig.update_layout(
    template="plotly_dark",
    height=520,
    xaxis_title="Time",
    yaxis_title="Price",
    legend=dict(orientation="h", yanchor="bottom", y=1.05)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ---------------------------------------------------------
# TABLE OF FUTURE PREDICTIONS
# ---------------------------------------------------------
st.subheader("🔮 Future Prediction Table (Next 24 Hours)")

if future_df is not None and len(future_df):
    st.dataframe(future_df)
else:
    st.info("Click 'Predict Next 24 Hours' to generate predictions.")
