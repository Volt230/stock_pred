import yfinance as yf
import matplotlib.pyplot as plt
from textblob import TextBlob
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# -----------------------------
# 1. FETCH STOCK PRICE DATA
# -----------------------------
def get_stock_data(symbol):
    data = yf.download(symbol, period="1mo", interval="1d")
    print("\n--- STOCK DATA (HEAD) ---")
    print(data.head())
    return data

# -----------------------------
# 2. BASIC PRICE GRAPH
# -----------------------------
def plot_stock(data, symbol):
    plt.figure(figsize=(10,5))
    plt.plot(data["Close"])
    plt.title(f"{symbol} Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()

# -----------------------------
# 3. NEWS FETCH + SENTIMENT
# -----------------------------
def fetch_news(api_key, query):
    print("\n--- FETCHING NEWS ---")
    
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url).json()

    articles = response.get("articles", [])
    
    news_sentiments = []

    for article in articles[:5]:  # Analyze top 5
        title = article["title"]
        sentiment = TextBlob(title).sentiment.polarity
        news_sentiments.append((title, sentiment))

    print("\n--- NEWS + SENTIMENT ---")
    for title, sentiment in news_sentiments:
        print(f"{title}  --> Sentiment: {sentiment}")

    return news_sentiments

# -----------------------------
# 4. SIMPLE MACHINE LEARNING PREDICTION
# -----------------------------
def train_prediction_model(data):
    data = data.reset_index()
    data["Day"] = np.arange(len(data))

    X = data["Day"].values.reshape(-1, 1)
    y = data["Close"].values

    model = LinearRegression()
    model.fit(X, y)

    next_day = [[len(data)]]
    prediction = model.predict(next_day)[0]

    print("\n--- NEXT DAY PRICE PREDICTION ---")
    print("Predicted Close Price:", prediction)

    return prediction

# -----------------------------
# MAIN DRIVER FUNCTION
# -----------------------------
if __name__ == "__main__":
    print("=== STOCK MARKET DEMO MODEL ===")

    symbol = input("Enter Stock Symbol (e.g., AAPL, TSLA, TCS.NS): ")
    API_KEY = input("Enter NewsAPI KEY for news analysis: ")

    stock_data = get_stock_data(symbol)

    plot_stock(stock_data, symbol)

    fetch_news(API_KEY, symbol)

    train_prediction_model(stock_data)
