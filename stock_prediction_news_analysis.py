# stock_prediction.py

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import matplotlib.pyplot as plt

# Constants
NEWS_API_URL = "https://newsapi.org/v2/everything"
NEWS_API_KEY = "YOUR_API_KEY"  # Replace with your News API key

# Step 1: Load Historical Stock Prices
def get_stock_data(stock_symbol, start, end):
    stock_data = yf.download(stock_symbol, start=start, end=end)
    stock_data['Price_Change'] = stock_data['Close'].diff()
    stock_data['Target'] = (stock_data['Price_Change'] > 0).astype(int)
    return stock_data.dropna()

# Step 2: Perform Sentiment Analysis on News Articles
def analyze_sentiment(article):
    return TextBlob(article).sentiment.polarity

# Step 3: Load News Articles
def get_news_data(stock_symbol):
    params = {
        'q': stock_symbol,
        'from': '2023-01-01',
        'to': '2023-12-31',
        'sortBy': 'relevancy',
        'apiKey': NEWS_API_KEY
    }
    response = requests.get(NEWS_API_URL, params=params)
    news_data = response.json()
    articles = news_data.get('articles', [])

    sentiments = []
    for article in articles:
        sentiments.append(analyze_sentiment(article['title'] + " " + article['description']))

    return pd.DataFrame({'Sentiment': sentiments})

# Step 4: Combine Stock Data and Sentiment Data
def prepare_data(stock_symbol):
    stock_data = get_stock_data(stock_symbol, '2023-01-01', '2023-12-31')
    news_data = get_news_data(stock_symbol)

    # Check if news_data['Sentiment'] is empty
    if news_data['Sentiment'].empty:
        # Handle empty news data, e.g., fill with 0 or a neutral sentiment
        combined_data = stock_data.copy()
        combined_data['Sentiment'] = 0  # Fill with 0 if no sentiment data
    else:
        # Adjust length of sentiments to match stock_data index
        if len(news_data) > len(stock_data):
            news_data = news_data[:len(stock_data)]

        combined_data = stock_data.copy()
        combined_data['Sentiment'] = np.random.choice(news_data['Sentiment'], size=len(stock_data))

    return combined_data.dropna()

# Step 5: Model Training
def train_model(combined_data):
    X = combined_data[['Open', 'High', 'Low', 'Volume', 'Sentiment']]
    y = combined_data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Step 6: Execute the Project
if __name__ == "__main__":
    stock_symbol = 'AAPL'  # Example: Apple Inc.
    combined_data = prepare_data(stock_symbol)
    train_model(combined_data)

    # Optional: Visualize stock prices and sentiment
    combined_data['Close'].plot(title='Stock Prices with Sentiment')
    plt.show()