from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from textblob import TextBlob

# Function to fetch and preprocess stock data
def fetch_and_preprocess(ticker):
    df = yf.download(ticker, period='2y')
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(window=14).mean()))
    df['Return'] = df['Close'].pct_change()

    features = ['Close', 'MA10', 'MA50', 'RSI', 'Return', 'Volume']  # Ensure the correct number of features

    # Add placeholders for missing features to match the expected input shape
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0  # Replace with appropriate values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features, index=df.index)
    return scaled_df, scaler, df

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

# Function to fetch options data
def fetch_options_data(ticker):
    stock = yf.Ticker(ticker)
    options_dates = stock.options
    options_data = []
    for date in options_dates:
        options = stock.option_chain(date)
        options_data.append({
            'expirationDate': date,
            'calls': options.calls.to_dict(orient='records'),
            'puts': options.puts.to_dict(orient='records')
        })
    return options_data

# Function to fetch news data and analyze sentiment
def fetch_news_data(ticker):
    stock = yf.Ticker(ticker)
    news = stock.news
    for article in news:
        analysis = TextBlob(article['title'])
        article['sentiment'] = analysis.sentiment.polarity
    return news

# Function to fetch economic events (for simplicity, we'll use placeholders)
def fetch_economic_events():
    return [
        {'date': '2023-06-15', 'event': 'Federal Reserve Meeting'},
        {'date': '2023-07-01', 'event': 'US Jobs Report'}
    ]

# Load the pre-trained model and error standard deviation
model = tf.keras.models.load_model('lstm_model.keras')
train_error_std = np.load('train_error_std.npy')
test_error_std = np.load('test_error_std.npy')

# Create the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    scaled_df, scaler, original_df = fetch_and_preprocess(ticker)
    seq_length = 60
    X = create_sequences(scaled_df.values, seq_length)
    last_sequence = X[-1]
    last_sequence_scaled = np.expand_dims(last_sequence, axis=0)
    prediction = model.predict(last_sequence_scaled)
    predicted_price = scaler.inverse_transform(np.array([[prediction[0][0], 0, 0, 0, 0, 0]]))[0][0]

    # Calculate prediction intervals
    lower_bound = predicted_price - 1.96 * test_error_std
    upper_bound = predicted_price + 1.96 * test_error_std

    options_data = fetch_options_data(ticker)
    news_data = fetch_news_data(ticker)
    economic_events = fetch_economic_events()

    return render_template('index.html',
                           ticker=ticker,
                           prediction_text=f'Predicted Next Day Close: {predicted_price:.2f}',
                           prediction_range=f'95% Prediction Interval: {lower_bound:.2f} - {upper_bound:.2f}',
                           options_data=options_data,
                           news_data=news_data,
                           economic_events=economic_events)

if __name__ == '__main__':
    app.run(debug=True)
