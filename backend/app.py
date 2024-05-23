import json
from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from textblob import TextBlob
from flask_cors import CORS, cross_origin

# Function to fetch and preprocess stock data
def fetch_and_preprocess(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period='2y')
    if df.empty:
        return None, None, None, None

    # Fetch current market price
    current_price = stock.history(period='1d')['Close'].iloc[-1]

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
    return scaled_df, scaler, df, current_price

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
        calls = options.calls.to_dict(orient='records')
        puts = options.puts.to_dict(orient='records')

        options_by_strike = {}

        for call in calls:
            strike = call['strike']
            if strike not in options_by_strike:
                options_by_strike[strike] = {'call': None, 'put': None}
            options_by_strike[strike]['call'] = call

        for put in puts:
            strike = put['strike']
            if strike not in options_by_strike:
                options_by_strike[strike] = {'call': None, 'put': None}
            options_by_strike[strike]['put'] = put

        options_by_strike = [{'strike': float(k), 'call': v['call'], 'put': v['put']} for k, v in options_by_strike.items()]
        options_by_strike.sort(key=lambda x: x['strike'])

        options_data.append({
            'expirationDate': date,
            'options': options_by_strike
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

# Function to fetch additional stock information
def fetch_additional_info(ticker):
    stock = yf.Ticker(ticker)
    info = {
        'info': stock.info,
        'history': stock.history(period='1mo').reset_index().to_dict(orient='records'),
        'history_metadata': stock.history_metadata,
        'actions': stock.actions.reset_index().to_dict(orient='records'),
        'dividends': stock.dividends.reset_index().to_dict(orient='records'),
        'splits': stock.splits.reset_index().to_dict(orient='records'),
        'capital_gains': stock.capital_gains.reset_index().to_dict(orient='records'),
        'shares': stock.get_shares_full(start="2022-01-01", end=None),
        'income_stmt': stock.income_stmt.reset_index().to_dict(orient='records'),
        'quarterly_income_stmt': stock.quarterly_income_stmt.reset_index().to_dict(orient='records'),
        'balance_sheet': stock.balance_sheet.reset_index().to_dict(orient='records'),
        'quarterly_balance_sheet': stock.quarterly_balance_sheet.reset_index().to_dict(orient='records'),
        'cashflow': stock.cashflow.reset_index().to_dict(orient='records'),
        'quarterly_cashflow': stock.quarterly_cashflow.reset_index().to_dict(orient='records'),
        'major_holders': stock.major_holders.to_dict(),
        'institutional_holders': stock.institutional_holders.to_dict(orient='records'),
        'mutualfund_holders': stock.mutualfund_holders.to_dict(orient='records'),
        'insider_transactions': stock.insider_transactions.reset_index().to_dict(orient='records'),
        'insider_purchases': stock.insider_purchases.reset_index().to_dict(orient='records'),
        'insider_roster_holders': stock.insider_roster_holders.reset_index().to_dict(orient='records'),
        'recommendations': stock.recommendations.reset_index().to_dict(orient='records'),
        'recommendations_summary': stock.recommendations_summary.reset_index().to_dict(orient='records'),
        'upgrades_downgrades': stock.upgrades_downgrades.reset_index().to_dict(orient='records'),
        'earnings_dates': stock.earnings_dates.reset_index().to_dict(orient='records'),
        'isin': stock.isin,
        'options': stock.options,
        'news': stock.news
    }
    return info

# Load the pre-trained model and error standard deviation
model = tf.keras.models.load_model('lstm_model.keras')
train_error_std = np.load('train_error_std.npy')
test_error_std = np.load('test_error_std.npy')

# Create the Flask app
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Stock Prediction API"

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    ticker = request.json['ticker']
    scaled_df, scaler, original_df, current_price = fetch_and_preprocess(ticker)
    if scaled_df is None:
        return jsonify({'error': f"Ticker symbol '{ticker}' not found or insufficient data."}), 400

    seq_length = 60
    X = create_sequences(scaled_df.values, seq_length)
    last_sequence = X[-1]
    last_sequence_scaled = np.expand_dims(last_sequence, axis=0)
    prediction = model.predict(last_sequence_scaled)
    predicted_price = scaler.inverse_transform(np.array([[prediction[0][0], 0, 0, 0, 0, 0]]))[0][0]

    lower_bound = predicted_price - 1.96 * test_error_std
    upper_bound = predicted_price + 1.96 * test_error_std

    options_data = fetch_options_data(ticker)
    news_data = fetch_news_data(ticker)
    economic_events = fetch_economic_events()
    additional_info = fetch_additional_info(ticker)

    # Ensure datetime data is kept as strings
    def convert_timestamps(data):
        if isinstance(data, pd.Timestamp):
            return data.isoformat()
        elif isinstance(data, dict):
            return {str(k): convert_timestamps(v) for k, v in data.items()}
        elif isinstance(data, (list, np.ndarray, pd.Series, pd.Index, tuple)):
            return [convert_timestamps(i) for i in data]
        elif pd.isna(data):
            return ''
        return data

    return jsonify({
        'ticker': ticker.upper(),
        'current_price': round(current_price, 2),
        'prediction_text': round(predicted_price, 2),
        'prediction_range_upper': round(upper_bound, 2),
        'prediction_range_lower': round(lower_bound, 2),
        'options_data': convert_timestamps(options_data),
        'news_data': convert_timestamps(news_data),
        'economic_events': convert_timestamps(economic_events),
        'additional_info': convert_timestamps(additional_info)
    })

if __name__ == '__main__':
    app.run(port=8000, debug=True)
