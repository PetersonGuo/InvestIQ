import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import datetime
import asyncio
import aiohttp
from ib_insync import *
import time

# Connect to IBKR TWS or IB Gateway
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

# Load the pre-trained model and scaler
model = tf.keras.models.load_model('lstm_model_ibkr.keras')
scalers = np.load('scalers.npy', allow_pickle=True).item()

# Function to fetch real-time stock data
async def fetch_stock_data(ticker, end_date, interval='1 secs', duration='1800 S', retries=3, delay=60):
    all_data = []
    current_date = end_date
    for _ in range(8):  # 2 hours = 4 chunks of 30 minutes
        for attempt in range(retries):
            try:
                contract = Stock(ticker, 'SMART', 'USD')
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=current_date.strftime('%Y%m%d %H:%M:%S'),
                    durationStr=duration,
                    barSizeSetting=interval,
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1)

                if bars:
                    df = util.df(bars)
                    df['symbol'] = ticker
                    all_data.append(df)
                    current_date -= datetime.timedelta(minutes=30)
                    await asyncio.sleep(REQUEST_INTERVAL)
                    break
                else:
                    print(f"No data found for {ticker}")
                    return None

            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
        else:
            print(f"Failed to fetch data for {ticker} after {retries} attempts.")
            return None
    if not all_data:
        return None
    return pd.concat(all_data).sort_index()

# Calculate technical indicators
def calculate_technical_indicators(df):
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Mid'] = df['close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['close'].rolling(window=20).std()

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Stochastic Oscillator
    low14 = df['low'].rolling(window=14).min()
    high14 = df['high'].rolling(window=14).max()
    df['%K'] = 100 * ((df['close'] - low14) / (high14 - low14))
    df['%D'] = df['%K'].rolling(window=3).mean()

    # ADX
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    tr1 = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    atr = tr1.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.rolling(window=14).mean()

    # Standard Deviation
    df['STD'] = df['close'].rolling(window=20).std()

    df['Return'] = df['close'].pct_change()
    return df

# Normalize the data
def normalize_data(df, scaler, features):
    symbol_data = df[features].drop(['symbol', 'date'], axis=1)
    scaled_data = scaler.transform(symbol_data)
    scaled_df = pd.DataFrame(scaled_data, columns=symbol_data.columns, index=symbol_data.index)
    return scaled_df

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

# Main function to fetch data, process it, and make predictions
async def predict_real_time(ticker):
    end_date = datetime.datetime.now()
    df = await fetch_stock_data(ticker, end_date)
    if df is not None:
        df = calculate_technical_indicators(df)

        features = ['open', 'high', 'low', 'close', 'volume', 'MA10', 'MA50', 'RSI', 'BB_Mid', 'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal', 'MACD_Hist', '%K', '%D', 'ADX', 'STD', 'Return']
        scaler = scalers[ticker]
        df_normalized = normalize_data(df, scaler, features)

        seq_length = 60
        X = create_sequences(df_normalized.values, seq_length)

        predictions = model.predict(X)

        df['Predicted_Close'] = np.nan
        df.iloc[seq_length:, df.columns.get_loc('Predicted_Close')] = predictions.flatten()

        print(df[['close', 'Predicted_Close']].tail(10))

if __name__ == '__main__':
    with open('ticker_symbols.json', 'r') as f:
        ticker_symbols = json.load(f)
    
    ticker = input("Enter a ticker symbol: ")
    print(f"Predicting for {ticker}")
    asyncio.run(predict_real_time(ticker))
