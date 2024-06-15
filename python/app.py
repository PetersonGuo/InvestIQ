import ib_insync
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
import tensorflow as tf
import datetime
import asyncio
import aiohttp
import time
import requests
import json
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()

FRED_API_KEY = os.getenv('FRED_API_KEY')
FRED_SERIES_ID = os.getenv('FRED_SERIES_ID')

IBKR_HOST_IP = os.getenv('IBKR_HOST_IP')
IBKR_HOST_PORT = int(os.getenv('IBKR_HOST_PORT'))
IBKR_CURRENCY = os.getenv('IBKR_CURRENCY')
IBKR_STOCK_EXCHANGE = os.getenv('IBKR_STOCK_EXCHANGE')
IBKR_DURATION = '14400 S'  # 4 hours duration
IBKR_INTERVAL = '1 secs'
IBKR_WHAT_TO_SHOW = os.getenv('IBKR_WHAT_TO_SHOW')
IBKR_USE_RTH = bool(os.getenv('IBKR_USE_RTH'))

TICKER_JSON = os.getenv('TICKER_JSON')

# Connect to IBKR TWS or IB Gateway
ib = ib_insync.IB()
ib.connect(IBKR_HOST_IP, IBKR_HOST_PORT, clientId=1)

print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

# Load the pre-trained model and scaler
model = tf.keras.models.load_model('lstm_model.keras')
scalers = np.load('scalers.npy', allow_pickle=True).item()

# Function to fetch real-time stock data
def fetch_stock_data(ticker, end_date, interval='1 secs', duration='14400 S', retries=3, delay=60):
    all_data = []
    for attempt in range(retries):
        try:
            contract = ib_insync.Stock(ticker, IBKR_STOCK_EXCHANGE, IBKR_CURRENCY)
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_date.strftime('%Y%m%d %H:%M:%S'),
                durationStr=duration,
                barSizeSetting=interval,
                whatToShow=IBKR_WHAT_TO_SHOW,
                useRTH=IBKR_USE_RTH,
                formatDate=1)

            if bars:
                df = ib_insync.util.df(bars)
                df['symbol'] = ticker
                all_data.append(df)
                break
            else:
                print(f"No data found for {ticker}")
                return None

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    else:
        print(f"Failed to fetch data for {ticker} after {retries} attempts.")
        return None
    if not all_data:
        return None
    return pd.concat(all_data).sort_index()

# Asynchronous function to fetch interest rates
async def fetch_interest_rates(session, start_date, end_date):
    url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates"
    params = {
        "filter": f"record_date:gte:{start_date},record_date:lte:{end_date}",
        "sort": "record_date",
        "page[size]": 5000  # Adjust as needed
    }
    async with session.get(url, params=params) as response:
        if response.status == 200:
            data = await response.json()
            records = [(item['record_date'], float(item['avg_interest_rate_amt'])) for item in data['data']]
            df = pd.DataFrame(records, columns=['date', 'interest_rate'])
            df['date'] = pd.to_datetime(df['date'])
            return df.set_index('date')
        else:
            print(f"Error fetching interest rates: {response.status}")
            return pd.DataFrame()

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
    df = fetch_stock_data(ticker, end_date)
    if df is not None:
        df = calculate_technical_indicators(df)

        async with aiohttp.ClientSession() as session:
            start_date = (end_date - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            interest_rates_df = await fetch_interest_rates(session, start_date, end_date_str)

        df['date'] = df.index
        df = df.merge(interest_rates_df, left_on='date', right_index=True, how='left')

        features = ['open', 'high', 'low', 'close', 'volume', 'interest_rate', 'MA10', 'MA50', 'RSI', 'BB_Mid', 'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal', 'MACD_Hist', '%K', '%D', 'ADX', 'STD', 'Return']
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

    for ticker in ticker_symbols:
        print(f"Predicting for {ticker}")
        asyncio.run(predict_real_time(ticker))
