import ib_insync
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from tensorflow.keras.models import Model # type: ignore
import tensorflow as tf
import datetime
from tqdm import tqdm
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
IBKR_DURATION = os.getenv('IBKR_DURATION')
IBKR_INTERVAL = os.getenv('IBKR_INTERVAL')
IBKR_WHAT_TO_SHOW = os.getenv('IBKR_WHAT_TO_SHOW')
IBKR_USE_RTH = bool(os.getenv('IBKR_USE_RTH'))

TICKER_JSON = os.getenv('TICKER_JSON')

# Interactive Brokers API setup
ib = ib_insync.IB()
ib.connect(IBKR_HOST_IP, IBKR_HOST_PORT, clientId=1)  # TWS paper trading port

print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

# Function to fetch intraday data from IBKR
def fetch_ibkr_data(ticker, end_date, whatToShow):
    contract = ib_insync.Stock(ticker, IBKR_STOCK_EXCHANGE, IBKR_CURRENCY)
    ib.qualifyContracts(contract)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_date.strftime('%Y%m%d %H:%M:%S'),
        durationStr=IBKR_DURATION,
        barSizeSetting=IBKR_INTERVAL,
        whatToShow=whatToShow,
        useRTH=IBKR_USE_RTH
    )

    if not bars:
        return pd.DataFrame()

    data = [{'date': bar.date, 'open': bar.open, 'high': bar.high, 'low': bar.low, 'close': bar.close, 'volume': bar.volume, 'symbol': ticker} for bar in bars]

    df = pd.DataFrame(data)
    return df

def fetch_interest_rates(start_date, end_date):
    base_url = f'https://api.stlouisfed.org/fred/series/observations'

    # Prepare the query parameters
    params = {
        'series_id': FRED_SERIES_ID,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'observation_start': start_date.strftime('%Y-%m-%d'),
        'observation_end': end_date.strftime('%Y-%m-%d')
    }

    # Make the request to the API
    response = requests.get(base_url, params=params)
    response.raise_for_status()  # Raise an error for bad responses

    # Parse the JSON response
    data = response.json()['observations']

    # Extract dates and rates into lists
    records = [{'date': datetime.datetime.strptime(item['date'], "%Y-%m-%d"), 'fed_rate': float(item['value'])} for item in data]

    # Create a DataFrame
    df = pd.DataFrame(records)
    return df

# Calculate additional technical indicators
def calculate_additional_indicators(df):
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

    return df

# Normalize the data
def normalize_data(df, symbol):
    scaler = MinMaxScaler()
    symbol_data = df[df['symbol'] == symbol].drop(['symbol', 'date'], axis=1)
    scaled_data = scaler.fit_transform(symbol_data)
    scaled_df = pd.DataFrame(scaled_data, columns=symbol_data.columns, index=symbol_data.index)
    scaled_df['symbol'] = symbol
    scaled_df['date'] = df[df['symbol'] == symbol]['date']
    return scaled_df, scaler

# Create sequences for LSTM
def create_sequences(data, target, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(target[i + seq_length])
    return np.array(sequences), np.array(targets)

# Split data into training and testing sets
def split_data(X, y, split_ratio=0.8):
    split = int(split_ratio * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test

# Define a list of ticker symbols
with open(TICKER_JSON, 'r') as f:
    ticker_symbols = json.load(f)

# Fetch historical data for multiple ticker symbols
end_date = datetime.datetime.now() - datetime.timedelta(minutes=15)
start_date = (end_date - datetime.timedelta(days=365))

all_data = []
for symbol in tqdm(ticker_symbols, desc="Fetching data"):
    for i in range(365):
        end_date = end_date - datetime.timedelta(days=1)
        to_concate = pd.DataFrame()
        for whatToShow in IBKR_WHAT_TO_SHOW.split():
            subset_df = fetch_ibkr_data(symbol, end_date, whatToShow)
            if not subset_df.empty:
                if to_concate.empty:
                    to_concate = subset_df
                else:
                    to_concate = pd.merge(to_concate, subset_df, how='outer', on='date, symbol')
            else:
                print(f"No data found for {symbol}")
        if not to_concate.empty:
            all_data.append(to_concate)

df = pd.concat(all_data) if all_data else None

if df is not None:
    # Fetch additional economic data (Fed rates)
    fed_rates_df = fetch_interest_rates(start_date, end_date)
    fed_rates_df['date'] = pd.to_datetime(fed_rates_df['date'])
    fed_rates_df['date'] = fed_rates_df['date'].dt.tz_localize(None)
    fed_rates_df.set_index('date', inplace=True)
    fed_rates_df.sort_values('date', inplace=True)

    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.tz_localize(None)
    df.set_index('date', inplace=True)
    df.sort_values('date', inplace=True)

    # Merge market data with economic data
    df = pd.merge_asof(df, fed_rates_df, on='date', direction='backward')

    # Calculate additional technical indicators
    df = calculate_additional_indicators(df)

    # Define features and target variable
    features = ['open', 'high', 'low', 'close', 'volume', 'fed_rate', 'BB_Mid', 'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal', 'MACD_Hist', '%K', '%D', 'ADX', 'STD']
    target = 'close'

    # Normalize the data and create sequences for each symbol
    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
    scalers = {}
    for symbol in ticker_symbols:
        if symbol in df['symbol'].values:
            df_normalized, scaler = normalize_data(df, symbol)
            scalers[symbol] = scaler
            symbol_data = df_normalized.drop('symbol', axis=1)
            X_symbol, y_symbol = create_sequences(symbol_data[features].values, symbol_data[target].values, seq_length=60)
            X_train, X_test, y_train, y_test = split_data(X_symbol, y_symbol)
            X_train_list.append(X_train)
            X_test_list.append(X_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)
        else:
            print(f"Skipping {symbol} due to insufficient data after scaling")

    # Concatenate all sequences
    X_train = np.concatenate(X_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # Define input shape
    input_shape = X_train.shape[1:]

    # Build the LSTM model
    input_layer = Input(shape=input_shape)
    lstm_layer1 = LSTM(50, return_sequences=True)(input_layer)
    dropout_layer1 = Dropout(0.2)(lstm_layer1)
    lstm_layer2 = LSTM(50, return_sequences=False)(dropout_layer1)
    dropout_layer2 = Dropout(0.2)(lstm_layer2)
    output_layer = Dense(1)(dropout_layer2)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

    # Evaluate the model to calculate prediction errors
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate training and testing errors
    train_errors = y_train - y_train_pred.flatten()
    test_errors = y_test - y_test_pred.flatten()

    # Calculate standard deviation of errors
    train_error_std = np.std(train_errors)
    test_error_std = np.std(test_errors)

    # Save the model and the standard deviation of errors
    model.save('lstm_model.keras')
    np.save('scalers.npy', scalers)
    np.save('train_error_std.npy', train_error_std)
    np.save('test_error_std.npy', test_error_std)

    # Logging predictions for inspection
    predictions = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_test_pred.flatten()
    })
    predictions.to_csv('predictions.csv', index=False)
else:
    print("No data available for the given date range and ticker symbols.")
