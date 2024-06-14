import blpapi
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Model
import tensorflow as tf
import datetime
from tqdm import tqdm

# Bloomberg API setup
SESSION_OPTIONS = blpapi.SessionOptions()
SESSION_OPTIONS.setServerHost('localhost')
SESSION_OPTIONS.setServerPort(8194)

session = blpapi.Session(SESSION_OPTIONS)
if not session.start():
    print("Failed to start Bloomberg session.")
    exit()

if not session.openService("//blp/refdata"):
    print("Failed to open Bloomberg service.")
    exit()

service = session.getService("//blp/refdata")

print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

# Function to fetch intraday data and technical indicators from Bloomberg
def fetch_bloomberg_data(ticker, start_date, end_date):
    request = service.createRequest("IntradayBarRequest")
    request.getElement("securities").appendValue(ticker)
    request.set("startDateTime", start_date)
    request.set("endDateTime", end_date)
    request.set("eventType", "TRADE")
    request.set("interval", 1)  # 1-minute intervals

    session.sendRequest(request)

    data = []
    while True:
        ev = session.nextEvent()
        for msg in ev:
            if msg.hasElement("barData"):
                bars = msg.getElement("barData").getElement("barTickData")
                for bar in bars.values():
                    data.append({
                        'date': bar.getElementAsDatetime("time"),
                        'open': bar.getElementAsFloat("open"),
                        'high': bar.getElementAsFloat("high"),
                        'low': bar.getElementAsFloat("low"),
                        'close': bar.getElementAsFloat("close"),
                        'volume': bar.getElementAsFloat("volume")
                    })
        if ev.eventType() == blpapi.Event.RESPONSE:
            break

    return pd.DataFrame(data)

# Function to fetch Fed rates from Bloomberg
def fetch_fed_rates(start_date, end_date):
    request = service.createRequest("HistoricalDataRequest")
    request.getElement("securities").appendValue("FDTR Index")  # Bloomberg ticker for US Fed Funds Rate
    request.getElement("fields").appendValue("PX_LAST")
    request.set("startDate", start_date)
    request.set("endDate", end_date)
    request.set("periodicitySelection", "DAILY")

    session.sendRequest(request)

    data = []
    while True:
        ev = session.nextEvent()
        for msg in ev:
            if msg.hasElement("securityData"):
                sd = msg.getElement("securityData")
                fs = sd.getElement("fieldData")
                for i in range(fs.numValues()):
                    f = fs.getValue(i)
                    data.append({
                        'date': f.getElementAsDatetime("date"),
                        'fed_rate': f.getElementAsFloat("PX_LAST")
                    })
        if ev.eventType() == blpapi.Event.RESPONSE:
            break

    return pd.DataFrame(data)

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
with open('ticker_symbols.json', 'r') as f:
    ticker_symbols = json.load(f)

# Fetch historical data for multiple ticker symbols
end_date = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%dT%H:%M:%S')
all_data = []
for symbol in tqdm(ticker_symbols, desc="Fetching data"):
    df = fetch_bloomberg_data(symbol, start_date, end_date)
    if not df.empty:
        df['symbol'] = symbol
        all_data.append(df)
    else:
        print(f"No data found for {symbol}")

if all_data:
    df = pd.concat(all_data)
else:
    df = None

if df is not None:
    # Fetch additional economic data (Fed rates)
    fed_rates_df = fetch_fed_rates(start_date, end_date)

    # Merge market data with economic data
    df['date'] = pd.to_datetime(df['date'])
    df = df.merge(fed_rates_df, left_on='date', right_on='date', how='left')

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
    model.save('lstm_model_bloomberg.keras')
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
