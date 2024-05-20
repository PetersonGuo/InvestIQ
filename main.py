import json
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Model
import tensorflow as tf
import datetime

print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

# Fetch data for multiple ticker symbols
def fetch_data(ticker_symbols, start_date, end_date):
    ticker_string = ' '.join(ticker_symbols)
    df = yf.download(ticker_string, start=start_date, end=end_date, group_by='tickers')

    dfs = []
    for symbol in ticker_symbols:
        if symbol in df.columns.levels[0]:
            symbol_data = df[symbol].copy()
            symbol_data['MA10'] = symbol_data['Adj Close'].rolling(window=10).mean()
            symbol_data['MA50'] = symbol_data['Adj Close'].rolling(window=50).mean()
            symbol_data['RSI'] = 100 - (100 / (1 + symbol_data['Adj Close'].pct_change().rolling(window=14).mean()))
            symbol_data['Return'] = symbol_data['Adj Close'].pct_change()
            symbol_data['Symbol'] = symbol
            symbol_data = symbol_data.dropna()
            if not symbol_data.empty:
                dfs.append(symbol_data)
            else:
                print(f"Skipping {symbol} due to insufficient data after preprocessing")
        else:
            print(f"Skipping {symbol} due to lack of data in the given date range")

    if len(dfs) > 0:
        return pd.concat(dfs)
    else:
        return None

# Normalize the data
def normalize_data(df, symbol):
    scaler = MinMaxScaler()
    symbol_data = df[df['Symbol'] == symbol].drop('Symbol', axis=1)
    scaled_data = scaler.fit_transform(symbol_data)
    df.loc[df['Symbol'] == symbol, symbol_data.columns] = scaled_data
    return df, scaler

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
df = fetch_data(ticker_symbols, '2020-01-01', datetime.datetime.today().date())

if df is not None:
    # Define features and target variable
    features = ['Adj Close', 'MA10', 'MA50', 'RSI', 'Return', 'Volume']
    target = 'Adj Close'

    # Normalize the data and create sequences for each symbol
    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
    scalers = {}
    for symbol in ticker_symbols:
        if symbol in df['Symbol'].values:
            df, scaler = normalize_data(df, symbol)
            scalers[symbol] = scaler
            symbol_data = df[df['Symbol'] == symbol].drop('Symbol', axis=1)
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
