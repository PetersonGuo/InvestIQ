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
    # Join ticker symbols into a single string separated by spaces
    ticker_string = ' '.join(ticker_symbols)

    # Fetch historical stock data for all ticker symbols simultaneously
    df = yf.download(ticker_string, start=start_date, end=end_date, group_by='tickers')

    # Add additional features, handle missing values, etc.
    dfs = []
    for symbol in ticker_symbols:
        symbol_data = df[symbol].copy()
        # symbol_data = symbol_data.to_frame()  # Convert Series to DataFrame
        symbol_data['MA10'] = symbol_data['Adj Close'].rolling(window=10).mean()
        symbol_data['MA50'] = symbol_data['Adj Close'].rolling(window=50).mean()
        symbol_data['RSI'] = 100 - (100 / (1 + symbol_data['Adj Close'].pct_change().rolling(window=14).mean()))
        symbol_data['Return'] = symbol_data['Adj Close'].pct_change()
        symbol_data['Symbol'] = symbol
        symbol_data = symbol_data.dropna()
        dfs.append(symbol_data)

    if len(dfs) > 0:
        return pd.concat(dfs)
    else:
        return None


# Normalize the data
def normalize_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.iloc[:, :-1])  # Exclude the 'Symbol' column
    df.iloc[:, :-1] = scaled_data
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
df = fetch_data(ticker_symbols, '1970-01-01', datetime.datetime.today().date())

if df is not None:
    # Define features and target variable
    features = ['Close', 'MA10', 'MA50', 'RSI', 'Return']
    target = 'Close'

    # Normalize the data
    scaled_df, scaler = normalize_data(df[features + ['Symbol']])  # Include the 'Symbol' column

    # Create sequences for LSTM for each ticker symbol
    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
    for symbol in ticker_symbols:
        symbol_data = scaled_df[scaled_df['Symbol'] == symbol].drop('Symbol', axis=1)
        X_symbol, y_symbol = create_sequences(symbol_data[features].values, symbol_data[target].values, seq_length=60)
        X_train, X_test, y_train, y_test = split_data(X_symbol, y_symbol)
        X_train_list.extend(X_train)
        X_test_list.extend(X_test)
        y_train_list.extend(y_train)
        y_test_list.extend(y_test)

    # Convert lists to numpy arrays
    X_train = np.array(X_train_list)
    X_test = np.array(X_test_list)
    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)

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

    # Calculate training errors for each ticker symbol
    train_errors = []
    for i in range(len(y_train)):
        train_errors.append(y_train[i] - y_train_pred[i].flatten())

    # Convert train_errors to numpy array
    train_errors = np.array(train_errors)

    # Calculate standard deviation of errors
    error_std = np.std(train_errors)

    # Save the model and the standard deviation of errors
    model.save('lstm_model.keras')
    np.save('error_std.npy', error_std)
