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
    dfs = []
    for symbol in ticker_symbols:
        # Fetch historical stock data
        df = yf.download(symbol, start=start_date, end=end_date)

        # Preprocess data (calculate additional features, handle missing values, etc.)
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(window=14).mean()))
        df['Return'] = df['Close'].pct_change()
        df = df.dropna()

        # Add a column to identify the ticker symbol
        df['Symbol'] = symbol

        dfs.append(df.copy())  # Create a copy of the dataframe

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
        sequences.append(data[i:i+seq_length])
        targets.append(target[i+seq_length])
    return np.array(sequences), np.array(targets)

# Split data into training and testing sets
def split_data(X, y, split_ratio=0.8):
    split = int(split_ratio * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test

# Define a list of ticker symbols
ticker_symbols = [
    "MSFT", "AAPL", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "BRK.B", "LLY", "AVGO",
    "JPM", "TSLA", "V", "XOM", "WMT", "UNH", "MA", "PG", "JNJ", "COST", "HD", "ORCL",
    "MRK", "BAC", "CVX", "ABBV", "CRM", "KO", "NFLX", "AMD", "PEP", "TMO", "ADBE",
    "QCOM", "WFC", "LIN", "DHR", "MCD", "CSCO", "TMUS", "ACN", "DIS", "INTU", "ABT",
    "TXN", "AMAT", "GE", "AXP", "CAT", "VZ", "AMGN", "MS", "PFE", "NOW", "NEE", "IBM",
    "PM", "CMCSA", "BX", "GS", "UNP", "COP", "ISRG", "SCHW", "NKE", "MU", "RTX", "SPGI",
    "UBER", "INTC", "HON", "LOW", "ETN", "UPS", "SYK", "ELV", "BKNG", "T", "PGR", "C",
    "BLK", "LRCX", "VRTX", "MDT", "TJX", "BA", "LMT", "CB", "DE", "BSX", "REGN", "ADI",
    "MMC", "ADP", "PLD", "PANW", "KLAC", "ANET", "CI", "MDLZ", "ABNB", "AMT", "FI", "BMY",
    "CMG", "SBUX", "SO", "SNPS", "HCA", "WM", "GILD", "GD", "DUK", "ZTS", "ICE", "APH",
    "SHW", "MO", "CDNS", "FCX", "CL", "CME", "EQIX", "MCO", "ITW", "EOG", "TT", "TGT",
    "MCK", "CVS", "TDG", "CTAS", "PH", "NOC", "SLB", "NXPI", "BDX", "MAR", "PYPL", "CEG",
    "ECL", "CSX", "USB", "EMR", "PNC", "AON", "FDX", "MPC", "PSX", "MSI", "WELL", "ORLY",
    "RSG", "CARR", "APD", "MMM", "ROP", "MNST", "AJG", "OXY", "PCAR", "VLO", "COF", "BLK",
    "EW", "TFC", "AIG", "MET", "CPRT", "NSC", "DXCM", "BDX", "ICE", "APH", "SHW", "MO",
    "CDNS", "FCX", "CL", "CME", "EQIX", "MCO", "ITW", "EOG", "TT", "TGT", "MCK", "CVS",
    "TDG", "CTAS", "PH", "NOC", "SLB", "NXPI", "BDX", "MAR", "PYPL", "CEG", "ECL", "CSX",
    "USB", "EMR", "PNC", "AON", "FDX", "MPC", "PSX", "MSI", "WELL", "ORLY", "RSG", "CARR",
    "APD", "MMM", "ROP", "MNST", "AJG", "OXY", "PCAR", "VLO", "COF", "AFL", "DHI", "SRE",
    "AEP", "HES", "SPG", "EL", "OKE", "F", "O", "ADSK", "FTNT", "STZ", "JCI", "DLR", "GWW",
    "TEL", "LEN", "KDP", "URI", "PAYX", "KMB", "A", "IDXX", "D", "ALL", "CCI", "GEV", "BK",
    "ROST", "COR", "KMI", "KHC", "FIS", "PRU", "AMP", "HUM", "LHX", "IQV", "HSY", "CNC",
    "MSA", "BLL", "ED", "GIS", "LUV", "FAST", "CERN", "WEC", "RMD", "WBA", "WY", "APH",
    "NLOK", "HRL", "DAL", "TROW", "KEYS", "CMI", "VFC", "CMS", "LULU", "BAX", "ZBRA",
    "PXD", "BR", "SYY", "PHM", "RCL", "MTB", "EXC", "PKI", "TTWO", "VTR", "ODFL", "HAL",
    "AFL", "DHI", "SRE", "AEP", "HES", "SPG", "EL", "OKE", "F", "O", "ADSK", "FTNT", "STZ",
    "JCI", "DLR", "GWW", "TEL", "LEN", "KDP", "URI", "PAYX", "KMB", "A", "IDXX", "D", "ALL",
    "CCI", "GEV", "BK", "ROST", "COR", "KMI", "KHC", "FIS", "PRU", "AMP", "HUM", "LHX", "IQV",
    "HSY", "CNC", "MSA", "BLL", "ED", "GIS", "LUV", "FAST", "CERN", "WEC", "RMD", "WBA",
    "WY", "APH", "NLOK", "HRL", "DAL", "TROW", "KEYS", "CMI", "VFC", "CMS", "LULU", "BAX",
    "ZBRA", "PXD", "BR", "SYY", "PHM", "RCL", "MTB", "EXC", "PKI", "TTWO", "VTR", "ODFL",
    "HAL", "LH", "WLTW", "DG", "AMP", "WMB", "MXIM", "CNP", "ESS", "J", "CAG", "HSIC",
    "VRSK", "EOC", "MGM", "XLNX", "AJRD", "FFIV", "ZION", "BKR", "ULTA", "AIZ", "LYB",
    "TYL", "MCHP", "CCL", "LNC", "MNST", "HLT", "DTE", "FTV", "AME", "COO", "WYNN", "XRAY",
    "UAL", "TT", "EDU", "CFG", "CERN", "WEC", "RMD", "WBA", "WY", "APH", "NLOK", "HRL",
    "DAL", "TROW", "KEYS", "CMI", "VFC", "CMS", "LULU", "BAX", "ZBRA", "PXD", "BR", "SYY",
    "PHM", "RCL", "MTB", "EXC", "PKI", "TTWO", "VTR", "ODFL", "HAL", "LH", "WLTW", "DG",
    "AMP", "WMB", "MXIM", "CNP", "ESS", "J", "CAG", "HSIC", "VRSK", "EOC", "MGM", "XLNX",
    "AJRD", "FFIV", "ZION", "BKR", "ULTA", "AIZ", "LYB", "TYL", "MCHP", "CCL", "LNC",
    "MNST", "HLT", "DTE", "FTV", "AME", "COO", "WYNN", "XRAY", "UAL", "TT", "EDU", "CFG"
]

# Fetch historical data for multiple ticker symbols
df = fetch_data(ticker_symbols, '2010-01-01', datetime.datetime.today().date())

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
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

    # Concatenate sequences for all ticker symbols
    X_train = np.concatenate(X_train_list)
    X_test = np.concatenate(X_test_list)
    y_train = np.concatenate(y_train_list)
    y_test = np.concatenate(y_test_list)

    # Define input shape
    input_shape = X_train.shape[1:]

    # Create the input layer
    input_layer = Input(shape=input_shape)

    # Build the LSTM model
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
