# Stock Price Prediction App

This project is a web application that predicts stock prices using a Long Short-Term Memory (LSTM) neural network model. The application allows users to input a stock ticker symbol and receive a prediction for the next day's closing price, along with a 95% prediction interval. It also provides options data, recent news related to the stock, and upcoming economic events.

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/PetersonGuo/StockPricePredictor
    ```

2. Install the required Python packages:

    MacOS:
    ```
    pip install -r requirements-macos.txt
    ```

    Windows/Linux:
    ```
    pip install -r requirements.txt
    ```

3. Run the Flask app:

    ```
    python app.py
    ```

4. Open a web browser and go to `http://localhost:5000` to access the application.

## Usage

1. Enter a stock ticker symbol in the input field.
2. Click the "Predict" button.
3. View the predicted next day's closing price and the 95% prediction interval.
4. Explore options data, recent news, and upcoming economic events related to the stock.

## Model Training

1. The LSTM model is trained using historical stock data fetched from Yahoo Finance.
2. Additional features such as moving averages and relative strength index (RSI) are calculated and used for training.
3. The model is trained on multiple ticker symbols to improve prediction accuracy.
4. GPU acceleration is used if available to speed up model training.

## Files

- `app.py`: Flask web application code.
- `lstm_model.py`: Script for training the LSTM model.
- `templates/index.html`: HTML template for the web interface.
- `requirements.txt`: List of Python packages required for the project.

## Credits

- This project uses [TensorFlow](https://www.tensorflow.org/) for deep learning.
- Stock data is fetched from [Yahoo Finance](https://finance.yahoo.com/).
- Options data is obtained from [Interactive Brokers](https://www.interactivebrokers.com/).
- News data is retrieved using the [TextBlob](https://textblob.readthedocs.io/en/dev/) library for sentiment analysis.
- Economic events data is for demonstration purposes and is not real-time.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
