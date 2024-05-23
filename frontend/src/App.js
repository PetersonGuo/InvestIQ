import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const App = () => {
  const [ticker, setTicker] = useState('');
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedDate, setSelectedDate] = useState('');
  const [selectedOptions, setSelectedOptions] = useState(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setData(null);
      const response = await axios.post('http://localhost:8000/predict', { ticker });
      if (response.status !== 200) {
        throw new Error('An error occurred');
      }
      console.log(response.data);
      setData(response.data);
      setSelectedDate(response.data.options_data[0].expirationDate);
      setSelectedOptions(response.data.options_data[0]);
      setLoading(false);
      setError(null);
    } catch (err) {
      setError(err.response ? err.response.data.error : 'An error occurred');
      setData(null);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    fetchData();
  };

  const handleDateChange = (event) => {
    const date = event.target.value;
    setSelectedDate(date);
    const options = data.options_data.find(option => option.expirationDate === date);
    setSelectedOptions(options);
  };

  return (
    <div className="container">
      <h1>Stock Prediction App</h1>
      <form onSubmit={handleSubmit}>
        <div className='flex gap-2'>
          <input
            type="text"
            id="ticker"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            required
          />
          <button type="submit">Predict</button>
        </div>
      </form>
      <div className="mt-5 w-full flex items-center justify-center">
        {loading && <div className="lds-default"><div></div><div></div><div></div><div></div><div></div><div></div><div></div><div></div><div></div><div></div><div></div><div></div></div>}
      </div>
      {error && <div className="error">{error}</div>}
      {data && (
        <div className="data-container">
          <h2>{data.ticker}</h2>
          <p><strong>Current Price: {data.current_price}</strong></p>
          <p><strong>Predicted Next Day Close: {data.prediction_text}</strong></p>
          <p><strong>Predicted Day Close Range: {`${data.prediction_range_lower}-${data.prediction_range_upper}`}</strong></p>
          <h3>Options Data:</h3>
          <div>
            <div className='mx-auto flex justify-center items-center mb-2'>
              <label htmlFor="options-date">Select Expiration Date: </label>
              <select id="options-date" value={selectedDate} onChange={handleDateChange} className='px-2 py-1 ms-1 rounded-md'>
                {data.options_data.map(option => (
                  <option key={option.expirationDate} value={option.expirationDate}>
                    {option.expirationDate}
                  </option>
                ))}
              </select>
            </div>
            <div className='flex'>
              <table border="0">
                <thead>
                  <tr>
                    <th colSpan={6}>Calls</th>
                    <th></th>
                    <th colSpan={6}>Puts</th>
                  </tr>
                  <tr>
                    <th>Last Price</th>
                    <th>Bid</th>
                    <th>Ask</th>
                    <th>Change</th>
                    <th>Volume</th>
                    <th>Open Interest</th>
                    <th>Strike</th>
                    <th>Last Price</th>
                    <th>Bid</th>
                    <th>Ask</th>
                    <th>Change</th>
                    <th>Volume</th>
                    <th>Open Interest</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedOptions.options.map((option, idx) => (
                    <tr key={`option-${idx}`}>
                      <td className={`${option.strike <= data.current_price ? 'bg-blue-400' : ''}`}>{option.call.lastPrice}</td>
                      <td className={`${option.strike <= data.current_price ? 'bg-blue-400' : ''}`}>{option.call.bid}</td>
                      <td className={`${option.strike <= data.current_price ? 'bg-blue-400' : ''}`}>{option.call.ask}</td>
                      <td className={`${option.strike <= data.current_price ? 'bg-blue-400' : ''}`}>{option.call.change}</td>
                      <td className={`${option.strike <= data.current_price ? 'bg-blue-400' : ''}`}>{option.call.volume}</td>
                      <td className={`${option.strike <= data.current_price ? 'bg-blue-400' : ''}`}>{option.call.openInterest}</td>
                      <td>{option.strike}</td>
                      <td className={`${option.strike >= data.current_price ? 'bg-blue-400' : ''}`}>{option.put.lastPrice}</td>
                      <td className={`${option.strike >= data.current_price ? 'bg-blue-400' : ''}`}>{option.put.bid}</td>
                      <td className={`${option.strike >= data.current_price ? 'bg-blue-400' : ''}`}>{option.put.ask}</td>
                      <td className={`${option.strike >= data.current_price ? 'bg-blue-400' : ''}`}>{option.put.change}</td>
                      <td className={`${option.strike >= data.current_price ? 'bg-blue-400' : ''}`}>{option.put.volume}</td>
                      <td className={`${option.strike >= data.current_price ? 'bg-blue-400' : ''}`}>{option.put.openInterest}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <h3>News Data:</h3>
          <ul>
            {data.news_data.map((article, idx) => (
              <li key={idx}>
                <a href={article.link} target="_blank" rel="noopener noreferrer">{article.title}</a><br />
                {article.publisher} - {new Date(article.date).toLocaleDateString()}<br />
                Sentiment: {article.sentiment > 0 ? 'Positive' : article.sentiment < 0 ? 'Negative' : 'Neutral'}
              </li>
            ))}
          </ul>
          <h3>Economic Events:</h3>
          <ul>
            {data.economic_events.map((event, idx) => (
              <li key={idx}>
                {event.date} - {event.event}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default App;
