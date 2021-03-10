import pandas as pd
import numpy as np
import pandas_datareader as pdr
from sklearn import preprocessing

gain = lambda x: x if x > 0 else 0 # works as a map function or in list comprehension
loss = lambda x: abs(x) if x < 0 else 0 # works as a map function or in list comprehension

def calc_fourier_transform(sentiment_df):
    close_fft = np.fft.fft(np.asarray(sentiment_df['score'].tolist()))
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    fft_list = np.asarray(fft_df['fft'].tolist())

    for num_ in [5, 10, 15, 20]:
        fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
        sentiment_df['fourier '+str(num_)]=np.fft.ifft(fft_list_m10)
    
    return sentiment_df

def get_ticker_data(start, end, ticker):
    """
    Get historical OHLC data for given date range and ticker.
    Tries to get from Investors Exchange (IEX), but falls back
    to Yahoo! Finance if IEX doesn't have it.

    Parameter:
        - ticker: The stock symbol to lookup as a string.

    Returns:
        A pandas dataframe with the stock data.
    """
    try:
        print('Getting ticker historical data from IEX')
        data = pdr.DataReader(ticker, 'iex', start, end)
        data.index = pd.to_datetime(data.index)
    except:
        print('Getting ticker historical data from Yahoo')
        data = pdr.get_data_yahoo(ticker, start, end)
    return data 

def calc_bollinger_bands(stock, window=14):
    rolling_mean = stock.Close.rolling(window).mean()
    rolling_std = stock.Close.rolling(window).std()
    rolling_mean_s = stock.sentiment.rolling(window).mean()
    rolling_std_s = stock.sentiment.rolling(window).std()
    upper_band = rolling_mean + (rolling_std*2)
    lower_band = rolling_mean - (rolling_std*2)
    upper_band_s = rolling_mean_s + (rolling_std_s*2)
    lower_band_s = rolling_mean_s - (rolling_std_s*2)
    return upper_band, lower_band, upper_band_s, lower_band_s

def calc_rsi_ewma_sma(close, window_length):
    delta = close.diff()
    delta = delta[1:] 
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.ewm(span=window_length).mean()
    roll_down1 = down.abs().ewm(span=window_length).mean()
    RS1 = roll_up1 / roll_down1
    RSI_EWMA = 100.0 - (100.0 / (1.0 + RS1))
    roll_up2 = up.rolling(window_length).mean()
    roll_down2 = down.abs().rolling(window_length).mean()   
    RS2 = roll_up2 / roll_down2
    RSI_SMA = 100.0 - (100.0 / (1.0 + RS2))
    return RSI_EWMA, RSI_SMA