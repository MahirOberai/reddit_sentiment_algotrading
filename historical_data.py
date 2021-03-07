#import yfinance as yf
#from broadwaybet import full_index
import pandas as pd
import pandas_datareader as pdr
import datetime 
import pytz
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler
#from broadwaybet import start_time, end_time
#from sklearn.preprocessing import StandardScaler
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import train_test_split
#from sklearn import metrics

#start_time = 01

start = '01-24-2020'
end = '01-24-2021'

sentiment_df = pd.read_csv('test_quant.csv')
sentiment_df.index = pd.to_datetime(sentiment_df['date'])
sentiment_df.drop(columns=['date'])

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

tsla_historicals = get_ticker_data(start, end, 'TSLA')

tsla_historicals['2_SMA'] = tsla_historicals['Close'].rolling(window=2).mean()
tsla_historicals['5_SMA'] = tsla_historicals['Close'].rolling(window=5).mean()

tsla_historicals = tsla_historicals[tsla_historicals['5_SMA'].notna()]

sentiment_df = sentiment_df[sentiment_df.index.isin(tsla_historicals.index)]

tsla_historicals = tsla_historicals[tsla_historicals.index.isin(tsla_historicals.index.intersection(sentiment_df.index))]

tsla_historicals.to_csv('tsla_historicals_test.csv')
sentiment_df.to_csv('sentiment_df_test.csv')

print(len(tsla_historicals))
print(len(sentiment_df))

close_fft = np.fft.fft(np.asarray(sentiment_df['score'].tolist()))
fft_df = pd.DataFrame({'fft':close_fft})
fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
fft_list = np.asarray(fft_df['fft'].tolist())

for num_ in [5, 10, 15, 20]:
    fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
    sentiment_df['fourier '+str(num_)]=np.fft.ifft(fft_list_m10)

sc= MinMaxScaler(feature_range=(0,1))
norm_df = pd.DataFrame(index=tsla_historicals.index)
norm_df['norm_price']=sc.fit_transform(tsla_historicals['Close'].to_numpy().reshape(-1, 1))
norm_df['price_log']=np.log(tsla_historicals['Close']/tsla_historicals['Close'].shift(1))
norm_df['norm_sentiment']=sc.fit_transform(sentiment_df['score'].to_numpy().reshape(-1, 1))
norm_df['norm_fourier5']=sc.fit_transform(np.asarray(list([(float(x)) for x in sentiment_df['fourier 5'].to_numpy()])).reshape(-1, 1))
norm_df['norm_fourier10']=sc.fit_transform(np.asarray(list([(float(x)) for x in sentiment_df['fourier 10'].to_numpy()])).reshape(-1, 1))
norm_df['norm_fourier15']=sc.fit_transform(np.asarray(list([(float(x)) for x in sentiment_df['fourier 15'].to_numpy()])).reshape(-1, 1))
norm_df['norm_fourier20']=sc.fit_transform(np.asarray(list([(float(x)) for x in sentiment_df['fourier 20'].to_numpy()])).reshape(-1, 1))

# SMA trade calls
sma_buys=[]
sma_sells=[]
for i in range(len(tsla_historicals)-1):
    if ((tsla_historicals['2_SMA'].values[i] < tsla_historicals['5_SMA'].values[i]) 
    & (tsla_historicals['2_SMA'].values[i+1] > tsla_historicals['5_SMA'].values[i+1])):
        #print("Trade Call for {row} is Buy.".format(row=tsla_historicals.index[i].date()))
        sma_buys.append(i)
    elif ((tsla_historicals['2_SMA'].values[i] > tsla_historicals['5_SMA'].values[i]) & (tsla_historicals['2_SMA'].values[i+1] < tsla_historicals['5_SMA'].values[i+1])):
        #print("Trade Call for {row} is Sell.".format(row=tsla_historicals.index[i].date()))
        sma_sells.append(i)

sma_profit_gross = 0
for i in range(len(sma_buys)):
    sma_profit_gross += sma_buys[i] - sma_sells[i]

#profit_gross = sum(buys) - sum(sells)
sma_profit_percent = round((sum(sma_buys) - sum(sma_sells)) / sum(sma_buys) * 100, 2)
#print("Profit for period {start} to {end} using SMA strategy is ${sma_profit_gross} / {sma_profit_percent}%."
        #.format(start=start, end=end, sma_profit_gross=sma_profit_gross, sma_profit_percent=sma_profit_percent))

# Sentiment SMA trade calls
norm_df['2_SMA'] = norm_df['norm_fourier20'].rolling(window=2).mean()
norm_df['5_SMA'] = norm_df['norm_fourier20'].rolling(window=5).mean()

senti_sma_buys=[]
senti_sma_sells=[]

#norm_fourier20 = np.array(norm_df['norm_fourier20']).convolve()

# Buy at local max sell at local min??

#np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]

for i in range(len(tsla_historicals)-1):
    if ((norm_df['2_SMA'].values[i] < norm_df['5_SMA'].values[i]) 
    & (norm_df['2_SMA'].values[i+1] > norm_df['5_SMA'].values[i+1])):
        #print("Trade Call for {row} is Buy.".format(row=norm_df.index[i].date()))
        senti_sma_buys.append(tsla_historicals['Close'].values[i])
    elif ((norm_df['2_SMA'].values[i] > norm_df['5_SMA'].values[i]) & (norm_df['2_SMA'].values[i+1] < norm_df['5_SMA'].values[i+1])):
        #print("Trade Call for {row} is Sell.".format(row=norm_df.index[i].date()))
        senti_sma_sells.append(tsla_historicals['Close'].values[i])

senti_sma_profit_gross = 0
for i in range(len(senti_sma_buys)):
    senti_sma_profit_gross += senti_sma_buys[i] - senti_sma_sells[i]

#profit_gross = sum(buys) - sum(sells)
senti_sma_profit_percent = round((sum(senti_sma_buys) - sum(senti_sma_sells)) / sum(senti_sma_buys) * 100, 2)
#print("Profit for period {start} to {end} using Sentiment SMA strategy is ${senti_sma_profit_gross} / {senti_sma_profit_percent}%."
        #.format(start=start, end=end, senti_sma_profit_gross=senti_sma_profit_gross, senti_sma_profit_percent=senti_sma_profit_percent))

fig = plt.figure(figsize=(20, 10),dpi=80)
ax1 = fig.add_subplot(111)
#ax1.plot(tsla_historicals.index, tsla_historicals['Close'])
#ax1.plot(tsla_historicals.index, tsla_historicals['2_SMA'],'-^', markevery=sma_buys, ms=15, color='green')
#ax1.plot(tsla_historicals.index, tsla_historicals['5_SMA'],'-v', markevery=sma_sells, ms=15, color='red')
ax2 = ax1.twinx()
#ax2.plot(norm_df.index, sentiment_df['norm_price'],'-^', markevery=sma_buys, ms=15, color='green')
#ax2.plot(norm_df.index, norm_df['5_SMA'],'-v', markevery=sma_sells, ms=15, color='red')
ax1.plot(tsla_historicals.index, norm_df['norm_price'], ms=15)
#ax2.plot(tsla_historicals.index, norm_df['norm_sentiment'], ms=15, color='purple')
#ax2.plot(tsla_historicals.index, norm_df['norm_fourier5'], ms=15)
#ax2.plot(tsla_historicals.index, norm_df['norm_fourier10'], ms=15)
#ax2.plot(tsla_historicals.index, norm_df['norm_fourier15'], ms=15)
ax1.plot(tsla_historicals.index, norm_df['norm_fourier20'], ms=15, color='purple')
#ax2.legend(['norm_price','norm_sentiment','norm_fourier5','norm_fourier10', 'norm_fourier15', 'norm_fourier20'])
#ax2.legend(['norm_price','norm_fourier20'])
plt.xlabel('Date',fontsize=14)
#plt.ylabel('Price in Dollars', fontsize = 14)
plt.ylabel('Price in Dollars and Sentiment Score', fontsize = 14)
plt.xticks(rotation='60',fontsize=12)
plt.yticks(fontsize=12)
#plt.title('Normalized Sentiment Fourier - Trade Calls - Moving Averages Crossover - ' + str(sma_profit_percent) + '%' , fontsize = 16)
plt.title('TSLA r/wallstreetbets Sentiment Analysis 1/24/2020-1/24/2021 (normalized)', fontsize = 16)
ax1.legend(['norm_price','norm_fourier20'])
#ax2.legend(['norm_2_SMA', 'norm_5_SMA'])
plt.grid()
#plt.savefig('TSLA_SMA')
plt.savefig('TSLA_Sentiment and Price (normalized)2')
#plt.show() 


#all_days = pd.date_range('01-24-2020', '01-24-2021')

 #pd.read_csv('TSLA_historicals_1Y.csv', header=0)
#print(tsla_historicals)


# tsla_historicals_dates = tsla_historicals['Date']
# tsla_historicals.index = tsla_historicals_dates
# tsla_df = pd.DataFrame(index = all_days, columns = ['score', 'price'])
# tsla_sentiments_df = pd.read_csv('quant_output.csv')
# tsla_sentiments_df.set_index(all_days, inplace = True)
# tsla_df['score'] = tsla_sentiments_df.score

# ##print(tsla_historicals.index)

# for i in range(0, len(all_days)):
#     for j in range(0, len(tsla_historicals_dates)):
#         if all_days[i] == datetime.datetime.strptime(tsla_historicals_dates[j], '%Y-%m-%d'):
#             tsla_df.at[all_days[i], 'price'] = tsla_historicals.at[tsla_historicals_dates[j], 'Close']



# #tsla_df['price'] = tsla_historicals['Close']
# #tsla_df.reindex(all_days)
# #tsla_df.drop(['Date','Open', 'High', 'Low', 'Adj Close','Volume'], axis=1, inplace=True)

# #print(tsla_sentiments_df.index.equals(tsla_df.index))

# tsla_df = tsla_df[tsla_df['price'].notna()]

# print(tsla_df)

# def plot_data(X, Y):
#     plt.scatter(X,Y)
#     plt.xlabel('date')
#     plt.ylabel('prices')
#     #plt.show()

# sc = StandardScaler()

# X = np.array(tsla_df['score'])
# y = np.array(tsla_df['price'])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# X_train = X_train.reshape(-1, 1)
# X_train = sc.fit_transform(X_train)
# #y_train = y_train
# X_test = X_test.reshape(-1, 1)
# X_test = sc.transform(X_test)
# #y_test = y_test.reshape(-1, 1)
# #print(y_train)
# model = RandomForestRegressor()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# #print(tsla_sentiments_prices_df.price.tail(20))
# fig, ax = plt.subplots()
# tsla_df.plot(use_index=True, y = 'price', ax = ax) 
# tsla_df.plot(use_index=True, y = 'score', ax = ax, secondary_y = True) 
# plt.savefig("test3.png")

# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

