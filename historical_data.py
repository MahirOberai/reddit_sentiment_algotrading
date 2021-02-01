#import yfinance as yf
#from broadwaybet import full_index
import pandas as pd
import pandas_datareader as pdr
import datetime 
import pytz
import numpy as np
import matplotlib.pylab as plt
#from broadwaybet import start_time, end_time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

#start_time = 01

start = '01-24-2020'
end = '01-24-2021'

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
        data = pdr.DataReader(ticker, 'iex', start, end)
        data.index = pd.to_datetime(data.index)
    except:
        data = pdr.get_data_yahoo(ticker, start, end)
    return data 

tsla_historicals = get_ticker_data(start, end, 'TSLA')

tsla_historicals['2_SMA'] = tsla_historicals['Close'].rolling(window=2).mean()
tsla_historicals['5_SMA'] = tsla_historicals['Close'].rolling(window=5).mean()

tsla_historicals = tsla_historicals[tsla_historicals['5_SMA'].notna()]

# SMA trade calls
buys=[]
sells=[]
for i in range(len(tsla_historicals)-1):
    if ((tsla_historicals['2_SMA'].values[i] < tsla_historicals['5_SMA'].values[i]) 
    & (tsla_historicals['2_SMA'].values[i+1] > tsla_historicals['5_SMA'].values[i+1])):
        print("Trade Call for {row} is Buy.".format(row=tsla_historicals.index[i].date()))
        buys.append(i)
    elif ((tsla_historicals['2_SMA'].values[i] > tsla_historicals['5_SMA'].values[i]) & (tsla_historicals['2_SMA'].values[i+1] < tsla_historicals['5_SMA'].values[i+1])):
        print("Trade Call for {row} is Sell.".format(row=tsla_historicals.index[i].date()))
        sells.append(i)

profit_gross = sum(buys) - sum(sells)
profit_percent = round((sum(buys) - sum(sells)) / sum(buys) * 100, 2)
print("Profit for period {start} to {end} using SMA strategy is ${profit_gross} / {profit_percent}%."
        .format(start=start, end=end, profit_gross=profit_gross, profit_percent=profit_percent))

plt.figure(figsize=(20, 10),dpi=80)
plt.plot(tsla_historicals.index, tsla_historicals['Close'])
plt.plot(tsla_historicals.index, tsla_historicals['2_SMA'],'-^', markevery=buys, ms=15, color='green')
plt.plot(tsla_historicals.index, tsla_historicals['5_SMA'],'-v', markevery=sells, ms=15, color='red')
plt.xlabel('Date',fontsize=14)
plt.ylabel('Price in Dollars', fontsize = 14)
plt.xticks(rotation='60',fontsize=12)
plt.yticks(fontsize=12)
plt.title('Trade Calls - Moving Averages Crossover - ' + str(profit_percent) + '%' , fontsize = 16)
plt.legend(['Close','2_SMA','5_SMA'])
plt.grid()
plt.savefig('TSLA_SMA')
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

