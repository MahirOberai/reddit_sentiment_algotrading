#!/usr/bin/python3
"""
broadwaybet.py

This file serves as the main entrypoint for our live-interaction tool. 
Via this tool, you can create a .csv file containing historical
sentiment analysis data based on queries from reddit. 

Usage:

$ python broadwaybet.py [-h] [--start START] [--end END] [--reddit_client_id REDDIT_CLIENT_ID]
                        [--reddit_client_secret REDDIT_CLIENT_SECRET] [--reddit_user_agent REDDIT_USER_AGENT]
                        [--reddit_username REDDIT_USERNAME] [--reddit_password REDDIT_PASSWORD] [--lex_dir LEX_DIR] [--sub SUB]
                        [--csv_out CSV_OUT] [--nobacktest NOBACKTEST] [--notearsheet NOTEARSHEET]
                        QUERY [QUERY ...]

The query can either be a single string, or several strings separated by spaces (use quotes to query a string including spaces). 

Try this:

python3 broadwaybet.py --quandlkey="gA22YWcyysEyaNxu1jti" --start="01-24-2020" --end="01-24-2021" --no_output_csv=True --csv_out="out_test.csv" --quant_csv_out="test_quant.csv" TSLA tesla 'Elon Musk'
"""
import argparse
import datetime 
import pytz

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


#import seaborn

#import quandl
#import yfinance as yf
import praw
#from alphalens.tears import create_returns_tear_sheet
#from alphalens.utils import get_clean_factor_and_forward_returns
#import pyfolio as pf
#import empyrical as em

from reddit import query_subreddit
from sentiment import get_sentiment_analyzer, calculate_date_sentiments

parser = argparse.ArgumentParser()
parser.add_argument('--start', default="01-24-2020")
parser.add_argument('--end',  default="01-24-2021") #03/27/2018 is the last date in WIKI/PRICES database
parser.add_argument('--analyze_comments', default=True)
parser.add_argument('--no_output_csv', default=True)
parser.add_argument('--reddit_client_id', default="hpAt82a0aMv8DQ")
parser.add_argument('--reddit_client_secret', default="jI33vgtVVIvMJODsLmLEWtL-eP4")
parser.add_argument('--reddit_user_agent', default="trading_sentiment")
parser.add_argument('--reddit_username', default="MahirOberai")
parser.add_argument('--reddit_password', default="Algotrading4995")
# parser.add_argument('--lex_dir', required=False)
parser.add_argument('--quandlkey', default="gA22YWcyysEyaNxu1jti")
parser.add_argument('--ticker', default="TSLA")
parser.add_argument('--sub', default="wallstreetbets")
parser.add_argument('--csv_out', required=False, default="output.csv")
parser.add_argument('--quant_csv_out', required=False, default="quant_output.csv")
# parser.add_argument('--nobacktest', required=False, default=False)
# parser.add_argument('--notearsheet', required=False, default=False)
parser.add_argument('query', metavar='QUERY', nargs='+')

if __name__ == "__main__":
  args = parser.parse_args()

  start_time = datetime.datetime.strptime(args.start, "%m-%d-%Y")
  end_time = datetime.datetime.strptime(args.end, "%m-%d-%Y")

  posts_list = []

  if args.no_output_csv: 
    if (len(args.query) == 1):
      print("Querying reddit for posts containing '{}'".format((args.query[0])))
      posts_list = posts_list + query_subreddit(args.query[0], start_time, end_time, args.sub)

    else: 
      for query in args.query:
        print("Querying reddit for posts containing '{}'".format(query))
        posts_list = posts_list + query_subreddit(query, start_time, end_time, args.sub)

    print("Calculating sentiments for posts: ",format(len(posts_list)))

    analyzer = get_sentiment_analyzer()

    if args.analyze_comments:
      print("Will calculate sentiments for comments. Will take some time.")
      reddit_api = praw.Reddit(client_id=args.reddit_client_id,
                            client_secret=args.reddit_client_secret,
                            user_agent=args.reddit_user_agent,
                            password=args.reddit_password,
                            username=args.reddit_username)
    else:
      print("Will not calculate sentiments for comments.")
      reddit_api = None


    date_sentiments = calculate_date_sentiments(analyzer, reddit_api, posts_list, args.analyze_comments)

    dates = list(date_sentiments.keys())
    #dates.sort()

    sentiment_scores = [date_sentiments[date] for date in dates]
    sentiment_df_data = {'date': dates, 'score': sentiment_scores}

    sentiment_df = pd.DataFrame(sentiment_df_data)
    sentiment_df['date'] = sentiment_df['date'].dt.date
    sentiment_df.sort_values(by=['date'])
    sentiment_df.to_csv(args.csv_out, index=False)

    print('CSV file written to: {}'.format(args.csv_out))
  
  else:
    print('Reading sentiments from: {}'.format(args.csv_out))
    sentiment_df = pd.read_csv(args.csv_out)

  quantized_df = sentiment_df.groupby(['date']).mean()

  print('Writing quantized sentiment data to: {}'.format(args.quant_csv_out))
  quantized_df.to_csv(args.quant_csv_out)

  #sentiment_df['date'] = pd.to_datetime(sentiment_df.date).dt.date


  # quantized_date_sentiments = {}
  # for unique_date in sentiment_df['date'].dt.date.unique():
  #   print(unique_date)
  #   unique_date_score = 0
  #   count = 0
  #   for date, score in zip(sentiment_df['date'], sentiment_df['score']):
  #       if date == unique_date:
  #           count += 1
  #           unique_date_score += score
  #   quantized_date_sentiments[unique_date] = unique_date_score/count       

  

  #quantized_df = pd.DataFrame(data = quantized_date_sentiments.values, index = quantized_date_sentiments.keys, columns = ['score'])
  

  #print(len(quantized_date_sentiments))
  #quandl.ApiConfig.api_key = args.quandlkey

  # quandl_prices = quandl.get_table(
  #   'WIKI/PRICES',
  #   qopts = { 'columns': ['date', 'close'] },
  #   ticker = args.ticker,
  #   date = {
  #     'gte': datetime.datetime.fromtimestamp(start_time.timestamp(), tz=pytz.timezone('US/Eastern')).strftime("%Y-%m-%d"),
  #     'lte': datetime.datetime.fromtimestamp(end_time.timestamp(), tz=pytz.timezone('US/Eastern')).strftime("%Y-%m-%d")
  #   }
  # )
  # # quandl_prices = prices.set_index('date')
  # quandl_prices = list(quandl_prices.iterrows())

  # price_dict = {}
  # for row in quandl_prices:
  #   price_dict[row[1][0]] = row[1][1]



  # full_index = pd.date_range(
  #   start=datetime.datetime.fromtimestamp(start_time.timestamp(), tz=pytz.timezone('US/Eastern')).strftime("%Y-%m-%d"),
  #   end=datetime.datetime.fromtimestamp(end_time.timestamp(), tz=pytz.timezone('US/Eastern')).strftime("%Y-%m-%d")
  # )

  # factors = []

  
  # prices = []
  # for date in full_index:
  #   if (date in price_dict.keys()):
  #     prices.append(price_dict[date])
  #   else:
  #     try:
  #       prices.append(prices[-1])
  #     except:
  #       continue
    
  # for date in full_index:
  #   if (date.date() in quantized_date_sentiments.keys()):
  #       factors.append(quantized_date_sentiments[date.date()])
  #   else:
  #       factors.append(0)
  
  #print(factors)

  # prices = pd.DataFrame(index=full_index, columns=[args.ticker], data=prices)

  # print(prices.head())

  # factor_series = pd.DataFrame(index=full_index, columns=[args.ticker], data=factors).rolling(window=3).mean().stack()
  # factor_series = pd.DataFrame(columns=['score'], data=factors)
  # factor_series['date'] = full_index

  

  #

  # factor_data = get_clean_factor_and_forward_returns(
  #     factor_series,
  #     prices,
  #     periods=(1, 5, 15), 
  #     filter_zscore=None,
  #     quantiles=None,
  #     bins=1)

  # factor_data.replace([np.inf, -np.inf], 0)

  # # replace this for a cleaner way to access the dates level in the multi-index dataframe
  # index_values = list(factor_data.index.values)

  # index_dates = []
  # for i in range(0, len(index_values)):
  #   index_dates.append(index_values[i][0])

  # create_returns_tear_sheet(factor_data, long_short=False, group_neutral=False, by_group=False)

  # plt.savefig("alphalens_tear_sheet.png")

  # stock_rets = pd.Series(factor_data['1D'].values, index = index_dates)

  # # must run 'pip install git+https://github.com/quantopian/pyfolio'
  # # also run 'pip install empyrical'

  # f = pf.create_returns_tear_sheet(stock_rets, return_fig=True)
  # f.savefig('stocks_tsla_titles.png')

  # import IPython; IPython.embed()
  
  
