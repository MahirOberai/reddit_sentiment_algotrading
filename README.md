# reddit_sentiment_algotrading
## NLP sentiment analysis trading algorithm that scrapes subreddits (eg. r/wallstreetbets for tsla)
    
### trading algorithm pseudocode:
    #initialize tradeable stocks list
    Iterate through list of stocks (only TSLA for now):
        if stock is tradeable
        AND if stock's most recent news is positive
        AND if stock most recent news has been posted in last 30 minutes:
        AND stock volume < avg. volume:
            add stock ticker and number of clicks to tradeable stock list
    sort tradeable stocks list by number of clicks
    for each stock ticker:
        calculate 20 minute moving average
            stocks_price_day = rs.stocks.get_historicals(inputSymbols='tsla', span='day')
            get last 4 open prices and divide by 4 = 20 minute moving average
        #see if stock is trending upwards
        if 20 minute moving average > current_price:
            #place buy-limit order to buy at the current price 
            robin_stocks.orders.order_buy_limit(symbol, quantity, limitPrice, timeInForce='gtc', extendedHours=False)
