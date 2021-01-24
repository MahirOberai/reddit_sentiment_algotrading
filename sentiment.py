import csv
from enum import Enum

import nltk
import pandas as pd

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class Sentiment(Enum):
  NEGATIVE = 0
  NEUTRAL = 1
  POSITIVE = 2

def get_sentiment_analyzer():
  sia = SentimentIntensityAnalyzer()

  # stock market lexicon
  stock_lex = pd.read_csv('stock_lex.csv')
  stock_lex['sentiment'] = (stock_lex['Aff_Score'] + stock_lex['Neg_Score'])/2
  stock_lex = dict(zip(stock_lex.Item, stock_lex.sentiment))
  stock_lex = {k:v for k,v in stock_lex.items() if len(k.split(' '))==1}
  stock_lex_scaled = {}
  for k, v in stock_lex.items():
    if v > 0:
      stock_lex_scaled[k] = v / max(stock_lex.values()) * 4
    else:
      stock_lex_scaled[k] = v / min(stock_lex.values()) * -4

  # Loughran and McDonald
  positive = []
  with open('lm_positive.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
      positive.append(row[0].strip())
      
  negative = []
  with open('lm_negative.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
      entry = row[0].strip().split(" ")
      if len(entry) > 1:
        negative.extend(entry)
      else:
        negative.append(entry[0])

  final_lex = {}
  final_lex.update({word:2.0 for word in positive})
  final_lex.update({word:-2.0 for word in negative})
  final_lex.update(stock_lex_scaled)
  final_lex.update(sia.lexicon)
  sia.lexicon = final_lex

  return sia

def get_nltk_sentiment(analyzer, blob):
  vs = analyzer.polarity_scores(blob)

  if not vs['neg'] > 0.05:
    if vs['pos'] - vs['neg'] > 0:
      return Sentiment.POSITIVE
    else:
      return Sentiment.NEUTRAL

  elif not vs['pos'] > 0.05:
    if vs['pos'] - vs['neg'] <= 0:
      return Sentiment.NEGATIVE
    else:
      return Sentiment.NEUTRAL
  else:
    return Sentiment.NEUTRAL

def replies_of(analyzer, top_level_comment):
  if len(top_level_comment.replies) == 0:
    return []
  else:
    sentiments = []

    for num, comment in enumerate(top_level_comment.replies):
          try:
              comment_sentiment = get_nltk_sentiment(analyzer, comment.body)
              sentiments.append(comment_sentiment)
          except:
              continue
          sentiments.append(replies_of(analyzer, comment))

    return sentiments

def calculate_date_sentiments(analyzer, api, posts_list, analyze_comments):
  date_sentiments = {}

  for post in posts_list:
    date_sentiments.setdefault(post.created, 0)

    if analyze_comments:
      req = api.submission(id=post.post_id)

      for count, top_level_comment in enumerate(req.comments):
        try:
            top_comment_sentiment = get_nltk_sentiment(analyzer, top_level_comment.body)
            comment_sentiments = replies_of(analyzer, top_level_comment)
            comment_sentiments.append(top_comment_sentiment)
            for comment_sentiment in comment_sentiments:
              if (comment_sentiment == Sentiment.POSITIVE):
                date_sentiments[post.created] = date_sentiments[post.created] + 1
              elif (comment_sentiment == Sentiment.NEGATIVE):
                date_sentiments[post.created] = date_sentiments[post.created] - 1
        except:
            continue

    title_sentiment = get_nltk_sentiment(analyzer, post.title)

    if (title_sentiment == Sentiment.POSITIVE):
      date_sentiments[post.created] = date_sentiments[post.created] + 1
    elif (title_sentiment == Sentiment.NEGATIVE):
      date_sentiments[post.created] = date_sentiments[post.created] - 1

  return date_sentiments