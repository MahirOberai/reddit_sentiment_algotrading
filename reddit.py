"""
Scripts and functions for interacting with reddit api and download subreddit data.  
"""

import time
import json
import time
import requests
import datetime
from collections import namedtuple

import praw


Post = namedtuple("Post", "post_id, title, url, author, score, created, num_comments, permalink, flair")

def get_reddit_api(client_id, client_secret, user_agent, username, password):
  reddit_api = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
    password=password,
    username=username
  )
  return reddit_api

def get_pushshift_data(query, start, end, subreddit):
  url = 'https://api.pushshift.io/reddit/search/submission/?title='+str(query)+'&size=1000&after='+str(start)+'&before='+str(end)+'&subreddit='+str(subreddit)
  time.sleep(1) # to avoid rate limit
  r = requests.get(url)
  data = json.loads(r.text)
  return data['data']

def parse_post(post):
  post_id = post['id']
  title = post['title']
  url = post['url']
  author = post['author']
  score = post['score']
  created = datetime.datetime.fromtimestamp(post['created_utc'])
  num_comments = post['num_comments'] 
  permalink = post['permalink']
  
  try:
    flair = post['link_flair_text']
  except KeyError:
    flair = "NaN"

  return Post(post_id, title, url, author, score, created, num_comments, permalink, flair)

def query_subreddit(query, start, end, subreddit):
  data = get_pushshift_data(query, start, end, subreddit)

  posts_list = []

  # Will run until all posts have been gathered 
  # from the 'after' date up until before date
  while len(data) > 0:
    for post in data:
      posts_list.append(parse_post(post))

    start = data[-1]['created_utc']
    data = get_pushshift_data(query, start, end, subreddit)
  
  return posts_list
  
def get_post(client_id, client_secret, user_agent, username, password, post_id):
  api = get_reddit_api(client_id, client_secret, user_agent, username, password)

  return api.submission(id=post_id)
