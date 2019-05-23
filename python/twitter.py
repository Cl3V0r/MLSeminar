import twitter  #conda install -c conda-forge python-twitter
import re
import datetime
import pandas as pd


class twitterminer():
    request_limit = 20
    api = False
    data = []
    twitter_keys = {
          'consumer_key':12,  # add your consumer key
          'consumer_secret':123,  # add your consumer secret key
          'access_token_key':1234,  # add your access token key
          'access_token_secret':12345  # add your access token secret key
    }
  
    def __init__(self,  request_limit=20):
  
          self.request_limit = request_limit
  
          # This sets the twitter API object for use internall within the class
          self.set_api()
  
    def set_api(self):
  
          self.api = twitter.Api(
              consumer_key=self.twitter_keys['consumer_key'],
              consumer_secret=self.twitter_keys['consumer_secret'],
              access_token_key=self.twitter_keys['access_token_key'],
              access_token_secret=self.twitter_keys['access_token_secret']
          )
  
  
    def mine_user_tweets(self, user=" set default user to get data from", mine_retweets=False):

        statuses = self.api.GetUserTimeline(screen_name=user, count=self.request_limit)
        data = []
  
        for item in statuses:
            mined = {
                  'tweet_id': item.id,
                  'handle': item.user.name,
                  'retweet_count': item.retweet_count,
                  'text': item.text,
                  'mined_at': datetime.datetime.now(),
                  'created_at': item.created_at,
              }
            data.append(mined)
  
        return data
  
mine = twitterminer()
# insert handle we like
trump_tweets = miner.mine_user_tweets("realDonaldTrump")
trump_df = pd.DataFrame(trump_tweets)
