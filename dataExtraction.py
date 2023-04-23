import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "(#loksabhaelections2019 OR #bjp OR #congress OR #nda OR #upa) lang:en until:2019-05-19 since:2018-06-01 -filter:links -filter:replies"
tweets = []
limit = 50000


for tweet in sntwitter.TwitterSearchScraper(query).get_items():
  if tweet.coordinates is not None:
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.user.username, tweet.rawContent, tweet.hashtags, tweet.coordinates, tweet.place, tweet.likeCount, tweet.replyCount, tweet.retweetCount, tweet.quoteCount, tweet.user.verified, tweet.sourceLabel])
        
df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet','Hashtags', 'Coordinates', 'Place', 'Like Count', 'Reply Count', "Retweet Count", "Quote Count", "isVerified", "Source Label"])
print(df)
# df = df[~df['Coordinates'].isnull()]
# to save to csv
df.to_csv('political_tweets_v1.csv')
