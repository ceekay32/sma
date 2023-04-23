import pandas as pd
import re
import matplotlib.pyplot as plt
from textblob import TextBlob


df=pd.read_csv('preprocessed_tweets_v2.csv')

#Sentiment analysis
sentiment = []
for tweet in df['Tweet']:
  analysis=TextBlob(tweet)
  if(analysis.sentiment[0]>0):
    sentiment.append("Positive")
  elif(analysis.sentiment[0]<0):
    sentiment.append("Negative")
  else:
    sentiment.append("Neutral")

#plotting the sentiment graph
df['sentiment'] = sentiment
df.head()   

df = df.dropna(subset=['Date'])
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = pd.IntervalIndex(pd.cut(df['Date'], pd.date_range('2018-06-01', '2019-05-19',freq='1M'))).left

# count sentiment
tweet_count1 = df.groupby(['Date','sentiment'])['Tweet'].count().reset_index().rename(columns={'Tweet':'count'})
tweet_count1.head()

times = tweet_count1.loc[tweet_count1['sentiment'] == 'Negative']['Date'].reset_index(drop = True)
pos = tweet_count1.loc[tweet_count1['sentiment'] == 'Positive']['count'].reset_index(drop = True)
neutral = tweet_count1.loc[tweet_count1['sentiment'] == 'Neutral']['count'].reset_index(drop = True)
neg = tweet_count1.loc[tweet_count1['sentiment'] == 'Negative']['count'].reset_index(drop = True)

#plot the graph
plt.figure(figsize=(10,6))
plt.xticks(rotation=45)
plt.title("Sentiment count vs. Time")
lin1=plt.plot(times, pos, 'ro-', label='positive')
lin2=plt.plot(times, neutral, 'g^-', label='neutral')
lin3=plt.plot(times, neg, 'b--', label='negative')
plt.legend()
plt.show()