#exp9
import pandas as pd
from textblob import TextBlob
df=pd.read_csv("preprocessed_tweets_v2.csv")
df = df.dropna()

# Performing sentiment analysis on the "Tweet" column
df['Sentiment'] = df['Tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['Sentiment_Label'] = pd.cut(df['Sentiment'], bins=3, labels=['Negative', 'Neutral', 'Positive'])

# Creating new dataframe with only negative sentiment labels
df_negative = df.loc[(df['Sentiment_Label'] == 'Negative')|(df['Sentiment_Label'] == 'Neutral')]
# df_negative = df.loc[df['Sentiment_Label'] == 'Negative']

# Create a new dataframe with hashtags containing "BJP"
df_bjp = df_negative[df_negative['Hashtags'].apply(lambda x: (('bjp' or 'BJP') in x))]

# Print the new dataframe
df_bjp.head()
# Create a new dataframe with hashtags containing "CONGRESS"
df_congress = df_negative[df_negative['Hashtags'].apply(lambda x: (('congress' or 'Congress') in x))]

# Print the new dataframe
df_congress.head()

import matplotlib.pyplot as plt

# Count the number of tweets containing "BJP" and "CONGRESS"
count_bjp = len(df_bjp)
count_congress = len(df_congress)

# Create a bar chart
fig, ax = plt.subplots()
bars = ax.bar(['BJP', 'CONGRESS'], [count_bjp, count_congress], color=['orange', 'blue'])
ax.set_xlabel('Political Parties')
ax.set_ylabel('Number of Tweets')
ax.set_title('Sentiment Analysis of Tweets Containing BJP and CONGRESS Hashtags')

# Add counts on top of bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom')

plt.show()
