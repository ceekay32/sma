import pandas as pd
import re
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import poisson
import statistics as st


df=pd.read_csv('preprocessed_tweets_v2.csv')

#-------------------------Wordcloud-------------------------
from wordcloud import WordCloud
import matplotlib.pyplot as plt

hashtags = df['Hashtags'].dropna()  # drop rows with no hashtags
all_hashtags = ' '.join(hashtags)  # concatenate all hashtags into a single string

# create a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='tab20').generate(all_hashtags)

# display the word cloud
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

data=pd.read_csv("AirQualityUCI.csv")
print(data.corr())

sns.heatmap(data.corr(),annot=True)
plt.show()

sns.distplot(data['CO(GT)'],kde=True)
plt.show()

#boxplot
plt.figure(figsize=(10,10))
sns.boxplot(data=data)
plt.show()
#histogram
plt.figure(figsize=(10,7))
sns.histplot(data)
plt.show()
#scatterplot
plt.figure(figsize=(10,10))
plt.gcf().text(.5, .9, "Scatter Plot", fontsize = 40, color='Black' ,ha='center', va='center')
sns.scatterplot(x=data['CO(GT)'] , y=data['T'])
plt.show()

    
#boxplots
sns.boxplot(data['CO(GT)'])
plt.show()
#timeseriesplots
# x = data.drop(['Date','Time'],axis = 1)
# x.dtypes
# data["Date"] = data["Date"].astype("datetime64")
# for i in x.columns:
#     plt.figure(figsize=(16,6))
#     plt.title("Air quality vs diffrent Parameters",fontsize = 14)
#     sns.set(rc={"axes.facecolor":"#283747", "axes.grid":False,'xtick.labelsize':10,'ytick.labelsize':10})
#     plt.xticks(rotation=45) # Rotating X tickts by 45 degrees
#     sns.lineplot(x=data['Date'],y=data[i])
#     plt.show()

#piechart
y = np.array([35, 25, 25, 15])
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]

plt.pie(y, labels = mylabels)
plt.show()