import pandas as pd
import re
import matplotlib.pyplot as plt
from textblob import TextBlob


df=pd.read_csv('preprocessed_tweets_v2.csv')

all_text = ' '.join(df['Hashtags'].values)
all_text = re.sub(r'\] \[', ',', all_text)
all_text = re.sub(r']', '', all_text)
all_text = re.sub(r'\[', '', all_text)
all_text = re.sub(r', ', ',', all_text)
all_text = all_text.lower()
all_text_l = all_text.split(',')

#Hashtags counter
from collections import Counter
hash_counts = Counter(all_text_l)
top_hash = hash_counts.most_common(15)
print(top_hash)

#Graph of most commonly used hashtags
top_hash = hash_counts.most_common(20)
x_values = [word[0] for word in top_hash]
y_values = [word[1] for word in top_hash]

plt.barh(x_values, y_values)

plt.xlabel('Frequency')
plt.ylabel('Word')
plt.title('Most Commonly Used Hashtags')
plt.rcParams["figure.figsize"] = (20,20)

for index, value in enumerate(y_values):
    plt.text(value, index,str(value),color="red")

plt.show()


#-------------------------different user groups-------------------------
# # Define a function to plot a bar graph of the top n hashtags for a given source label
# def plot_hashtag_counts(source_label, n):
#     # Extract the hashtags for tweets with the current source label
#     all_text = ' '.join(df.loc[df['Source Label']==source_label, 'Hashtags'].values)
#     all_text = re.sub(r'\] \[', ',', all_text)
#     all_text = re.sub(r']', '', all_text)
#     all_text = re.sub(r'\[', '', all_text)
#     all_text = re.sub(r', ', ',', all_text)
#     all_text = all_text.lower()
#     all_text_l = all_text.split(',')
    
#     # Count the frequency of each hashtag
#     hash_counts = Counter(all_text_l)
#     top_hash = hash_counts.most_common(n)
#     x_values = [word[0] for word in top_hash]
#     y_values = [word[1] for word in top_hash]

#     # Plot a bar graph of the top hashtags
#     plt.barh(x_values, y_values)
#     plt.xlabel('Frequency')
#     plt.ylabel('Hashtag')
#     plt.title('Top {} Hashtags for {}'.format(n, source_label))
#     plt.gca().invert_yaxis()
#     plt.show()

# # Loop through each unique value in the 'Source Label' column and plot a bar graph for the top 10 hashtags
# for source_label in df['Source Label'].unique():
#     plot_hashtag_counts(source_label, 10)

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
