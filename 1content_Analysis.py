# !pip install pyLDAvis
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
import pyLDAvis.gensim
import matplotlib.pyplot as plt
# Download the stopwords resource
# nltk.download('stopwords')
# nltk.download('punkt')
df=pd.read_csv('preprocessed_tweets_v2.csv')

# Tokenize the text and remove stop words
stop_words = set(stopwords.words('english'))
texts = [[word for word in nltk.word_tokenize(doc.lower()) if word.isalpha() and word not in stop_words] for doc in df['Tweet']]

# Create a dictionary and corpus for the LDA model
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Train the LDA model with 10 topics
lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10)

# Print the top 10 words for each topic
for i, topic in lda_model.show_topics(num_topics=10, num_words=10, formatted=False):
    print('Topic {}: {}'.format(i, ', '.join([word[0] for word in topic])))

# Visualize the topics using pyLDAvis
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
# pyLDAvis.show(vis)
pyLDAvis.display(vis)
