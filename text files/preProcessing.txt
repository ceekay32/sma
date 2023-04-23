import numpy as np
import pandas as pd
import re
import nltk
import string
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

pd.options.mode.chained_assignment = None
df = pd.read_csv("political_tweets_v1.csv")
# Loop through each column in the dataframe and remove any float values
for col in df.columns:
    df[col] = df[col].apply(lambda x: x if not isinstance(x, float) else None)

# Drop any rows that contain null values
df = df.dropna()

print(df)
# remove data from the "Place" column
df = df.dropna(subset=['Place'])

df['Place'] = df['Place'].str.replace('Place', '') # define a function to extract each attribute from the "Place" column
def extract_place_attribute(place_string, attribute):
    start_index = place_string.find(attribute) + len(attribute) + 2
    end_index = place_string.find(',', start_index)
    return place_string[start_index:end_index]

# use apply method to extract each attribute from the "Place" column and create separate columns for each attribute
df['Place ID'] = df['Place'].apply(lambda x: extract_place_attribute(x, "id"))
df['Place Full Name'] = df['Place'].apply(lambda x: extract_place_attribute(x, "fullName"))
df['Place Name'] = df['Place'].apply(lambda x: extract_place_attribute(x, "name"))
df['Place Type'] = df['Place'].apply(lambda x: extract_place_attribute(x, "type"))
df['Country'] = df['Place'].apply(lambda x: extract_place_attribute(x, "country"))
df['Country Code'] = df['Place'].apply(lambda x: extract_place_attribute(x, "countryCode"))

# drop the original "Place" column
df.drop('Place', axis=1, inplace=True)

df['Coordinates'] = df['Coordinates'].str.replace('Coordinates', '')

# save the updated data to a new CSV file
df.to_csv('seperated_tweets_v2.csv', index=False)
# df.head()

#To lower case
df["Tweet"] = df["Tweet"].str.lower()
# df.drop(["tweet_lower"], axis=1, inplace=True)

#-------------------------Remove Punctuations-------------------------
PUNCT_TO_REMOVE = string.punctuation

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

df["Tweet"] = df["Tweet"].apply(lambda text: remove_punctuation(text))
df["Place ID"] = df["Place ID"].apply(lambda text: remove_punctuation(text))
df["Place Name"] = df["Place Name"].apply(lambda text: remove_punctuation(text))
df["Place Type"] = df["Place Type"].apply(lambda text: remove_punctuation(text))
df["Country"] = df["Country"].apply(lambda text: remove_punctuation(text))
df["Country Code"] = df["Country Code"].apply(lambda text: remove_punctuation(text))



# df.head()
# df
#-------------------------Remove stop words-------------------------
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
print(stopwords.words('english'))
en_stopwords = stopwords.words('english')
def remove_stopwords(text):
    result = []
    #text is a string and not an object
    #splitted into words to form a list of words
    lst = text.split(" ")
    # print(lst)
    for token in lst:
        if token not in en_stopwords:
            result.append(token)
            
    return result
df['tweet_wo_stopwords'] = df['tweet_wo_url'].apply(remove_stopwords)
# df

#-------------------------Stemming-------------------------
from nltk.stem import PorterStemmer
def stemming(text):
    porter = PorterStemmer()
    
    result=[]
    for word in text:
        result.append(porter.stem(word))
    return result
df['stem']=df['tweet_wo_stopwords'].apply(stemming)
df.head()

#-------------------------Lemmatization-------------------------

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag

def lemmatization(text):
    result=[]
    wordnet = WordNetLemmatizer()
    for token,tag in pos_tag(text):
        pos=tag[0].lower()
        
        if pos not in ['a', 'r', 'n', 'v']:
            pos='n'
            
        result.append(wordnet.lemmatize(token,pos))
    
    return result

df['lemma']=df['tweet_wo_stopwords'].apply(lemmatization)
# df.head()

#-------------------------Count most common words-------------------------
from collections import Counter
cnt = Counter()
for text in df["Tweet"].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(15)

#-------------------------Remove New Lines-------------------------
def remove_newLines(text):
    html_pattern = re.compile('\n')
    return html_pattern.sub(r'', text)

df["Tweet"] = df["Tweet"].apply(lambda text: remove_newLines(text))
# df.head()
# df

df.to_csv("preprocessed_tweets_v2.csv")