

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Load Yelp dataset
yelp = pd.read_csv("yelp.csv")
yelp["text length"] = yelp["text"].apply(len)

# Visualization of text length by star rating
sns.boxplot(x="stars", y="text length", data=yelp)
g = sns.FacetGrid(data=yelp, col="stars")
g.map(plt.hist, 'text length', bins=50)
plt.show()

# Correlation matrix for stars
stars = yelp.groupby('stars').mean()
sns.heatmap(data=stars.corr(), annot=True)
plt.show()

# Filter the dataset for reviews with star ratings of 1 or 5
yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]

# Text preprocessing function
lmtzr = WordNetLemmatizer()

def text_process(text):
    # Remove punctuation
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    # Lemmatize and remove stopwords
    return [lmtzr.lemmatize(word) for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Apply text processing
X = yelp_class['text'].apply(text_process)

# Create a bag-of-words model using CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(X)

# Print vocabulary size
print(len(bow_transformer.vocabulary_))

# Get the vector for the 25th review
review_25 = yelp_class['text'].iloc[24]
bow_25 = bow_transformer.transform([review_25])
print(bow_25)
