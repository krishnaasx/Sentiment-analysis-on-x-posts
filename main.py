import pandas as pd
from textblob import TextBlob
import re
import numpy as np
from nltk.corpus import stopwords

# dataset = pd.read_csv("./training.1600000.processed.noemoticon.csv")
dataset = pd.read_csv("./temp.csv")

noisy_X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

def data_preprocessing(X):
    
    X = np.array([re.sub(r'[^a-z0-9A-Z\s]', '', str(s)) for s in noisy_X])
    X = np.array([re.sub(r'http\S+|www.\S+', '', str(s)) for s in X])
    X = np.array([re.sub(r'@\w+', '', str(s)) for s in X])

    tokens = []
    for x in X:
        tokens.append(TextBlob(x).words)
    tokenized_X = np.array(tokens, dtype = object)
    lowercased_X = [[word.lower() for word in sentence] for sentence in tokenized_X]
    stops = set(stopwords.words('english'))
    non_stopwords_X = [[word for word in sentence if word not in stops] for sentence in lowercased_X]
    
    return np.array(non_stopwords_X, dtype = object)

X = data_preprocessing(noisy_X)




