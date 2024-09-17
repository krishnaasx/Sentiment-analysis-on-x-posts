import pandas as pd
from textblob import TextBlob
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import math
import itertools
import nltk


class sentimental_analysis:

    def __init__(self):
        nltk.download("stopwords")
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        self.stops = set(stopwords.words("english"))
        self.lmtzr = WordNetLemmatizer()

    #Data Preprocessing
    def data_preprocessing(self, X):

        # Removing noise
        X = np.array([re.sub(r"@\w+", "", re.sub(r"http\S+|www.\S+", "", re.sub(r"[^a-z0-9A-Z\s]", "", str(s)))) for s in X])

        # Tokenization
        tokens = []
        for x in X:
            tokens.append(TextBlob(x).words)
        tokenized_X = np.array(tokens, dtype=object)

        # Lowercasing of tokenized data
        lowercased_X = [[word.lower() for word in sentence] for sentence in tokenized_X]

        # Removing stop words

        non_stopwords_X = [
            [word for word in sentence if word not in self.stops]
            for sentence in lowercased_X
        ]

        # lemmetization
        self.lmtzr = WordNetLemmatizer()
        lemmetized_X = [
            [self.lmtzr.lemmatize(word) for word in sentences]
            for sentences in non_stopwords_X
        ]

        return np.array(lemmetized_X, dtype=object)
    
    #Feature extration
    def feature_extraction(self, X):

        # BagOfWords
        bow = {}
        sen_count = 1
        for sentence in X:
            sent = {}
            for word in sentence:
                if word not in sent.keys():
                    sent[word] = 1
                else:
                    sent[word] += 1
            bow[f"Post-{sen_count}"] = sent
            sen_count += 1
            
        # TF
        TF = {}
        for insideDict in bow:
            TOTAL_D = 0
            for d in bow[insideDict].values():
                TOTAL_D += d

            temp = {}
            for key, value in zip(bow[insideDict].keys(), bow[insideDict].values()):
                temp[key] = value / TOTAL_D

            TF[insideDict] = temp

        TF = dict(
            itertools.chain.from_iterable(
                [[(key, value)for key, value in zip(TF[post].keys(), TF[post].values())]for post in TF]
            )
        )

        #IDF
        IDF = {}
        document_len = len(bow)
        stringContaniningDoc = {}
        for post in bow:
            for keys, values in zip(bow[post].keys(), bow[post].values()):
                if keys not in stringContaniningDoc:
                    stringContaniningDoc[keys] = values
                elif keys in stringContaniningDoc:
                    stringContaniningDoc[keys] += values

        for term, value in stringContaniningDoc.items():
            IDF[term] = math.log(document_len / value)
            

        #Calculating TF-IDF
        TF_IDF = {}
        for term,tf_val, idf_val in zip (TF.keys(), TF.values(), IDF.values()):
            TF_IDF[term] = tf_val * idf_val
        
        
        
        

if __name__ == "__main__":

    # dataset = pd.read_csv("./training.200000.processed.noemoticon.csv")
    dataset = pd.read_csv("./temp.csv")

    raw_X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    sen = sentimental_analysis()

    preprocessed_data = sen.data_preprocessing(raw_X)
    extracted_features = sen.feature_extraction(preprocessed_data)
    
    print(preprocessed_data)
