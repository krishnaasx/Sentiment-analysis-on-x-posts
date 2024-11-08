import re
import nltk
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report


class Sentimental_analysis:

    def __init__(self):
        # nltk.download("stopwords")
        # nltk.download("punkt")
        # nltk.download("wordnet")
        # nltk.download("omw-1.4")
        self.stops = set(stopwords.words("english"))
        self.lmtzr = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000) 
    
    # Data Preprocessing
    def data_preprocessing(self, X):
        
        X = np.array([re.sub(r"@\w+|http\S+|www.\S+|[^a-zA-Z\s!?]", "", str(s)) for s in X])

        tokenized_X = [word_tokenize(sentence.lower()) for sentence in X]

        non_stopwords_X = [
            [word for word in sentence if word not in self.stops or word in ["not", "no"]]
            for sentence in tokenized_X
        ]

        lemmatized_X = [[self.lmtzr.lemmatize(word) for word in sentence] for sentence in non_stopwords_X]
        
        processed_sentences = [' '.join(sentence) for sentence in lemmatized_X]
        
        return np.array(lemmatized_X, dtype=object), processed_sentences #("["weekend", "off","with,"unfortunately","nothing","to","do"], ["dfafda", "fdafda",""]")


    # Feature Extraction
    def feature_extraction(self, processed_sentences):
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_sentences)

        # self.word2vec_model = Word2Vec(sentences=X, vector_size=300, window=5, min_count=1, workers=4)

        # sentence_embeddings = []
        # for sentence in X:
        #     word_vectors = [self.word2vec_model.wv[word] for word in sentence if word in self.word2vec_model.wv]
        #     if word_vectors:
        #         sentence_vector = np.mean(word_vectors, axis=0)
        #     else:
        #         sentence_vector = np.zeros(self.word2vec_model.vector_size)
        #     sentence_embeddings.append(sentence_vector)

        # word2vec_embeddings = np.array(sentence_embeddings)

        return tfidf_matrix # word2vec_embeddings

    # Tranining and evaluation 
    def train_and_evaluate_model(self, X, y):
        
        processed_sentences = self.data_preprocessing(X)
        
        tfidf_matrix = self.feature_extraction(processed_sentences)
        
        X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.20, random_state=42)
        
        svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
        svm_model.fit(X_train, y_train)
        
        y_pred = svm_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        report = classification_report(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:\n", report)

if __name__ == "__main__":
    
    dataset = pd.read_csv("./training.200000.processed.noemoticon.csv", encoding="ISO-8859-1")
    raw_X = dataset["Text"].values  
    y = dataset["Target"].values  
    
    sentiment_analyzer = Sentimental_analysis()

    sentiment_analyzer.train_and_evaluate_model(raw_X, y)