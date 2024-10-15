import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.calibration import LabelEncoder
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

class ImprovedSentimentalAnalysis:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stops = set(stopwords.words("english"))
        self.lmtzr = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def clean_text(self, text):
        text = re.sub(r"@\w+|http\S+|www.\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s!?]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def data_preprocessing(self, X):
        X = np.array([self.clean_text(str(s)) for s in X])
        tokenized_X = [word_tokenize(sentence.lower()) for sentence in X]
        non_stopwords_X = [
            [word for word in sentence if word not in self.stops or word in ["not", "no"]]
            for sentence in tokenized_X
        ]
        lemmatized_X = [[self.lmtzr.lemmatize(word) for word in sentence] for sentence in non_stopwords_X]
        processed_sentences = [' '.join(sentence) for sentence in lemmatized_X]
        return lemmatized_X, processed_sentences

    def feature_extraction(self, X, processed_sentences):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_sentences)
        self.word2vec_model = Word2Vec(sentences=X, vector_size=300, window=5, min_count=1, workers=4)
        
        sentence_embeddings = []
        for sentence in X:
            word_vectors = [self.word2vec_model.wv[word] for word in sentence if word in self.word2vec_model.wv]
            if word_vectors:
                sentence_vector = np.mean(word_vectors, axis=0)
            else:
                sentence_vector = np.zeros(self.word2vec_model.vector_size)
            sentence_embeddings.append(sentence_vector)
        
        word2vec_embeddings = np.array(sentence_embeddings)
        combined_features = np.hstack((tfidf_matrix.toarray(), word2vec_embeddings))
        return self.scaler.fit_transform(combined_features)

    def predict_sentiment(self, sentence):
        processed_sentence, _ = self.data_preprocessing(np.array([sentence]))
        tfidf_features = self.tfidf_vectorizer.transform([' '.join(processed_sentence[0])])
        word2vec_features = np.mean([self.word2vec_model.wv[word] for word in processed_sentence[0] if word in self.word2vec_model.wv], axis=0)
        combined_features = np.hstack((tfidf_features.toarray(), word2vec_features.reshape(1, -1)))
        scaled_features = self.scaler.transform(combined_features)
        prediction = self.model.predict(scaled_features)
        return prediction[0]

    def preprocess_target(self, y):
        # Encode target labels to 0 and 1
        return self.label_encoder.fit_transform(y)

if __name__ == "__main__":
    dataset = pd.read_csv("./temp2.csv", encoding="ISO-8859-1")
    raw_X = dataset["Text"].values  
    y = dataset["Target"].values
    
    sentiment_analyzer = ImprovedSentimentalAnalysis()
    
    # Preprocess target variable
    y = sentiment_analyzer.preprocess_target(y)
    
    lemmatized_X, processed_sentences = sentiment_analyzer.data_preprocessing(raw_X)
    X = sentiment_analyzer.feature_extraction(lemmatized_X, processed_sentences)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Apply SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # XGBoost with hyperparameter tuning
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200, 300],
        'gamma': [0, 0.1, 0.2]
    }
    
    xgb_model = XGBClassifier(random_state=42)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    best_model = grid_search.best_estimator_
    sentiment_analyzer.model = best_model
    
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)
    
    while True:
        user_input = input("\nEnter a sentence to predict sentiment (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        predicted_sentiment = sentiment_analyzer.predict_sentiment(user_input)
        # Convert back to original label
        original_label = sentiment_analyzer.label_encoder.inverse_transform([predicted_sentiment])[0]
        print(f"Predicted sentiment: {original_label}")