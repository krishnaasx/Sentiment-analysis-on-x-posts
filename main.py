import re
import nltk
import pandas as pd
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from joblib import parallel_backend
from sklearn.calibration import CalibratedClassifierCV
import multiprocessing

class SentimentAnalyzer:
    def __init__(self, max_features=10000, n_jobs=-1):
        # Download NLTK resources only if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download("stopwords")
            nltk.download("punkt")
            nltk.download("wordnet")
            nltk.download("omw-1.4")
        
        self.stops = set(stopwords.words("english")) - {'not', 'no', 'won\'t', 'shouldn\'t', 'couldn\'t', 'wouldn\'t', 'hasn\'t', 'haven\'t', 'hadn\'t', 'doesn\'t', 'don\'t', 'didn\'t'}
        self.lmtzr = WordNetLemmatizer()
        self.max_features = max_features
        self.n_jobs = n_jobs
        
        # Improved regex pattern for text cleaning
        self.cleanup_pattern = re.compile(r'@\w+|https?://\S+|www\.\S+|[^a-zA-Z\s!?]')
        
        # Initialize pipeline components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.9,
            strip_accents='unicode',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        
        # Use LinearSVC for faster training
        self.classifier = CalibratedClassifierCV(
            LinearSVC(
                C=1.0,
                class_weight='balanced',
                dual=False,
                max_iter=1000,
                random_state=42
            ),
            n_jobs=self.n_jobs
        )

    def preprocess_text(self, text):
        # Clean text
        text = self.cleanup_pattern.sub(' ', str(text).lower())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lmtzr.lemmatize(token)
            for token in tokens
            if token not in self.stops and len(token) > 2
        ]
        
        return ' '.join(tokens)

    def prepare_data(self, X):
        # Use parallel processing for preprocessing
        with parallel_backend('threading', n_jobs=self.n_jobs):
            processed_texts = [self.preprocess_text(text) for text in X]
        return processed_texts

    def extract_features(self, processed_texts):
        return self.tfidf_vectorizer.transform(processed_texts)

    def train_and_evaluate(self, X, y, test_size=0.2):
        # Prepare data
        print("Preprocessing texts...")
        processed_texts = self.prepare_data(X)
        
        # Extract features
        print("Extracting features...")
        self.tfidf_vectorizer.fit(processed_texts)
        X_transformed = self.extract_features(processed_texts)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, 
            test_size=test_size, 
            random_state=42,
            stratify=y
        )
        
        # Train model
        print("Training model...")
        with parallel_backend('threading', n_jobs=self.n_jobs):
            self.classifier.fit(X_train, y_train)
        
        # Evaluate
        print("Evaluating model...")
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) 
        report = classification_report(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:\n", report)
        
        return accuracy, report
    
    def predict(self, texts):
        processed_texts = self.prepare_data(texts)
        features = self.extract_features(processed_texts)
        return self.classifier.predict(features)

if __name__ == "__main__":
    # Set number of processes for parallel processing
    n_jobs = multiprocessing.cpu_count() - 1
    print(f"using {n_jobs} CPUs")
    print("Loading dataset...")
    dataset = pd.read_csv(
        "./training.1600000.processed.noemoticon.csv",
        encoding="ISO-8859-1"
    )
    X = dataset["Text"].values
    y = dataset["Target"].values
    
    # Initialize and train model
    sentiment_analyzer = SentimentAnalyzer(
        max_features=1500000,
        n_jobs=n_jobs
    )
    
    sentiment_analyzer.train_and_evaluate(X, y)