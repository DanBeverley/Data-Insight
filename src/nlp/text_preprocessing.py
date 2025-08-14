"""Text Preprocessing Transformers for DataInsight AI"""

import logging
import re
import string
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk.sentiment import SentimentIntensityAnalyzer
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
        
except ImportError:
    nltk = None
    logging.warning("NLTK not available. Basic text processing only.")

class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """Clean and normalize text data with configurable options."""
    
    def __init__(
        self, 
        lemmatize: bool = True, 
        remove_stopwords: bool = True,
        min_length: int = 2,
        remove_numbers: bool = True,
        remove_urls: bool = True
    ):
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.min_length = min_length
        self.remove_numbers = remove_numbers
        self.remove_urls = remove_urls
        self.feature_names_ = []
        
        if nltk:
            self.lemmatizer = WordNetLemmatizer()
            if self.remove_stopwords:
                self.stop_words = set(stopwords.words('english'))
        else:
            self.lemmatizer = None
            self.stop_words = set()

    def fit(self, X: pd.DataFrame, y=None):
        text_cols = X.select_dtypes(include=['object']).columns
        self.feature_names_ = [f"{col}_cleaned" for col in text_cols]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=X.index)
        text_cols = X.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            result[f"{col}_cleaned"] = X[col].astype(str).apply(self._clean_text)
            
        return result

    def _clean_text(self, text: str) -> str:
        if pd.isna(text) or text == 'nan':
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize and process
        if nltk:
            tokens = word_tokenize(text)
        else:
            tokens = text.split()
            
        # Filter tokens
        tokens = [
            word for word in tokens 
            if word.isalpha() and len(word) >= self.min_length
        ]
        
        # Remove stopwords
        if self.remove_stopwords and self.stop_words:
            tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatize
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            
        return ' '.join(tokens)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract comprehensive features from text columns."""
    
    def __init__(
        self,
        vectorizer_type: str = 'tfidf',
        max_features: int = 1000,
        ngram_range: tuple = (1, 2),
        include_sentiment: bool = True,
        include_stats: bool = True
    ):
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.include_sentiment = include_sentiment
        self.include_stats = include_stats
        
        self.vectorizers_ = {}
        self.feature_names_ = []
        self.text_columns_ = []
        
        if nltk and self.include_sentiment:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        else:
            self.sentiment_analyzer = None

    def fit(self, X: pd.DataFrame, y=None):
        self.text_columns_ = X.select_dtypes(include=['object']).columns.tolist()
        self.feature_names_ = []
        
        for col in self.text_columns_:
            # Vectorizer features
            if self.vectorizer_type == 'tfidf':
                vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    stop_words='english'
                )
            else:
                vectorizer = CountVectorizer(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    stop_words='english'
                )
            
            text_data = X[col].astype(str).fillna('')
            vectorizer.fit(text_data)
            self.vectorizers_[col] = vectorizer
            
            # Add vectorizer feature names
            vector_features = [f"{col}_{feat}" for feat in vectorizer.get_feature_names_out()]
            self.feature_names_.extend(vector_features)
            
            # Add statistical features
            if self.include_stats:
                stats_features = [
                    f"{col}_length",
                    f"{col}_word_count",
                    f"{col}_unique_words",
                    f"{col}_avg_word_length"
                ]
                self.feature_names_.extend(stats_features)
            
            # Add sentiment features
            if self.include_sentiment and self.sentiment_analyzer:
                sentiment_features = [
                    f"{col}_sentiment_pos",
                    f"{col}_sentiment_neu", 
                    f"{col}_sentiment_neg",
                    f"{col}_sentiment_compound"
                ]
                self.feature_names_.extend(sentiment_features)
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.text_columns_:
            return pd.DataFrame(index=X.index)
            
        all_features = []
        
        for col in self.text_columns_:
            text_data = X[col].astype(str).fillna('')
            
            # Vectorizer features
            vectorizer = self.vectorizers_[col]
            vector_features = vectorizer.transform(text_data).toarray()
            vector_df = pd.DataFrame(
                vector_features,
                columns=[f"{col}_{feat}" for feat in vectorizer.get_feature_names_out()],
                index=X.index
            )
            all_features.append(vector_df)
            
            # Statistical features
            if self.include_stats:
                stats_df = pd.DataFrame(index=X.index)
                stats_df[f"{col}_length"] = text_data.str.len()
                stats_df[f"{col}_word_count"] = text_data.str.split().str.len()
                stats_df[f"{col}_unique_words"] = text_data.apply(lambda x: len(set(x.split())))
                stats_df[f"{col}_avg_word_length"] = text_data.apply(
                    lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
                )
                all_features.append(stats_df)
            
            # Sentiment features
            if self.include_sentiment and self.sentiment_analyzer:
                sentiment_scores = text_data.apply(self._get_sentiment_scores)
                sentiment_df = pd.DataFrame(
                    list(sentiment_scores),
                    columns=[
                        f"{col}_sentiment_pos",
                        f"{col}_sentiment_neu",
                        f"{col}_sentiment_neg", 
                        f"{col}_sentiment_compound"
                    ],
                    index=X.index
                )
                all_features.append(sentiment_df)
        
        return pd.concat(all_features, axis=1) if all_features else pd.DataFrame(index=X.index)

    def _get_sentiment_scores(self, text: str) -> List[float]:
        if not self.sentiment_analyzer or not text.strip():
            return [0.0, 1.0, 0.0, 0.0]  # neutral default
            
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            return [scores['pos'], scores['neu'], scores['neg'], scores['compound']]
        except:
            return [0.0, 1.0, 0.0, 0.0]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_

class TextTopicExtractor(BaseEstimator, TransformerMixin):
    """Extract topic features using simple clustering on TF-IDF vectors."""
    
    def __init__(self, n_topics: int = 5, max_features: int = 500):
        self.n_topics = n_topics
        self.max_features = max_features
        self.feature_names_ = []
        self.text_columns_ = []
        self.models_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        self.text_columns_ = X.select_dtypes(include=['object']).columns.tolist()
        self.feature_names_ = []
        
        try:
            from sklearn.decomposition import LatentDirichletAllocation
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            logging.warning("Scikit-learn components not available for topic modeling")
            return self
        
        for col in self.text_columns_:
            text_data = X[col].astype(str).fillna('')
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Fit LDA model
            tfidf_matrix = vectorizer.fit_transform(text_data)
            
            if tfidf_matrix.shape[0] > 0:
                lda = LatentDirichletAllocation(
                    n_components=self.n_topics,
                    random_state=42,
                    max_iter=10
                )
                lda.fit(tfidf_matrix)
                
                self.models_[col] = {
                    'vectorizer': vectorizer,
                    'lda': lda
                }
                
                # Add topic feature names
                topic_features = [f"{col}_topic_{i}" for i in range(self.n_topics)]
                self.feature_names_.extend(topic_features)
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.text_columns_ or not self.models_:
            return pd.DataFrame(index=X.index)
            
        all_features = []
        
        for col in self.text_columns_:
            if col not in self.models_:
                continue
                
            text_data = X[col].astype(str).fillna('')
            model_dict = self.models_[col]
            
            try:
                tfidf_matrix = model_dict['vectorizer'].transform(text_data)
                topic_probs = model_dict['lda'].transform(tfidf_matrix)
                
                topic_df = pd.DataFrame(
                    topic_probs,
                    columns=[f"{col}_topic_{i}" for i in range(self.n_topics)],
                    index=X.index
                )
                all_features.append(topic_df)
                
            except Exception as e:
                logging.warning(f"Topic extraction failed for {col}: {e}")
                continue
        
        return pd.concat(all_features, axis=1) if all_features else pd.DataFrame(index=X.index)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_