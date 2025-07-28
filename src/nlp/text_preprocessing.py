"""
Text Preprocessing Transformer for DataInsight AI

This module provides a robust, scikit-learn compatible transformer for cleaning
and normalizing raw text data. It handles common NLP preprocessing steps like
lowercasing, punctuation and stopword removal, and lemmatization.
"""

import logging
import re
import string
from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError:
    raise ImportError(
        "NLTK not found. Please install it and download required data:\n"
        "pip install nltk\n"
        "python -m nltk.downloader punkt stopwords wordnet omw-1.4"
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that performs a sequence of cleaning operations on text data.

    Parameters
    ----------
    lemmatize : bool, default=True
        If True, applies WordNet lemmatization to reduce words to their base form.

    remove_stopwords : bool, default=True
        If True, removes common English stopwords.
    """
    def __init__(self, lemmatize: bool = True, remove_stopwords: bool = True):
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self._setup_nltk_resources()

    def _setup_nltk_resources(self):
        """Initializes NLTK resources needed for processing."""
        self.lemmatizer = WordNetLemmatizer()
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))

    def fit(self, X: pd.Series, y=None):
        """
        Fit method (does nothing, as this is a stateless transformer).
        Returns the instance itself.
        """
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """
        Cleans and transforms the input text series.

        Args:
            X: A pandas Series containing raw text documents.

        Returns:
            A pandas Series with the processed text.
        """
        if not isinstance(X, pd.Series):
            raise TypeError("Input must be a pandas Series.")

        processed_series = X.str.lower()
        processed_series = processed_series.apply(
            lambda text: text.translate(str.maketrans('', '', string.punctuation))
        )
        processed_series = processed_series.apply(self._clean_tokens)
        return processed_series

    def _clean_tokens(self, text: str) -> str:
        """Helper function to process tokens for a single document."""
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalpha()]
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)