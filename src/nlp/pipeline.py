"""NLP Pipeline Construction for DataInsight AI"""

import logging
from typing import Dict, Any, List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .text_preprocessing import (
    TextCleanerTransformer,
    TextFeatureExtractor,
    TextTopicExtractor
)

def create_nlp_pipeline(
    df: pd.DataFrame,
    target_column: str = None,
    config: Dict[str, Any] = None
) -> Pipeline:
    """Build comprehensive NLP pipeline for text analysis and classification."""
    
    config = config or {}
    
    # Identify text columns
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_column and target_column in text_cols:
        text_cols.remove(target_column)
    
    if not text_cols:
        raise ValueError("No text columns found for NLP pipeline")
    
    transformers = []
    
    # Text cleaning
    if config.get('clean_text', True):
        cleaner = TextCleanerTransformer(
            lemmatize=config.get('lemmatize', True),
            remove_stopwords=config.get('remove_stopwords', True),
            min_length=config.get('min_word_length', 2),
            remove_numbers=config.get('remove_numbers', True),
            remove_urls=config.get('remove_urls', True)
        )
        transformers.append(('text_cleaner', cleaner, text_cols))
    
    # Feature extraction
    if config.get('extract_features', True):
        extractor = TextFeatureExtractor(
            vectorizer_type=config.get('vectorizer_type', 'tfidf'),
            max_features=config.get('max_features', 1000),
            ngram_range=tuple(config.get('ngram_range', [1, 2])),
            include_sentiment=config.get('include_sentiment', True),
            include_stats=config.get('include_stats', True)
        )
        transformers.append(('text_features', extractor, text_cols))
    
    # Topic modeling
    if config.get('extract_topics', False):
        topic_extractor = TextTopicExtractor(
            n_topics=config.get('n_topics', 5),
            max_features=config.get('topic_max_features', 500)
        )
        transformers.append(('text_topics', topic_extractor, text_cols))
    
    # Numeric features preprocessing
    numeric_cols = [col for col in df.select_dtypes(include=['number']).columns 
                   if col != target_column]
    
    if numeric_cols:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('numeric', numeric_transformer, numeric_cols))
    
    if not transformers:
        raise ValueError("No valid transformers could be created for NLP pipeline")
    
    # Create preprocessor
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    # Create full pipeline
    if target_column:
        # Classification task
        model = config.get('model') or RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    else:
        # Feature extraction only
        model = 'passthrough'
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline

def detect_text_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze DataFrame to suggest NLP configuration."""
    
    suggestions = {
        'text_columns': [],
        'avg_text_length': {},
        'unique_ratio': {},
        'suggested_max_features': 1000,
        'suggested_ngram_range': [1, 2],
        'suggested_topics': 5
    }
    
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in text_cols:
        # Check if column contains meaningful text
        sample_texts = df[col].astype(str).head(100)
        avg_length = sample_texts.str.len().mean()
        unique_ratio = df[col].nunique() / len(df)
        
        # Consider it a text column if average length > 10 and reasonable uniqueness
        if avg_length > 10 and 0.1 < unique_ratio < 0.95:
            suggestions['text_columns'].append(col)
            suggestions['avg_text_length'][col] = avg_length
            suggestions['unique_ratio'][col] = unique_ratio
    
    # Adjust parameters based on data characteristics
    total_samples = len(df)
    if total_samples > 10000:
        suggestions['suggested_max_features'] = 2000
    elif total_samples < 1000:
        suggestions['suggested_max_features'] = 500
    
    # Suggest topics based on data size and text variety
    if total_samples > 1000 and len(suggestions['text_columns']) > 0:
        suggestions['suggested_topics'] = min(10, max(3, total_samples // 200))
    else:
        suggestions['suggested_topics'] = 3
    
    return suggestions

def create_nlp_config(
    df: pd.DataFrame,
    target_column: str = None,
    task_type: str = 'classification',
    auto_detect: bool = True
) -> Dict[str, Any]:
    """Generate optimal NLP configuration for given data."""
    
    if auto_detect:
        patterns = detect_text_patterns(df)
        
        config = {
            'clean_text': True,
            'lemmatize': True,
            'remove_stopwords': True,
            'min_word_length': 2,
            'remove_numbers': True,
            'remove_urls': True,
            'extract_features': True,
            'vectorizer_type': 'tfidf',
            'max_features': patterns['suggested_max_features'],
            'ngram_range': patterns['suggested_ngram_range'],
            'include_sentiment': True,
            'include_stats': True,
            'extract_topics': len(df) > 500,
            'n_topics': patterns['suggested_topics'],
            'topic_max_features': 500
        }
        
        # Adjust for task type
        if task_type == 'sentiment':
            config['include_sentiment'] = True
            config['vectorizer_type'] = 'tfidf'
        elif task_type == 'topic_modeling':
            config['extract_topics'] = True
            config['n_topics'] = min(20, max(5, len(df) // 100))
        
    else:
        # Basic configuration
        config = {
            'clean_text': True,
            'lemmatize': True,
            'remove_stopwords': True,
            'extract_features': True,
            'vectorizer_type': 'tfidf',
            'max_features': 1000,
            'ngram_range': [1, 2],
            'include_sentiment': False,
            'include_stats': True,
            'extract_topics': False
        }
    
    return config

def create_hybrid_pipeline(
    df: pd.DataFrame,
    text_columns: List[str],
    target_column: str,
    config: Dict[str, Any] = None
) -> Pipeline:
    """Create pipeline for datasets with both text and structured data."""
    
    config = config or {}
    
    transformers = []
    
    # Text processing
    if text_columns:
        text_pipeline_steps = []
        
        # Text cleaning
        if config.get('clean_text', True):
            text_pipeline_steps.append(
                ('cleaner', TextCleanerTransformer())
            )
        
        # Feature extraction
        text_pipeline_steps.append(
            ('extractor', TextFeatureExtractor(
                max_features=config.get('max_features', 500),
                include_sentiment=config.get('include_sentiment', True),
                include_stats=config.get('include_stats', True)
            ))
        )
        
        text_pipeline = Pipeline(text_pipeline_steps)
        transformers.append(('text', text_pipeline, text_columns))
    
    # Numeric features
    numeric_cols = [col for col in df.select_dtypes(include=['number']).columns 
                   if col != target_column]
    
    if numeric_cols:
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('numeric', numeric_pipeline, numeric_cols))
    
    # Categorical features (non-text)
    categorical_cols = [
        col for col in df.select_dtypes(include=['object']).columns 
        if col not in text_columns and col != target_column
    ]
    
    if categorical_cols:
        from sklearn.preprocessing import OneHotEncoder
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('categorical', categorical_pipeline, categorical_cols))
    
    if not transformers:
        raise ValueError("No valid transformers could be created")
    
    # Create preprocessor
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    # Model selection based on target type
    if pd.api.types.is_numeric_dtype(df[target_column]):
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline