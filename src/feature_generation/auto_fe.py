"""Intelligent Automated Feature Engineering Module for DataInsight AI"""

import logging
import psutil
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, f_classif

try:
    import featuretools as ft
    from featuretools.primitives import (
        Count, Mean, Max, Min, Std, Sum, Mode,
        Day, Month, Year, Weekday, IsWeekend,
        Median, NumUnique, Trend
    )
    FEATURETOOLS_AVAILABLE = True
except ImportError:
    ft = None
    FEATURETOOLS_AVAILABLE = False
    logging.warning("Featuretools not available. Using basic feature engineering only.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_TEXT_AVAILABLE = True
except ImportError:
    SKLEARN_TEXT_AVAILABLE = False

class AutomatedFeatureEngineer:
    """Intelligent automated feature generation with domain awareness and memory management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.entityset: Optional[Any] = None
        self.generated_features: List = []
        self.feature_matrix: Optional[pd.DataFrame] = None
        self.domain_patterns = self._init_domain_patterns()
        self.memory_limit_mb = self.config.get('memory_limit_mb', 2000)
        self.max_features = self.config.get('max_features', 1000)
        self.feature_metadata: Dict[str, Dict] = {}
    
    def _init_domain_patterns(self) -> Dict[str, Dict]:
        """Initialize domain-specific feature patterns."""
        return {
            'ecommerce': {
                'patterns': ['price', 'cost', 'revenue', 'quantity', 'order', 'customer'],
                'features': ['rfm_recency', 'rfm_frequency', 'rfm_monetary', 'avg_order_value', 'purchase_velocity']
            },
            'finance': {
                'patterns': ['amount', 'balance', 'transaction', 'account', 'payment'],
                'features': ['amount_zscore', 'transaction_frequency', 'balance_velocity', 'amount_percentile']
            },
            'temporal': {
                'patterns': ['date', 'time', 'created', 'updated', 'timestamp'],
                'features': ['days_since', 'is_weekend', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
            }
        }

    def generate_features(
        self,
        dataframes: Dict[str, pd.DataFrame],
        relationships: List[Tuple[str, str, str, str]] = None,
        target_entity: str = None
    ) -> pd.DataFrame:
        """Generate intelligent features with memory management and domain awareness."""
        
        if not dataframes:
            raise ValueError("dataframes dictionary cannot be empty")
        
        start_memory = self._get_memory_usage()
        logging.info(f"Starting feature generation. Memory usage: {start_memory:.1f}MB")
        
        if len(dataframes) == 1 and not relationships:
            df_name, df = list(dataframes.items())[0]
            result = self._generate_intelligent_features(df)
        elif FEATURETOOLS_AVAILABLE and relationships:
            result = self._generate_relational_features(dataframes, relationships, target_entity)
        else:
            logging.warning("Using basic feature engineering only")
            main_df = list(dataframes.values())[0]
            result = self._generate_intelligent_features(main_df)
        
        result = self._apply_feature_pruning(result)
        
        end_memory = self._get_memory_usage()
        logging.info(f"Feature generation complete. Memory usage: {end_memory:.1f}MB (+{end_memory-start_memory:.1f}MB)")
        
        return result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return psutil.Process().memory_info().rss / 1024 / 1024
    
    def _apply_feature_pruning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prune features based on memory and feature count limits."""
        original_count = len(df.columns)
        
        if len(df.columns) > self.max_features:
            logging.warning(f"Feature count ({len(df.columns)}) exceeds limit ({self.max_features}). Pruning...")
            
            # Keep original columns plus best new features
            original_cols = [col for col in df.columns if not any(
                prefix in col for prefix in ['poly_', 'ratio_', '_binned', 'row_', '_x_', 'temporal_', 'domain_']
            )]
            
            generated_cols = [col for col in df.columns if col not in original_cols]
            
            # Simple pruning: keep features with reasonable variance
            kept_generated = []
            for col in generated_cols:
                if df[col].var() > 1e-10 and not df[col].isnull().all():
                    kept_generated.append(col)
                    if len(kept_generated) >= (self.max_features - len(original_cols)):
                        break
            
            df = df[original_cols + kept_generated]
            logging.info(f"Pruned features: {original_count} → {len(df.columns)}")
        
        return df

    def _generate_intelligent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate intelligent features with domain awareness."""
        result_df = df.copy()
        initial_count = len(df.columns)
        
        # Detect domain and add domain-specific features
        domain = self._detect_domain(df)
        if domain:
            logging.info(f"Detected domain: {domain}")
            result_df = self._add_domain_features(result_df, domain)
        
        # Add temporal features for datetime columns
        result_df = self._add_temporal_features(result_df)
        
        # Add text features for text columns
        result_df = self._add_text_features(result_df)
        
        # Add advanced temporal features
        result_df = self._add_advanced_temporal_features(result_df)
        
        # Add enhanced mathematical features
        result_df = self._add_mathematical_features(result_df)
        
        # Add statistical aggregation features
        result_df = self._add_statistical_features(result_df)
        
        logging.info(f"Generated {len(result_df.columns) - initial_count} intelligent features")
        return result_df
    
    def _detect_domain(self, df: pd.DataFrame) -> Optional[str]:
        """Detect domain based on column names and patterns."""
        col_names_lower = [col.lower() for col in df.columns]
        domain_scores = {}
        
        for domain, info in self.domain_patterns.items():
            score = 0
            for pattern in info['patterns']:
                matches = sum(1 for col in col_names_lower if pattern in col)
                score += matches
            domain_scores[domain] = score
        
        best_domain = max(domain_scores, key=domain_scores.get) if domain_scores else None
        return best_domain if domain_scores.get(best_domain, 0) > 0 else None
    
    def _add_domain_features(self, df: pd.DataFrame, domain: str) -> pd.DataFrame:
        """Add domain-specific features."""
        if domain == 'ecommerce':
            df = self._add_ecommerce_features(df)
        elif domain == 'finance':
            df = self._add_finance_features(df)
        return df
    
    def _add_ecommerce_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add e-commerce specific features."""
        # Find amount/price columns
        amount_cols = [col for col in df.columns if any(
            pattern in col.lower() for pattern in ['price', 'amount', 'cost', 'revenue']
        )]
        
        # Find quantity columns
        qty_cols = [col for col in df.columns if any(
            pattern in col.lower() for pattern in ['quantity', 'qty', 'count']
        )]
        
        for amt_col in amount_cols[:2]:  # Limit to prevent explosion
            if pd.api.types.is_numeric_dtype(df[amt_col]):
                # Price percentiles
                df[f'domain_{amt_col}_percentile'] = df[amt_col].rank(pct=True)
                
                # Price z-score
                df[f'domain_{amt_col}_zscore'] = (df[amt_col] - df[amt_col].mean()) / df[amt_col].std()
                
                # Price bins
                try:
                    df[f'domain_{amt_col}_tier'] = pd.qcut(df[amt_col], q=5, labels=False, duplicates='drop')
                except:
                    pass
        
        # Cross-column features
        if len(amount_cols) >= 1 and len(qty_cols) >= 1:
            amt_col, qty_col = amount_cols[0], qty_cols[0]
            if pd.api.types.is_numeric_dtype(df[amt_col]) and pd.api.types.is_numeric_dtype(df[qty_col]):
                df[f'domain_unit_price'] = df[amt_col] / df[qty_col].replace(0, np.nan)
        
        return df
    
    def _add_finance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add finance-specific features."""
        # Find amount/balance columns
        amount_cols = [col for col in df.columns if any(
            pattern in col.lower() for pattern in ['amount', 'balance', 'transaction']
        )]
        
        for col in amount_cols[:3]:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Rolling statistics (if we have enough data)
                if len(df) > 10:
                    window = min(7, len(df) // 4)
                    df[f'domain_{col}_rolling_mean'] = df[col].rolling(window=window, min_periods=1).mean()
                    df[f'domain_{col}_rolling_std'] = df[col].rolling(window=window, min_periods=1).std()
                
                # Amount volatility
                df[f'domain_{col}_volatility'] = abs(df[col] - df[col].mean()) / df[col].std()
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features for datetime columns."""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_cols:
            dt_series = pd.to_datetime(df[col])
            
            # Basic temporal features
            df[f'temporal_{col}_year'] = dt_series.dt.year
            df[f'temporal_{col}_month'] = dt_series.dt.month
            df[f'temporal_{col}_day'] = dt_series.dt.day
            df[f'temporal_{col}_dayofweek'] = dt_series.dt.dayofweek
            df[f'temporal_{col}_is_weekend'] = (dt_series.dt.dayofweek >= 5).astype(int)
            
            # Cyclical encoding for better ML performance
            df[f'temporal_{col}_month_sin'] = np.sin(2 * np.pi * dt_series.dt.month / 12)
            df[f'temporal_{col}_month_cos'] = np.cos(2 * np.pi * dt_series.dt.month / 12)
            df[f'temporal_{col}_hour_sin'] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
            df[f'temporal_{col}_hour_cos'] = np.cos(2 * np.pi * dt_series.dt.hour / 24)
            
            # Time since reference (days since first date)
            if not dt_series.isna().all():
                first_date = dt_series.min()
                df[f'temporal_{col}_days_since'] = (dt_series - first_date).dt.days
        
        return df
    
    def _add_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced text feature engineering."""
        text_cols = df.select_dtypes(include=['object']).columns
        text_cols = [col for col in text_cols if self._is_text_column(df[col])]
        
        if not text_cols:
            return df
            
        for col in text_cols[:3]:  # Limit to prevent explosion
            text_series = df[col].fillna('').astype(str)
            
            # Basic text statistics
            df[f'text_{col}_length'] = text_series.str.len()
            df[f'text_{col}_word_count'] = text_series.str.split().str.len()
            df[f'text_{col}_punct_ratio'] = text_series.str.count(r'[^\w\s]') / (text_series.str.len() + 1)
            df[f'text_{col}_upper_ratio'] = text_series.str.count(r'[A-Z]') / (text_series.str.len() + 1)
            
            # Sentiment analysis if available
            if TEXTBLOB_AVAILABLE and len(text_series) <= 1000:  # Limit for performance
                try:
                    sentiments = text_series.apply(lambda x: TextBlob(str(x)[:500]).sentiment.polarity if x else 0)
                    df[f'text_{col}_sentiment'] = sentiments
                    
                    subjectivity = text_series.apply(lambda x: TextBlob(str(x)[:500]).sentiment.subjectivity if x else 0)
                    df[f'text_{col}_subjectivity'] = subjectivity
                except:
                    logging.warning(f"Failed to compute sentiment for {col}")
            
            # Topic modeling features (basic)
            if SKLEARN_TEXT_AVAILABLE and len(text_series.unique()) > 10:
                try:
                    # Create topic features using simple TF-IDF + clustering
                    vectorizer = TfidfVectorizer(max_features=50, stop_words='english', 
                                               min_df=2, max_df=0.8, ngram_range=(1, 2))
                    tfidf_matrix = vectorizer.fit_transform(text_series)
                    
                    # Simple topic assignment using k-means
                    from sklearn.cluster import KMeans
                    n_topics = min(5, max(2, len(text_series.unique()) // 20))
                    kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
                    topics = kmeans.fit_predict(tfidf_matrix)
                    df[f'text_{col}_topic'] = topics
                    
                    # Topic confidence (distance to centroid)
                    distances = kmeans.transform(tfidf_matrix)
                    df[f'text_{col}_topic_confidence'] = 1 / (1 + distances.min(axis=1))
                    
                except Exception as e:
                    logging.warning(f"Failed to compute topics for {col}: {e}")
        
        return df
    
    def _is_text_column(self, series: pd.Series) -> bool:
        """Check if column contains meaningful text data."""
        if series.dtype != 'object':
            return False
            
        sample_values = series.dropna().head(100)
        if len(sample_values) == 0:
            return False
            
        # Check if values look like text (average length > 10, contains spaces)
        avg_length = sample_values.astype(str).str.len().mean()
        has_spaces = sample_values.astype(str).str.contains(' ').mean()
        
        return avg_length > 10 and has_spaces > 0.3
    
    def _add_advanced_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced temporal features with seasonal patterns and optimal lags."""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_cols:
            dt_series = pd.to_datetime(df[col])
            
            # Quarter and season features
            df[f'temporal_{col}_quarter'] = dt_series.dt.quarter
            month = dt_series.dt.month
            df[f'temporal_{col}_season'] = ((month - 1) // 3) + 1
            
            # Holiday and business day features
            df[f'temporal_{col}_is_month_start'] = dt_series.dt.is_month_start.astype(int)
            df[f'temporal_{col}_is_month_end'] = dt_series.dt.is_month_end.astype(int)
            df[f'temporal_{col}_is_quarter_start'] = dt_series.dt.is_quarter_start.astype(int)
            df[f'temporal_{col}_is_quarter_end'] = dt_series.dt.is_quarter_end.astype(int)
            
            # Advanced cyclical encoding with multiple periods
            df[f'temporal_{col}_dayofyear_sin'] = np.sin(2 * np.pi * dt_series.dt.dayofyear / 365.25)
            df[f'temporal_{col}_dayofyear_cos'] = np.cos(2 * np.pi * dt_series.dt.dayofyear / 365.25)
            df[f'temporal_{col}_week_sin'] = np.sin(2 * np.pi * dt_series.dt.isocalendar().week / 52)
            df[f'temporal_{col}_week_cos'] = np.cos(2 * np.pi * dt_series.dt.isocalendar().week / 52)
            
            # Time-based aggregations if we have numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0 and len(df) > 30:  # Sufficient data for patterns
                # Sort by datetime for lag features
                sorted_idx = dt_series.argsort()
                df_sorted = df.iloc[sorted_idx]
                
                # Add lag features for top numeric columns
                for num_col in numeric_cols[:3]:  # Limit to prevent explosion
                    if num_col.startswith(('temporal_', 'text_', 'domain_')):
                        continue
                        
                    # Multiple lag periods
                    for lag in [1, 7, 30]:
                        if lag < len(df):
                            df[f'temporal_{col}_{num_col}_lag_{lag}'] = df_sorted[num_col].shift(lag).iloc[sorted_idx.argsort()]
                    
                    # Rolling statistics 
                    for window in [7, 30]:
                        if window < len(df):
                            rolling_mean = df_sorted[num_col].rolling(window=window, min_periods=1).mean()
                            rolling_std = df_sorted[num_col].rolling(window=window, min_periods=1).std()
                            df[f'temporal_{col}_{num_col}_rolling_mean_{window}'] = rolling_mean.iloc[sorted_idx.argsort()]
                            df[f'temporal_{col}_{num_col}_rolling_std_{window}'] = rolling_std.iloc[sorted_idx.argsort()]
        
        return df
    
    def _add_mathematical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced mathematical transformation features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if not col.startswith(('temporal_', 'domain_', 'text_'))]
        
        if len(numeric_cols) < 2:
            return df
        
        # Intelligent column selection based on variance and correlation
        selected_cols = self._select_best_numeric_columns(df, numeric_cols)
        
        # Enhanced ratio features with zero-handling
        if self.config.get('ratio_features', True):
            for i in range(min(4, len(selected_cols))):
                for j in range(i + 1, min(i + 5, len(selected_cols))):
                    col1, col2 = selected_cols[i], selected_cols[j]
                    
                    # Safe division with multiple strategies
                    denominator = df[col2].replace(0, np.nan)
                    if not denominator.isna().all():
                        df[f"ratio_{col1}_{col2}"] = df[col1] / denominator
                        
                        # Log ratio for positive values
                        if (df[col1] > 0).any() and (df[col2] > 0).any():
                            safe_col1 = df[col1].clip(lower=1e-10)
                            safe_col2 = df[col2].clip(lower=1e-10)
                            df[f"log_ratio_{col1}_{col2}"] = np.log(safe_col1) - np.log(safe_col2)
        
        # Enhanced interaction features
        if self.config.get('interaction_features', True) and len(selected_cols) >= 2:
            for i in range(min(3, len(selected_cols))):
                for j in range(i + 1, min(i + 3, len(selected_cols))):
                    col1, col2 = selected_cols[i], selected_cols[j]
                    
                    # Multiplicative interaction
                    df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                    
                    # Additive interaction 
                    df[f"{col1}_plus_{col2}"] = df[col1] + df[col2]
                    
                    # Difference features
                    df[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
                    
                    # Min/Max features
                    df[f"{col1}_{col2}_min"] = np.minimum(df[col1], df[col2])
                    df[f"{col1}_{col2}_max"] = np.maximum(df[col1], df[col2])
        
        # Polynomial features for top columns (intelligent degree selection)
        if self.config.get('polynomial_features', True) and len(selected_cols) >= 1:
            top_cols = selected_cols[:3]  # Limit to top 3 columns
            degree = 2 if len(df) > 1000 else 2  # Conservative degree
            
            try:
                poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
                poly_features = poly.fit_transform(df[top_cols])
                poly_names = poly.get_feature_names_out(top_cols)
                
                # Add only new polynomial features (exclude original features)
                for i, name in enumerate(poly_names):
                    if name not in top_cols:  # Skip original features
                        df[f"poly_{name}"] = poly_features[:, i]
            except Exception as e:
                logging.warning(f"Failed to generate polynomial features: {e}")
        
        return df
    
    def _select_best_numeric_columns(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
        """Select best numeric columns based on variance and correlation structure."""
        if len(numeric_cols) <= 5:
            return numeric_cols
            
        scores = {}
        
        for col in numeric_cols:
            score = 0
            series = df[col]
            
            # Variance score (normalized)
            if series.var() > 0:
                cv = series.std() / abs(series.mean() + 1e-10)  # Coefficient of variation
                score += min(cv, 5)  # Cap at 5 to avoid extreme values
            
            # Non-missing data score
            score += (1 - series.isnull().mean()) * 2
            
            # Avoid highly correlated features
            for other_col in numeric_cols:
                if other_col != col and other_col in scores:
                    corr = abs(df[col].corr(df[other_col]))
                    if not np.isnan(corr) and corr > 0.95:
                        score -= 1
            
            scores[col] = score
        
        # Return top columns
        sorted_cols = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_cols[:8]
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical aggregation features."""
        if not self.config.get('statistical_features', True):
            return df
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if not col.startswith(('temporal_', 'domain_', 'ratio_', 'row_'))]
        
        if len(numeric_cols) >= 2:
            numeric_data = df[numeric_cols]
            df['row_mean'] = numeric_data.mean(axis=1)
            df['row_std'] = numeric_data.std(axis=1)
            df['row_max'] = numeric_data.max(axis=1)
            df['row_min'] = numeric_data.min(axis=1)
            df['row_range'] = df['row_max'] - df['row_min']
        
        return df

    def _generate_relational_features(
        self,
        dataframes: Dict[str, pd.DataFrame],
        relationships: List[Tuple[str, str, str, str]],
        target_entity: str
    ) -> pd.DataFrame:
        """Generate features using featuretools DFS."""
        
        if not FEATURETOOLS_AVAILABLE:
            raise ImportError("Featuretools is required for relational feature generation")
        
        logging.info("Starting Deep Feature Synthesis...")
        
        self.entityset = ft.EntitySet(id="datainsight_es")
        
        # Add dataframes to entityset
        for entity_name, df_original in dataframes.items():
            df = df_original.copy()
            index_col = self._find_best_index_column(df, entity_name)
            
            self.entityset.add_dataframe(
                dataframe_name=entity_name,
                dataframe=df,
                index=index_col,
                make_index=False
            )
        
        # Add relationships
        for parent_entity, parent_variable, child_entity, child_variable in relationships:
            self.entityset.add_relationship(
                parent_dataframe_name=parent_entity,
                parent_column_name=parent_variable,
                child_dataframe_name=child_entity,
                child_column_name=child_variable
            )
        
        # Enhanced primitives for better feature generation
        agg_primitives = [Sum, Mean, Max, Min, Std, Count, Mode, Median, NumUnique]
        trans_primitives = [Day, Month, Year, Weekday, IsWeekend, Trend]
        
        # Run DFS
        self.feature_matrix, self.generated_features = ft.dfs(
            entityset=self.entityset,
            target_dataframe_name=target_entity,
            agg_primitives=agg_primitives,
            trans_primitives=trans_primitives,
            verbose=self.config.get('dfs_verbose', False),
            max_depth=self.config.get('dfs_max_depth', 2),
            n_jobs=self.config.get('dfs_n_jobs', 1)
        )
        
        logging.info(f"DFS complete. Generated {len(self.generated_features)} features")
        return self.feature_matrix.reset_index()

    def _find_best_index_column(self, df: pd.DataFrame, entity_name: str) -> str:
        """Identify the best primary key column for an entity."""
        
        scores = {}
        total_rows = len(df)
        
        if total_rows == 0:
            raise ValueError(f"DataFrame for entity '{entity_name}' is empty")
        
        for col in df.columns:
            scores[col] = 0
            series = df[col]
            
            # High score for unique columns
            if series.nunique() == total_rows:
                scores[col] += 100
            
            # Bonus for ID-like names
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['id', 'key', 'pk', 'uuid']):
                scores[col] += 50
            
            # Prefer integer or object types
            if pd.api.types.is_integer_dtype(series.dtype):
                scores[col] += 20
            elif pd.api.types.is_object_dtype(series.dtype):
                scores[col] += 15
            
            # Bonus for no nulls
            if series.isnull().sum() == 0:
                scores[col] += 10
        
        if not scores:
            raise ValueError(f"Could not score any columns for entity '{entity_name}'")
        
        best_column = max(scores, key=scores.get)
        
        if df[best_column].nunique() != total_rows:
            logging.warning(
                f"Index column '{best_column}' for '{entity_name}' is not fully unique. "
                "This may cause issues in feature generation."
            )
        
        return best_column

class InteractionFeatureGenerator(BaseEstimator, TransformerMixin):
    """Generate interaction features between numeric columns."""
    
    def __init__(self, max_interactions: int = 20):
        self.max_interactions = max_interactions
        self.feature_names_ = []
        self.interaction_pairs_ = []

    def fit(self, X: pd.DataFrame, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if y is not None and len(numeric_cols) > 1:
            correlations = []
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    interaction = X[col1] * X[col2]
                    corr = abs(np.corrcoef(interaction, y)[0, 1])
                    if not np.isnan(corr):
                        correlations.append((corr, col1, col2))
            
            correlations.sort(reverse=True)
            self.interaction_pairs_ = [
                (col1, col2) for _, col1, col2 in correlations[:self.max_interactions]
            ]
        else:
            import itertools
            pairs = list(itertools.combinations(numeric_cols, 2))
            self.interaction_pairs_ = pairs[:self.max_interactions]
        
        self.feature_names_ = [f"{col1}_x_{col2}" for col1, col2 in self.interaction_pairs_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=X.index)
        
        for col1, col2 in self.interaction_pairs_:
            if col1 in X.columns and col2 in X.columns:
                result[f"{col1}_x_{col2}"] = X[col1] * X[col2]
        
        return result

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_

def create_feature_engineering_config(
    df: pd.DataFrame,
    task_type: str = 'regression',
    complexity: str = 'medium'
) -> Dict[str, Any]:
    """Generate intelligent feature engineering configuration based on data characteristics."""
    
    n_samples, n_features = df.shape
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    datetime_cols = len(df.select_dtypes(include=['datetime64']).columns)
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    config = {
        'ratio_features': True,
        'statistical_features': True,
        'interaction_features': numeric_cols >= 3,
        'polynomial_features': numeric_cols >= 2 and n_samples > 100,
        'temporal_features': datetime_cols > 0,
        'text_features': True,
        'domain_features': True,
        'memory_limit_mb': 2000,
        'max_features': 1000,
        'dfs_max_depth': 2,
        'dfs_chunk_size': 5000 if n_samples > 5000 else None
    }
    
    if memory_mb > 100:  
        config['max_features'] = min(500, config['max_features'])
        config['memory_limit_mb'] = 1500
    elif n_samples < 1000:  
        config['max_features'] = min(200, config['max_features'])
        config['interaction_features'] = numeric_cols >= 2
    
    if task_type == 'classification' and n_samples < 5000:
        config['max_features'] = min(300, config['max_features'])
    
    if complexity == 'high':
        config['max_features'] = min(2000, config['max_features'])
        config['memory_limit_mb'] = 3000
    elif complexity == 'low':
        config['max_features'] = min(100, config['max_features'])
        config['interaction_features'] = False
        config['domain_features'] = False
    
    logging.info(f"Feature engineering config: max_features={config['max_features']}, "
                f"memory_limit={config['memory_limit_mb']}MB")
    
    return config
AutoFeatureEngineer = AutomatedFeatureEngineer