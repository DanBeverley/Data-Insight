"""Advanced Missing Value Intelligence System for DataInsight AI"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from datetime import datetime

from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

class MissingPattern(Enum):
    MCAR = "missing_completely_at_random"  # Missing Completely At Random
    MAR = "missing_at_random"              # Missing At Random
    MNAR = "missing_not_at_random"         # Missing Not At Random

class ImputationStrategy(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    KNN = "knn"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"
    PREDICT = "predict"
    DOMAIN_SPECIFIC = "domain_specific"

@dataclass
class MissingAnalysis:
    column: str
    missing_count: int
    missing_percentage: float
    pattern_type: MissingPattern
    pattern_confidence: float
    correlations: Dict[str, float]
    recommended_strategy: ImputationStrategy
    imputation_confidence: float
    metadata: Dict[str, Any]

@dataclass
class ImputationResult:
    column: str
    strategy_used: ImputationStrategy
    values_imputed: int
    imputation_quality: float  # 0-1 quality score
    before_stats: Dict[str, Any]
    after_stats: Dict[str, Any]
    cross_validation_score: Optional[float]
    metadata: Dict[str, Any]

class AdvancedMissingValueIntelligence:
    """Intelligent missing value analysis and imputation system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.missing_analyses: Dict[str, MissingAnalysis] = {}
        self.imputation_history: List[ImputationResult] = []
        self.domain_knowledge: Dict[str, Any] = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for missing value intelligence."""
        return {
            'min_correlation_threshold': 0.3,  # Minimum correlation for pattern detection
            'mcar_p_value_threshold': 0.05,    # P-value threshold for MCAR test
            'pattern_confidence_threshold': 0.7,  # Minimum confidence for pattern classification
            'imputation_quality_threshold': 0.6,  # Minimum quality for accepting imputation
            'knn_neighbors': 5,                    # Default KNN neighbors
            'cross_validation_folds': 3,           # CV folds for strategy evaluation
            'max_categorical_unique': 50,          # Max unique values for categorical treatment
            'missing_threshold_analysis': 0.01,    # Min missing rate for detailed analysis (1%)
            'correlation_analysis_sample_size': 1000,  # Sample size for correlation analysis
            'domain_specific_rules': {
                'age': {'min': 0, 'max': 120, 'default': 'median'},
                'salary': {'min': 0, 'max': 1000000, 'default': 'median'},
                'price': {'min': 0, 'default': 'median'},
                'count': {'min': 0, 'default': 'zero_fill'},
                'percentage': {'min': 0, 'max': 100, 'default': 'mean'}
            }
        }
    
    def analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, MissingAnalysis]:
        """
        Comprehensive analysis of missing value patterns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to their missing analysis
        """
        logging.info(f"Analyzing missing value patterns in {len(df.columns)} columns")
        
        self.missing_analyses = {}
        
        # Get columns with missing values
        missing_info = df.isnull().sum()
        columns_with_missing = missing_info[missing_info > 0].index.tolist()
        
        if not columns_with_missing:
            logging.info("No missing values found in dataset")
            return self.missing_analyses
        
        logging.info(f"Found missing values in {len(columns_with_missing)} columns")
        
        # Sample data if too large for correlation analysis
        analysis_df = df
        if len(df) > self.config['correlation_analysis_sample_size']:
            analysis_df = df.sample(n=self.config['correlation_analysis_sample_size'], random_state=42)
        
        for column in columns_with_missing:
            missing_count = missing_info[column]
            missing_percentage = (missing_count / len(df)) * 100
            
            # Skip if missing rate is too low for meaningful analysis
            if missing_percentage < self.config['missing_threshold_analysis'] * 100:
                continue
            
            logging.info(f"Analyzing {column}: {missing_percentage:.2f}% missing")
            
            # Determine missing pattern
            pattern_type, pattern_confidence = self._classify_missing_pattern(analysis_df, column)
            
            # Find correlations with other variables
            correlations = self._find_missing_correlations(analysis_df, column)
            
            # Recommend imputation strategy
            recommended_strategy, imputation_confidence = self._recommend_imputation_strategy(
                analysis_df, column, pattern_type, correlations
            )
            
            # Additional metadata
            metadata = self._collect_column_metadata(df, column)
            
            self.missing_analyses[column] = MissingAnalysis(
                column=column,
                missing_count=missing_count,
                missing_percentage=missing_percentage,
                pattern_type=pattern_type,
                pattern_confidence=pattern_confidence,
                correlations=correlations,
                recommended_strategy=recommended_strategy,
                imputation_confidence=imputation_confidence,
                metadata=metadata
            )
        
        logging.info(f"Completed missing pattern analysis for {len(self.missing_analyses)} columns")
        return self.missing_analyses
    
    def intelligent_imputation(self, df: pd.DataFrame, 
                             strategies: Optional[Dict[str, ImputationStrategy]] = None,
                             validate_quality: bool = True) -> Tuple[pd.DataFrame, Dict[str, ImputationResult]]:
        """
        Perform intelligent imputation with quality validation.
        
        Args:
            df: DataFrame to impute
            strategies: Optional custom strategies per column
            validate_quality: Whether to validate imputation quality
            
        Returns:
            Tuple of (imputed_dataframe, imputation_results)
        """
        logging.info(f"Starting intelligent imputation on {len(df.columns)} columns")
        
        if not self.missing_analyses:
            self.analyze_missing_patterns(df)
        
        strategies = strategies or {}
        imputation_results = {}
        df_imputed = df.copy()
        
        # Process each column with missing values
        for column, analysis in self.missing_analyses.items():
            if column not in df_imputed.columns:
                continue
                
            # Use custom strategy if provided, otherwise use recommended
            strategy = strategies.get(column, analysis.recommended_strategy)
            
            logging.info(f"Imputing {column} using {strategy.value} strategy")
            
            # Store before stats
            before_stats = self._calculate_column_stats(df_imputed[column])
            
            # Perform imputation
            try:
                imputed_values, quality_score = self._apply_imputation_strategy(
                    df_imputed, column, strategy, analysis
                )
                
                # Update dataframe
                missing_mask = df_imputed[column].isnull()
                df_imputed.loc[missing_mask, column] = imputed_values[missing_mask]
                
                # Calculate after stats
                after_stats = self._calculate_column_stats(df_imputed[column])
                
                # Cross-validation score if requested
                cv_score = None
                if validate_quality and quality_score >= self.config['imputation_quality_threshold']:
                    cv_score = self._cross_validate_imputation(df, column, strategy)
                
                # Store results
                imputation_results[column] = ImputationResult(
                    column=column,
                    strategy_used=strategy,
                    values_imputed=int(missing_mask.sum()),
                    imputation_quality=quality_score,
                    before_stats=before_stats,
                    after_stats=after_stats,
                    cross_validation_score=cv_score,
                    metadata={
                        'original_missing_count': analysis.missing_count,
                        'pattern_type': analysis.pattern_type.value,
                        'pattern_confidence': analysis.pattern_confidence
                    }
                )
                
                logging.info(f"Successfully imputed {column} with quality score {quality_score:.3f}")
                
            except Exception as e:
                logging.error(f"Failed to impute {column}: {e}")
                # Fallback to simple strategy
                try:
                    fallback_strategy = self._get_fallback_strategy(df_imputed[column])
                    imputed_values, quality_score = self._apply_imputation_strategy(
                        df_imputed, column, fallback_strategy, analysis
                    )
                    
                    missing_mask = df_imputed[column].isnull()
                    df_imputed.loc[missing_mask, column] = imputed_values[missing_mask]
                    
                    imputation_results[column] = ImputationResult(
                        column=column,
                        strategy_used=fallback_strategy,
                        values_imputed=int(missing_mask.sum()),
                        imputation_quality=quality_score,
                        before_stats=before_stats,
                        after_stats=self._calculate_column_stats(df_imputed[column]),
                        cross_validation_score=None,
                        metadata={'fallback_used': True, 'original_error': str(e)}
                    )
                    
                    logging.info(f"Used fallback strategy {fallback_strategy.value} for {column}")
                    
                except Exception as e2:
                    logging.error(f"Fallback imputation also failed for {column}: {e2}")
                    continue
        
        # Store in history
        self.imputation_history.extend(imputation_results.values())
        
        logging.info(f"Completed imputation for {len(imputation_results)} columns")
        return df_imputed, imputation_results
    
    def _classify_missing_pattern(self, df: pd.DataFrame, column: str) -> Tuple[MissingPattern, float]:
        """Classify the missing data pattern for a column."""
        missing_mask = df[column].isnull()
        
        if missing_mask.sum() == 0:
            return MissingPattern.MCAR, 1.0
        
        # Test for MCAR using Little's MCAR test approximation
        # This is a simplified version - in practice, you'd use more sophisticated tests
        other_columns = [col for col in df.columns if col != column and df[col].dtype in ['int64', 'float64']]
        
        if len(other_columns) == 0:
            # No numeric columns to test against, assume MCAR
            return MissingPattern.MCAR, 0.5
        
        # Sample correlation analysis
        try:
            correlations = []
            for other_col in other_columns[:10]:  # Limit to first 10 for efficiency
                if df[other_col].isnull().sum() == len(df):  # Skip if all missing
                    continue
                    
                # Create binary missing indicator
                missing_indicator = missing_mask.astype(int)
                
                # Calculate correlation between missingness and other variable
                non_missing_other = df[other_col].dropna()
                if len(non_missing_other) > 10:
                    # Point-biserial correlation
                    corr = self._point_biserial_correlation(missing_indicator, df[other_col])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if not correlations:
                return MissingPattern.MCAR, 0.5
            
            max_correlation = max(correlations)
            avg_correlation = np.mean(correlations)
            
            # Classification logic
            if max_correlation < 0.1 and avg_correlation < 0.05:
                return MissingPattern.MCAR, 0.8
            elif max_correlation < 0.3:
                return MissingPattern.MAR, 0.7
            else:
                return MissingPattern.MNAR, 0.6
                
        except Exception as e:
            logging.warning(f"Failed to classify missing pattern for {column}: {e}")
            return MissingPattern.MCAR, 0.3  # Default to MCAR with low confidence
    
    def _find_missing_correlations(self, df: pd.DataFrame, column: str) -> Dict[str, float]:
        """Find correlations between missingness patterns."""
        missing_mask = df[column].isnull()
        correlations = {}
        
        for other_col in df.columns:
            if other_col == column:
                continue
                
            other_missing_mask = df[other_col].isnull()
            
            # Skip if no variation in missing patterns
            if other_missing_mask.sum() == 0 or other_missing_mask.sum() == len(df):
                continue
            
            try:
                # Calculate correlation between missing patterns
                corr = np.corrcoef(missing_mask.astype(int), other_missing_mask.astype(int))[0, 1]
                if not np.isnan(corr) and abs(corr) > self.config['min_correlation_threshold']:
                    correlations[other_col] = corr
            except:
                continue
        
        return correlations
    
    def _recommend_imputation_strategy(self, df: pd.DataFrame, column: str, 
                                     pattern_type: MissingPattern,
                                     correlations: Dict[str, float]) -> Tuple[ImputationStrategy, float]:
        """Recommend optimal imputation strategy based on analysis."""
        series = df[column]
        confidence = 0.5  # Base confidence
        
        # Check domain-specific rules first
        for domain_key, rules in self.config['domain_specific_rules'].items():
            if domain_key.lower() in column.lower():
                strategy_name = rules.get('default', 'median')
                try:
                    strategy = ImputationStrategy(strategy_name)
                    return strategy, 0.8
                except ValueError:
                    pass  # Invalid strategy name, continue with analysis
        
        # Determine strategy based on data type and pattern
        if pd.api.types.is_numeric_dtype(series):
            # Numeric columns
            non_missing = series.dropna()
            
            if len(non_missing) == 0:
                return ImputationStrategy.MEAN, 0.3
            
            # Check distribution for strategy selection
            skewness = stats.skew(non_missing)
            
            if pattern_type == MissingPattern.MCAR:
                if abs(skewness) < 0.5:  # Roughly normal
                    confidence = 0.8
                    return ImputationStrategy.MEAN, confidence
                else:  # Skewed
                    confidence = 0.7
                    return ImputationStrategy.MEDIAN, confidence
                    
            elif pattern_type == MissingPattern.MAR:
                if len(correlations) > 0:
                    confidence = 0.7
                    return ImputationStrategy.KNN, confidence
                else:
                    confidence = 0.6
                    return ImputationStrategy.PREDICT, confidence
                    
            else:  # MNAR
                confidence = 0.5
                return ImputationStrategy.MEDIAN, confidence  # Conservative approach
        
        else:
            # Categorical columns
            non_missing = series.dropna()
            unique_values = non_missing.nunique()
            
            if unique_values <= self.config['max_categorical_unique']:
                if pattern_type == MissingPattern.MCAR:
                    confidence = 0.7
                    return ImputationStrategy.MODE, confidence
                elif pattern_type == MissingPattern.MAR and len(correlations) > 0:
                    confidence = 0.6
                    return ImputationStrategy.KNN, confidence
                else:
                    confidence = 0.5
                    return ImputationStrategy.MODE, confidence
            else:
                # High cardinality categorical
                confidence = 0.4
                return ImputationStrategy.MODE, confidence
    
    def _apply_imputation_strategy(self, df: pd.DataFrame, column: str, 
                                 strategy: ImputationStrategy,
                                 analysis: MissingAnalysis) -> Tuple[pd.Series, float]:
        """Apply the specified imputation strategy."""
        series = df[column].copy()
        quality_score = 0.5  # Default quality
        
        if strategy == ImputationStrategy.MEAN:
            if pd.api.types.is_numeric_dtype(series):
                fill_value = series.mean()
                series = series.fillna(fill_value)
                quality_score = 0.7
            else:
                raise ValueError("Mean imputation not applicable to non-numeric data")
        
        elif strategy == ImputationStrategy.MEDIAN:
            if pd.api.types.is_numeric_dtype(series):
                fill_value = series.median()
                series = series.fillna(fill_value)
                quality_score = 0.8
            else:
                raise ValueError("Median imputation not applicable to non-numeric data")
        
        elif strategy == ImputationStrategy.MODE:
            mode_value = series.mode()
            if not mode_value.empty:
                fill_value = mode_value.iloc[0]
                series = series.fillna(fill_value)
                quality_score = 0.6
            else:
                # Fallback to forward fill
                series = series.fillna(method='ffill').fillna(method='bfill')
                quality_score = 0.4
        
        elif strategy == ImputationStrategy.KNN:
            quality_score = self._knn_imputation(df, column, series, analysis)
        
        elif strategy == ImputationStrategy.FORWARD_FILL:
            series = series.fillna(method='ffill')
            quality_score = 0.5
        
        elif strategy == ImputationStrategy.BACKWARD_FILL:
            series = series.fillna(method='bfill')
            quality_score = 0.5
        
        elif strategy == ImputationStrategy.INTERPOLATE:
            if pd.api.types.is_numeric_dtype(series):
                series = series.interpolate()
                quality_score = 0.7
            else:
                raise ValueError("Interpolation not applicable to non-numeric data")
        
        elif strategy == ImputationStrategy.PREDICT:
            quality_score = self._predictive_imputation(df, column, series, analysis)
        
        elif strategy == ImputationStrategy.DOMAIN_SPECIFIC:
            series, quality_score = self._domain_specific_imputation(df, column, series, analysis)
        
        # Handle any remaining missing values with fallback
        if series.isnull().any():
            fallback_strategy = self._get_fallback_strategy(series)
            series, _ = self._apply_imputation_strategy(df, column, fallback_strategy, analysis)
            quality_score *= 0.8  # Reduce quality score for using fallback
        
        return series, quality_score
    
    def _knn_imputation(self, df: pd.DataFrame, column: str, series: pd.Series,
                       analysis: MissingAnalysis) -> float:
        """Perform KNN imputation."""
        try:
            # Select numeric columns for KNN
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if column not in numeric_cols and not pd.api.types.is_numeric_dtype(series):
                # Encode categorical target
                le = LabelEncoder()
                encoded_series = series.copy()
                non_missing_mask = ~encoded_series.isnull()
                if non_missing_mask.sum() > 0:
                    encoded_series[non_missing_mask] = le.fit_transform(encoded_series[non_missing_mask])
                    series = encoded_series
                    numeric_cols.append(column)
            
            if len(numeric_cols) < 2:
                raise ValueError("Insufficient numeric columns for KNN")
            
            # Prepare data for KNN
            X = df[numeric_cols].copy()
            
            # Use KNNImputer
            imputer = KNNImputer(n_neighbors=self.config['knn_neighbors'])
            X_imputed = imputer.fit_transform(X)
            
            # Extract imputed values for target column
            col_index = numeric_cols.index(column)
            imputed_column = X_imputed[:, col_index]
            
            # Decode if originally categorical
            if column not in df.select_dtypes(include=[np.number]).columns:
                # Round to nearest integer for categorical
                imputed_column = np.round(imputed_column).astype(int)
                # Map back to original categories
                try:
                    imputed_column = le.inverse_transform(imputed_column)
                except:
                    pass  # Keep as numbers if inverse transform fails
            
            # Update series
            missing_mask = series.isnull()
            series.loc[missing_mask] = imputed_column[missing_mask]
            
            return 0.75  # Good quality score for KNN
            
        except Exception as e:
            logging.warning(f"KNN imputation failed for {column}: {e}")
            raise e
    
    def _predictive_imputation(self, df: pd.DataFrame, column: str, series: pd.Series,
                              analysis: MissingAnalysis) -> float:
        """Perform predictive imputation using machine learning."""
        try:
            # Prepare features (other columns)
            feature_cols = [col for col in df.columns if col != column]
            
            if not feature_cols:
                raise ValueError("No features available for prediction")
            
            # Select numeric and low-cardinality categorical features
            X_features = []
            feature_names = []
            
            for col in feature_cols[:20]:  # Limit features for efficiency
                if pd.api.types.is_numeric_dtype(df[col]):
                    X_features.append(df[col].fillna(df[col].median()).values)
                    feature_names.append(col)
                elif df[col].nunique() <= 20:  # Low cardinality categorical
                    le = LabelEncoder()
                    col_encoded = df[col].fillna('missing')
                    encoded_values = le.fit_transform(col_encoded)
                    X_features.append(encoded_values)
                    feature_names.append(col)
            
            if not X_features:
                raise ValueError("No suitable features found")
            
            X = np.column_stack(X_features)
            
            # Prepare target
            y = series.copy()
            if not pd.api.types.is_numeric_dtype(y):
                le_target = LabelEncoder()
                y_encoded = y.dropna()
                if len(y_encoded) == 0:
                    raise ValueError("No non-missing target values")
                le_target.fit(y_encoded)
                y = y.map(lambda x: le_target.transform([x])[0] if pd.notna(x) else np.nan)
            
            # Split into train/predict
            train_mask = ~y.isnull()
            predict_mask = y.isnull()
            
            if train_mask.sum() < 10:
                raise ValueError("Insufficient training samples")
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_predict = X[predict_mask]
            
            # Train model
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict missing values
            predictions = model.predict(X_predict)
            
            # Decode if categorical
            if not pd.api.types.is_numeric_dtype(series):
                predictions = le_target.inverse_transform(predictions)
            
            # Update series
            series_copy = series.copy()
            series_copy.loc[predict_mask] = predictions
            
            # Calculate quality based on model score
            quality_score = min(0.9, model.score(X_train, y_train))
            
            # Update original series
            series.loc[predict_mask] = predictions
            
            return quality_score
            
        except Exception as e:
            logging.warning(f"Predictive imputation failed for {column}: {e}")
            raise e
    
    def _domain_specific_imputation(self, df: pd.DataFrame, column: str, series: pd.Series,
                                   analysis: MissingAnalysis) -> Tuple[pd.Series, float]:
        """Apply domain-specific imputation rules."""
        column_lower = column.lower()
        
        # Age-specific rules
        if 'age' in column_lower:
            median_age = series.median()
            if pd.isna(median_age):
                median_age = 35  # Default reasonable age
            series = series.fillna(median_age)
            return series, 0.7
        
        # Count/quantity rules
        elif any(word in column_lower for word in ['count', 'quantity', 'number']):
            series = series.fillna(0)  # Assume zero for missing counts
            return series, 0.8
        
        # Price/monetary rules
        elif any(word in column_lower for word in ['price', 'cost', 'amount', 'salary']):
            median_value = series.median()
            if pd.isna(median_value):
                median_value = 0
            series = series.fillna(median_value)
            return series, 0.6
        
        # Percentage rules
        elif any(word in column_lower for word in ['percent', 'pct', 'rate']):
            mean_pct = series.mean()
            if pd.isna(mean_pct):
                mean_pct = 50  # Default to 50% if no data
            series = series.fillna(mean_pct)
            return series, 0.6
        
        # Default fallback
        else:
            if pd.api.types.is_numeric_dtype(series):
                series = series.fillna(series.median())
            else:
                mode_val = series.mode()
                if not mode_val.empty:
                    series = series.fillna(mode_val.iloc[0])
            return series, 0.5
    
    def _get_fallback_strategy(self, series: pd.Series) -> ImputationStrategy:
        """Get fallback imputation strategy."""
        if pd.api.types.is_numeric_dtype(series):
            return ImputationStrategy.MEDIAN
        else:
            return ImputationStrategy.MODE
    
    def _cross_validate_imputation(self, df: pd.DataFrame, column: str, 
                                  strategy: ImputationStrategy) -> float:
        """Cross-validate imputation strategy quality."""
        try:
            # This is a simplified CV - in practice you'd do proper k-fold
            series = df[column]
            non_missing_mask = ~series.isnull()
            
            if non_missing_mask.sum() < 20:  # Need sufficient data
                return 0.5
            
            # Create artificial missing values for testing
            test_indices = np.random.choice(
                series[non_missing_mask].index, 
                size=min(10, non_missing_mask.sum() // 4),
                replace=False
            )
            
            # Create test scenario
            df_test = df.copy()
            original_values = df_test.loc[test_indices, column].copy()
            df_test.loc[test_indices, column] = np.nan
            
            # Apply imputation
            analysis = self.missing_analyses.get(column)
            if analysis is None:
                return 0.5
            
            imputed_series, _ = self._apply_imputation_strategy(df_test, column, strategy, analysis)
            predicted_values = imputed_series.loc[test_indices]
            
            # Calculate accuracy
            if pd.api.types.is_numeric_dtype(series):
                # RMSE for numeric
                mse = np.mean((original_values - predicted_values) ** 2)
                rmse = np.sqrt(mse)
                # Normalize by standard deviation
                std = original_values.std()
                if std > 0:
                    normalized_rmse = rmse / std
                    accuracy = max(0, 1 - normalized_rmse)
                else:
                    accuracy = 0.5
            else:
                # Accuracy for categorical
                matches = (original_values == predicted_values).sum()
                accuracy = matches / len(original_values)
            
            return min(1.0, accuracy)
            
        except Exception as e:
            logging.warning(f"Cross-validation failed for {column}: {e}")
            return 0.5
    
    def _calculate_column_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate basic statistics for a column."""
        stats_dict = {
            'count': len(series),
            'missing_count': series.isnull().sum(),
            'missing_percentage': (series.isnull().sum() / len(series)) * 100
        }
        
        if pd.api.types.is_numeric_dtype(series):
            clean_series = series.dropna()
            if len(clean_series) > 0:
                stats_dict.update({
                    'mean': clean_series.mean(),
                    'median': clean_series.median(),
                    'std': clean_series.std(),
                    'min': clean_series.min(),
                    'max': clean_series.max()
                })
        else:
            clean_series = series.dropna()
            if len(clean_series) > 0:
                stats_dict.update({
                    'unique_count': clean_series.nunique(),
                    'most_common': clean_series.value_counts().index[0] if not clean_series.empty else None
                })
        
        return stats_dict
    
    def _collect_column_metadata(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Collect metadata about the column."""
        series = df[column]
        metadata = {
            'dtype': str(series.dtype),
            'total_count': len(series),
            'unique_count': series.nunique(),
            'first_valid_index': series.first_valid_index(),
            'last_valid_index': series.last_valid_index()
        }
        
        if pd.api.types.is_numeric_dtype(series):
            metadata['is_numeric'] = True
            clean_series = series.dropna()
            if len(clean_series) > 0:
                metadata.update({
                    'has_negative': (clean_series < 0).any(),
                    'has_zero': (clean_series == 0).any(),
                    'range': clean_series.max() - clean_series.min() if clean_series.max() != clean_series.min() else 0
                })
        else:
            metadata['is_numeric'] = False
            metadata['sample_values'] = series.dropna().head(5).tolist()
        
        return metadata
    
    def _point_biserial_correlation(self, binary_var: pd.Series, continuous_var: pd.Series) -> float:
        """Calculate point-biserial correlation."""
        try:
            # Remove missing values
            mask = ~(binary_var.isnull() | continuous_var.isnull())
            x = binary_var[mask]
            y = continuous_var[mask]
            
            if len(x) < 10:
                return np.nan
            
            # Calculate point-biserial correlation
            n1 = (x == 1).sum()
            n0 = (x == 0).sum()
            
            if n1 == 0 or n0 == 0:
                return np.nan
            
            m1 = y[x == 1].mean()
            m0 = y[x == 0].mean()
            s = y.std()
            
            if s == 0:
                return np.nan
            
            rpb = ((m1 - m0) / s) * np.sqrt((n1 * n0) / ((n1 + n0) ** 2))
            return rpb
            
        except:
            return np.nan
    
    def get_imputation_summary(self) -> Dict[str, Any]:
        """Get summary of imputation results."""
        if not self.imputation_history:
            return {'message': 'No imputation history available'}
        
        total_imputed = sum(result.values_imputed for result in self.imputation_history)
        avg_quality = np.mean([result.imputation_quality for result in self.imputation_history])
        
        strategy_counts = {}
        for result in self.imputation_history:
            strategy = result.strategy_used.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            'total_columns_imputed': len(self.imputation_history),
            'total_values_imputed': total_imputed,
            'average_quality_score': round(avg_quality, 3),
            'strategies_used': strategy_counts,
            'high_quality_imputations': len([r for r in self.imputation_history if r.imputation_quality > 0.7]),
            'low_quality_imputations': len([r for r in self.imputation_history if r.imputation_quality < 0.5])
        }
    
    def get_missing_value_report(self) -> Dict[str, Any]:
        """Generate comprehensive missing value report."""
        if not self.missing_analyses:
            return {'message': 'No missing value analysis available'}
        
        pattern_counts = {}
        for analysis in self.missing_analyses.values():
            pattern = analysis.pattern_type.value
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        total_missing = sum(analysis.missing_count for analysis in self.missing_analyses.values())
        high_confidence_patterns = len([a for a in self.missing_analyses.values() if a.pattern_confidence > 0.7])
        
        return {
            'columns_with_missing': len(self.missing_analyses),
            'total_missing_values': total_missing,
            'missing_patterns': pattern_counts,
            'high_confidence_patterns': high_confidence_patterns,
            'recommended_strategies': {
                analysis.column: analysis.recommended_strategy.value 
                for analysis in self.missing_analyses.values()
            },
            'correlations_found': sum(len(a.correlations) for a in self.missing_analyses.values()),
            'avg_imputation_confidence': round(
                np.mean([a.imputation_confidence for a in self.missing_analyses.values()]), 3
            )
        }

def analyze_and_impute_missing_values(df: pd.DataFrame, 
                                    strategies: Optional[Dict[str, ImputationStrategy]] = None,
                                    config: Optional[Dict[str, Any]] = None,
                                    validate_quality: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function for complete missing value analysis and imputation.
    
    Args:
        df: DataFrame to process
        strategies: Optional custom imputation strategies
        config: Optional configuration
        validate_quality: Whether to validate imputation quality
        
    Returns:
        Tuple of (imputed_dataframe, comprehensive_report)
    """
    intelligence = AdvancedMissingValueIntelligence(config)
    
    # Analyze missing patterns
    missing_analyses = intelligence.analyze_missing_patterns(df)
    
    # Perform intelligent imputation
    df_imputed, imputation_results = intelligence.intelligent_imputation(
        df, strategies, validate_quality
    )
    
    # Generate reports
    missing_report = intelligence.get_missing_value_report()
    imputation_summary = intelligence.get_imputation_summary()
    
    # Combine into comprehensive report
    comprehensive_report = {
        'missing_analysis': missing_report,
        'imputation_summary': imputation_summary,
        'detailed_analyses': {col: {
            'pattern_type': analysis.pattern_type.value,
            'pattern_confidence': analysis.pattern_confidence,
            'missing_percentage': analysis.missing_percentage,
            'recommended_strategy': analysis.recommended_strategy.value,
            'correlations': analysis.correlations
        } for col, analysis in missing_analyses.items()},
        'imputation_results': {col: {
            'strategy_used': result.strategy_used.value,
            'values_imputed': result.values_imputed,
            'quality_score': result.imputation_quality,
            'cross_validation_score': result.cross_validation_score
        } for col, result in imputation_results.items()}
    }
    
    return df_imputed, comprehensive_report