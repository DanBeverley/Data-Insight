import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Set, Optional
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.metrics import mutual_info_score
from dataclasses import dataclass
import itertools

@dataclass
class Relationship:
    column1: str
    column2: str
    relationship_type: str
    strength: float
    confidence: float
    description: str
    statistical_test: Optional[str] = None
    p_value: Optional[float] = None

class RelationshipDiscovery:
    def __init__(self, significance_threshold: float = 0.05, min_strength: float = 0.3):
        self.significance_threshold = significance_threshold
        self.min_strength = min_strength
        self.relationship_detectors = {
            'primary_foreign_key': self._detect_key_relationships,
            'numerical_correlation': self._detect_numerical_correlation,
            'categorical_association': self._detect_categorical_association,
            'numerical_categorical': self._detect_numerical_categorical,
            'temporal_dependency': self._detect_temporal_dependency,
            'functional_dependency': self._detect_functional_dependency,
            'hierarchical': self._detect_hierarchical_relationship
        }
    
    def discover_relationships(self, df: pd.DataFrame, 
                             column_profiles: Optional[Dict] = None) -> List[Relationship]:
        """Discover all types of relationships between columns"""
        relationships = []
        
        for detector_name, detector_func in self.relationship_detectors.items():
            try:
                detected = detector_func(df, column_profiles)
                relationships.extend(detected)
            except Exception as e:
                continue
        
        # Remove duplicates and sort by strength
        unique_relationships = self._deduplicate_relationships(relationships)
        return sorted(unique_relationships, key=lambda x: x.strength, reverse=True)
    
    def _detect_key_relationships(self, df: pd.DataFrame, 
                                column_profiles: Optional[Dict] = None) -> List[Relationship]:
        """Detect primary key and foreign key relationships"""
        relationships = []
        
        # Identify potential primary keys (unique values, high cardinality)
        primary_keys = []
        for col in df.columns:
            if df[col].nunique() == len(df) and df[col].notna().all():
                primary_keys.append(col)
        
        # Identify potential foreign keys
        for col in df.columns:
            if col not in primary_keys:
                # Check if column values are subset of any primary key
                for pk in primary_keys:
                    pk_values = set(df[pk].dropna())
                    col_values = set(df[col].dropna())
                    
                    if col_values.issubset(pk_values) and len(col_values) > 1:
                        overlap_ratio = len(col_values) / len(pk_values)
                        
                        relationships.append(Relationship(
                            column1=pk,
                            column2=col,
                            relationship_type='primary_foreign_key',
                            strength=overlap_ratio,
                            confidence=0.9 if overlap_ratio > 0.5 else 0.7,
                            description=f'{pk} appears to be primary key, {col} is foreign key reference'
                        ))
        
        return relationships
    
    def _detect_numerical_correlation(self, df: pd.DataFrame, 
                                    column_profiles: Optional[Dict] = None) -> List[Relationship]:
        """Detect correlations between numerical columns"""
        relationships = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return relationships
        
        for col1, col2 in itertools.combinations(numeric_cols, 2):
            # Clean data
            mask = df[col1].notna() & df[col2].notna()
            if mask.sum() < 10:  # Need minimum samples
                continue
            
            x, y = df.loc[mask, col1], df.loc[mask, col2]
            
            # Pearson correlation
            try:
                pearson_r, pearson_p = pearsonr(x, y)
                if abs(pearson_r) >= self.min_strength and pearson_p < self.significance_threshold:
                    relationships.append(Relationship(
                        column1=col1,
                        column2=col2,
                        relationship_type='linear_correlation',
                        strength=abs(pearson_r),
                        confidence=1 - pearson_p,
                        description=f'{"Strong" if abs(pearson_r) > 0.7 else "Moderate"} linear correlation',
                        statistical_test='pearson',
                        p_value=pearson_p
                    ))
            except:
                pass
            
            # Spearman correlation (monotonic relationships)
            try:
                spearman_r, spearman_p = spearmanr(x, y)
                if abs(spearman_r) >= self.min_strength and spearman_p < self.significance_threshold:
                    if abs(spearman_r) - abs(pearson_r) > 0.2:  # Non-linear monotonic
                        relationships.append(Relationship(
                            column1=col1,
                            column2=col2,
                            relationship_type='monotonic_correlation',
                            strength=abs(spearman_r),
                            confidence=1 - spearman_p,
                            description='Non-linear monotonic relationship',
                            statistical_test='spearman',
                            p_value=spearman_p
                        ))
            except:
                pass
        
        return relationships
    
    def _detect_categorical_association(self, df: pd.DataFrame, 
                                      column_profiles: Optional[Dict] = None) -> List[Relationship]:
        """Detect associations between categorical columns"""
        relationships = []
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(categorical_cols) < 2:
            return relationships
        
        for col1, col2 in itertools.combinations(categorical_cols, 2):
            # Create contingency table
            try:
                contingency = pd.crosstab(df[col1], df[col2])
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                
                # Chi-square test
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                
                # Cramér's V (measure of association strength)
                n = contingency.sum().sum()
                cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
                
                if cramers_v >= self.min_strength and p_value < self.significance_threshold:
                    relationships.append(Relationship(
                        column1=col1,
                        column2=col2,
                        relationship_type='categorical_association',
                        strength=cramers_v,
                        confidence=1 - p_value,
                        description=f'{"Strong" if cramers_v > 0.6 else "Moderate"} categorical association',
                        statistical_test='chi_square',
                        p_value=p_value
                    ))
            except:
                continue
        
        return relationships
    
    def _detect_numerical_categorical(self, df: pd.DataFrame, 
                                    column_profiles: Optional[Dict] = None) -> List[Relationship]:
        """Detect relationships between numerical and categorical columns"""
        relationships = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                try:
                    # Group numerical values by categories
                    groups = [group[num_col].dropna() for name, group in df.groupby(cat_col)]
                    groups = [g for g in groups if len(g) > 2]  # Filter small groups
                    
                    if len(groups) < 2:
                        continue
                    
                    # ANOVA F-test
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    if p_value < self.significance_threshold:
                        # Calculate effect size (eta-squared)
                        ss_between = sum(len(g) * (g.mean() - df[num_col].mean())**2 for g in groups)
                        ss_total = ((df[num_col] - df[num_col].mean())**2).sum()
                        eta_squared = ss_between / ss_total if ss_total > 0 else 0
                        
                        if eta_squared >= self.min_strength:
                            relationships.append(Relationship(
                                column1=cat_col,
                                column2=num_col,
                                relationship_type='categorical_numerical',
                                strength=eta_squared,
                                confidence=1 - p_value,
                                description=f'{cat_col} significantly affects {num_col} distribution',
                                statistical_test='anova',
                                p_value=p_value
                            ))
                except:
                    continue
        
        return relationships
    
    def _detect_temporal_dependency(self, df: pd.DataFrame, 
                                  column_profiles: Optional[Dict] = None) -> List[Relationship]:
        """Detect temporal dependencies and trends"""
        relationships = []
        
        # Identify datetime columns
        datetime_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            elif df[col].dtype == 'object':
                try:
                    sample = df[col].dropna().iloc[:100]
                    if len(sample) > 0:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            parsed = pd.to_datetime(sample, errors='coerce')
                            valid_ratio = parsed.notna().sum() / len(sample)
                            if valid_ratio > 0.7:
                                datetime_cols.append(col)
                except:
                    continue
        
        if not datetime_cols:
            return relationships
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for dt_col in datetime_cols:
            for num_col in numeric_cols:
                try:
                    # Convert to datetime if needed
                    if not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            dt_series = pd.to_datetime(df[dt_col], errors='coerce')
                    else:
                        dt_series = df[dt_col]
                    
                    # Create time-ordered dataset
                    temp_df = pd.DataFrame({
                        'time': dt_series,
                        'value': df[num_col]
                    }).dropna().sort_values('time')
                    
                    if len(temp_df) < 10:
                        continue
                    
                    # Calculate trend strength using linear regression
                    x = np.arange(len(temp_df))
                    y = temp_df['value'].values
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    if abs(r_value) >= self.min_strength and p_value < self.significance_threshold:
                        trend_direction = "increasing" if slope > 0 else "decreasing"
                        relationships.append(Relationship(
                            column1=dt_col,
                            column2=num_col,
                            relationship_type='temporal_trend',
                            strength=abs(r_value),
                            confidence=1 - p_value,
                            description=f'{num_col} shows {trend_direction} trend over {dt_col}',
                            statistical_test='linear_regression',
                            p_value=p_value
                        ))
                except:
                    continue
        
        return relationships
    
    def _detect_functional_dependency(self, df: pd.DataFrame, 
                                    column_profiles: Optional[Dict] = None) -> List[Relationship]:
        """Detect functional dependencies (determinant -> dependent)"""
        relationships = []
        
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 == col2:
                    continue
                
                # Check if col1 functionally determines col2
                # (each value of col1 maps to exactly one value of col2)
                try:
                    grouped = df.groupby(col1)[col2].nunique()
                    if (grouped == 1).all() and len(grouped) > 1:
                        # Calculate dependency strength
                        strength = 1.0 - (grouped.var() / grouped.mean() if grouped.mean() > 0 else 0)
                        
                        relationships.append(Relationship(
                            column1=col1,
                            column2=col2,
                            relationship_type='functional_dependency',
                            strength=min(strength, 1.0),
                            confidence=0.95,
                            description=f'{col1} functionally determines {col2}',
                        ))
                except:
                    continue
        
        return relationships
    
    def _detect_hierarchical_relationship(self, df: pd.DataFrame, 
                                        column_profiles: Optional[Dict] = None) -> List[Relationship]:
        """Detect hierarchical relationships (parent-child, part-whole)"""
        relationships = []
        
        # Look for naming patterns that suggest hierarchy
        hierarchy_patterns = [
            ('category', 'subcategory'),
            ('parent', 'child'),
            ('group', 'subgroup'),
            ('country', 'state'),
            ('state', 'city'),
            ('department', 'team'),
            ('brand', 'product')
        ]
        
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 == col2:
                    continue
                
                col1_lower = col1.lower()
                col2_lower = col2.lower()
                
                # Check naming patterns
                hierarchy_match = False
                for parent_pattern, child_pattern in hierarchy_patterns:
                    if parent_pattern in col1_lower and child_pattern in col2_lower:
                        hierarchy_match = True
                        break
                
                if hierarchy_match:
                    try:
                        # Verify hierarchical structure
                        parent_child_map = df.groupby(col1)[col2].nunique()
                        child_parent_map = df.groupby(col2)[col1].nunique()
                        
                        # Parent should have multiple children, child should have one parent
                        avg_children = parent_child_map.mean()
                        max_parents = child_parent_map.max()
                        
                        if avg_children > 1 and max_parents == 1:
                            strength = min(avg_children / df[col2].nunique(), 1.0)
                            
                            relationships.append(Relationship(
                                column1=col1,
                                column2=col2,
                                relationship_type='hierarchical',
                                strength=strength,
                                confidence=0.8,
                                description=f'{col1} is parent level of {col2}',
                            ))
                    except:
                        continue
        
        return relationships
    
    def _deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Remove duplicate relationships keeping the strongest ones"""
        seen_pairs = set()
        unique_relationships = []
        
        for rel in sorted(relationships, key=lambda x: x.strength, reverse=True):
            pair = tuple(sorted([rel.column1, rel.column2]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_relationships.append(rel)
        
        return unique_relationships
    
    def generate_relationship_graph(self, relationships: List[Relationship]) -> Dict:
        """Generate a graph representation of relationships"""
        nodes = set()
        edges = []
        
        for rel in relationships:
            nodes.add(rel.column1)
            nodes.add(rel.column2)
            edges.append({
                'source': rel.column1,
                'target': rel.column2,
                'type': rel.relationship_type,
                'strength': rel.strength,
                'description': rel.description
            })
        
        return {
            'nodes': [{'id': node} for node in nodes],
            'edges': edges
        }
    
    def get_relationship_recommendations(self, relationships: List[Relationship]) -> List[str]:
        """Generate recommendations based on discovered relationships"""
        recommendations = []
        
        relationship_types = [rel.relationship_type for rel in relationships]
        type_counts = pd.Series(relationship_types).value_counts()
        
        if 'primary_foreign_key' in type_counts:
            recommendations.append("Create aggregation features based on key relationships")
        
        if 'linear_correlation' in type_counts or 'monotonic_correlation' in type_counts:
            recommendations.append("Consider dimensionality reduction techniques (PCA, Factor Analysis)")
        
        if 'categorical_association' in type_counts:
            recommendations.append("Create interaction features between associated categorical variables")
        
        if 'temporal_trend' in type_counts:
            recommendations.append("Add lag features and trend decomposition for temporal patterns")
        
        if 'functional_dependency' in type_counts:
            recommendations.append("Consider removing redundant dependent variables")
        
        if 'hierarchical' in type_counts:
            recommendations.append("Create roll-up aggregation features across hierarchy levels")
        
        return recommendations