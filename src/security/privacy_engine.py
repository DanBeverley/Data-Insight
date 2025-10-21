import numpy as np
import pandas as pd
import hashlib
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class PrivacyTechnique(Enum):
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    K_ANONYMITY = "k_anonymity"
    L_DIVERSITY = "l_diversity"
    T_CLOSENESS = "t_closeness"
    SYNTHETIC_DATA = "synthetic_data"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"

class PrivacyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

@dataclass
class PrivacyConfiguration:
    target_privacy_level: PrivacyLevel
    epsilon: float = 1.0
    delta: float = 1e-5
    k_value: int = 5
    l_value: int = 2
    t_threshold: float = 0.2
    enable_synthetic_data: bool = False
    preserve_utility: bool = True

@dataclass
class PrivacyAssessment:
    privacy_score: float
    risk_level: str
    techniques_applied: List[PrivacyTechnique]
    utility_preservation: float
    reidentification_risk: float
    recommendations: List[str]
    detected_columns: Dict[str, float] = None
    detected_types: List[str] = None

class PrivacyEngine:
    def __init__(self, config: Optional[PrivacyConfiguration] = None):
        self.config = config or PrivacyConfiguration(PrivacyLevel.MEDIUM)
        self.privacy_transformations = {}
        self.utility_metrics = {}
        
    def assess_privacy_risk(self, df: pd.DataFrame,
                           quasi_identifiers: List[str] = None,
                           sensitive_attributes: List[str] = None) -> PrivacyAssessment:

        quasi_identifiers = quasi_identifiers or self._auto_detect_quasi_identifiers(df)
        sensitive_attributes = sensitive_attributes or self._auto_detect_sensitive_attributes(df)

        reidentification_risk = self._calculate_reidentification_risk(df, quasi_identifiers)
        privacy_score = self._calculate_privacy_score(df, quasi_identifiers, sensitive_attributes)
        risk_level = self._determine_risk_level(reidentification_risk, privacy_score)

        recommendations = self._generate_privacy_recommendations(
            reidentification_risk, privacy_score, quasi_identifiers, sensitive_attributes
        )

        detected_columns = self._calculate_column_sensitivity(df, quasi_identifiers, sensitive_attributes)
        detected_types = list(set([self._categorize_column_type(col) for col in detected_columns.keys()]))

        return PrivacyAssessment(
            privacy_score=privacy_score,
            risk_level=risk_level,
            techniques_applied=[],
            utility_preservation=1.0,
            reidentification_risk=reidentification_risk,
            recommendations=recommendations,
            detected_columns=detected_columns,
            detected_types=detected_types
        )
    
    def _auto_detect_quasi_identifiers(self, df: pd.DataFrame) -> List[str]:
        quasi_identifiers = []
        
        identifier_patterns = ['id', 'name', 'address', 'zip', 'postal', 'birth', 'age', 'gender']
        
        for column in df.columns:
            column_lower = column.lower()
            
            if any(pattern in column_lower for pattern in identifier_patterns):
                quasi_identifiers.append(column)
                continue
            
            if df[column].dtype == 'object':
                unique_ratio = df[column].nunique() / len(df)
                if 0.1 < unique_ratio < 0.9:
                    quasi_identifiers.append(column)
            elif df[column].dtype in ['int64', 'float64']:
                if 'year' in column_lower or 'age' in column_lower:
                    quasi_identifiers.append(column)
        
        return quasi_identifiers
    
    def _auto_detect_sensitive_attributes(self, df: pd.DataFrame) -> List[str]:
        sensitive_attributes = []
        
        sensitive_patterns = ['salary', 'income', 'medical', 'health', 'diagnosis', 'treatment', 
                            'credit', 'score', 'rating', 'political', 'religion', 'race', 'ethnic']
        
        for column in df.columns:
            column_lower = column.lower()
            if any(pattern in column_lower for pattern in sensitive_patterns):
                sensitive_attributes.append(column)
        
        return sensitive_attributes
    
    def _calculate_reidentification_risk(self, df: pd.DataFrame, quasi_identifiers: List[str]) -> float:
        if not quasi_identifiers:
            return 0.1
        
        available_qi = [col for col in quasi_identifiers if col in df.columns]
        if not available_qi:
            return 0.1
        
        try:
            qi_combinations = df[available_qi].drop_duplicates()
            unique_combinations = len(qi_combinations)
            total_records = len(df)
            
            if total_records == 0:
                return 0.1
            
            uniqueness_ratio = unique_combinations / total_records
            
            equivalence_class_sizes = df.groupby(available_qi).size()
            avg_class_size = equivalence_class_sizes.mean()
            min_class_size = equivalence_class_sizes.min()
            
            size_based_risk = 1 / avg_class_size if avg_class_size > 0 else 1.0
            uniqueness_risk = uniqueness_ratio
            minimal_protection = 1 / min_class_size if min_class_size > 0 else 1.0
            
            overall_risk = (size_based_risk * 0.4 + uniqueness_risk * 0.4 + minimal_protection * 0.2)
            
            return min(1.0, max(0.0, overall_risk))
            
        except Exception:
            return 0.5
    
    def _calculate_privacy_score(self, df: pd.DataFrame, quasi_identifiers: List[str], 
                               sensitive_attributes: List[str]) -> float:
        
        score_components = []
        
        if quasi_identifiers:
            qi_protection = 1 - self._calculate_reidentification_risk(df, quasi_identifiers)
            score_components.append(qi_protection * 0.4)
        
        if sensitive_attributes:
            sensitive_diversity = self._calculate_sensitive_diversity(df, sensitive_attributes)
            score_components.append(sensitive_diversity * 0.3)
        
        data_sparsity = self._calculate_data_sparsity(df)
        score_components.append(data_sparsity * 0.2)
        
        generalization_level = self._assess_generalization_level(df)
        score_components.append(generalization_level * 0.1)
        
        return sum(score_components) if score_components else 0.5
    
    def _calculate_sensitive_diversity(self, df: pd.DataFrame, sensitive_attributes: List[str]) -> float:
        diversity_scores = []
        
        for attr in sensitive_attributes:
            if attr in df.columns:
                value_counts = df[attr].value_counts()
                if len(value_counts) > 1:
                    total_count = len(df)
                    entropy = 0
                    for count in value_counts:
                        p = count / total_count
                        if p > 0:
                            entropy -= p * np.log2(p)
                    max_entropy = np.log2(len(value_counts))
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    diversity_scores.append(normalized_entropy)
        
        return np.mean(diversity_scores) if diversity_scores else 0.5
    
    def _calculate_data_sparsity(self, df: pd.DataFrame) -> float:
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        return min(1.0, missing_ratio * 2)
    
    def _assess_generalization_level(self, df: pd.DataFrame) -> float:
        generalization_indicators = 0
        total_columns = len(df.columns)
        
        for column in df.columns:
            if df[column].dtype == 'object':
                if any(indicator in str(df[column].iloc[0]).lower() for indicator in ['*', 'other', 'unknown', 'range']):
                    generalization_indicators += 1
        
        return generalization_indicators / total_columns if total_columns > 0 else 0
    
    def _determine_risk_level(self, reidentification_risk: float, privacy_score: float) -> str:
        combined_risk = (reidentification_risk * 0.7 + (1 - privacy_score) * 0.3)
        
        if combined_risk < 0.2:
            return "low"
        elif combined_risk < 0.5:
            return "medium"
        elif combined_risk < 0.8:
            return "high"
        else:
            return "critical"
    
    def _generate_privacy_recommendations(self, reidentification_risk: float, privacy_score: float,
                                        quasi_identifiers: List[str], sensitive_attributes: List[str]) -> List[str]:
        recommendations = []
        
        if reidentification_risk > 0.7:
            recommendations.append("Apply k-anonymity with k >= 5 to reduce reidentification risk")
            
        if reidentification_risk > 0.5:
            recommendations.append("Consider generalization or suppression of quasi-identifiers")
        
        if sensitive_attributes and privacy_score < 0.6:
            recommendations.append("Apply l-diversity to protect sensitive attributes")
        
        if len(quasi_identifiers) > 5:
            recommendations.append("Reduce number of quasi-identifiers through feature selection")
        
        if privacy_score < 0.5:
            recommendations.append("Consider differential privacy for strong theoretical guarantees")

        return recommendations

    def _calculate_column_sensitivity(self, df: pd.DataFrame,
                                     quasi_identifiers: List[str],
                                     sensitive_attributes: List[str]) -> Dict[str, float]:
        column_sensitivity = {}

        for col in quasi_identifiers:
            if col not in df.columns:
                continue

            uniqueness = df[col].nunique() / len(df) if len(df) > 0 else 0
            entropy = self._calculate_entropy(df[col])
            cardinality_factor = min(1.0, df[col].nunique() / 1000)

            base_sensitivity = 0.3
            uniqueness_contribution = uniqueness * 0.4
            entropy_contribution = (entropy / 10) * 0.2
            cardinality_contribution = cardinality_factor * 0.1

            sensitivity = base_sensitivity + uniqueness_contribution + entropy_contribution + cardinality_contribution
            column_sensitivity[col] = round(min(0.95, sensitivity), 2)

        for col in sensitive_attributes:
            if col not in df.columns:
                continue

            uniqueness = df[col].nunique() / len(df) if len(df) > 0 else 0
            entropy = self._calculate_entropy(df[col])

            base_sensitivity = 0.7
            uniqueness_penalty = (1 - uniqueness) * 0.15
            entropy_contribution = (entropy / 10) * 0.1

            sensitivity = base_sensitivity + uniqueness_penalty + entropy_contribution
            column_sensitivity[col] = round(min(0.99, sensitivity), 2)

        return column_sensitivity

    def _calculate_entropy(self, series: pd.Series) -> float:
        value_counts = series.value_counts(normalize=True)
        entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
        return entropy

    def _categorize_column_type(self, column: str) -> str:
        return 'Sensitive Data'

    def apply_k_anonymity(self, df: pd.DataFrame, quasi_identifiers: List[str], 
                         k: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        
        k = k or self.config.k_value
        anonymized_df = df.copy()
        transformations = {}
        
        for qi in quasi_identifiers:
            if qi not in df.columns:
                continue
                
            if df[qi].dtype in ['int64', 'float64']:
                anonymized_df[qi], transformation = self._generalize_numeric(df[qi], k)
            else:
                anonymized_df[qi], transformation = self._generalize_categorical(df[qi], k)
            
            transformations[qi] = transformation
        
        utility_score = self._calculate_utility_preservation(df, anonymized_df, quasi_identifiers)
        
        metadata = {
            "technique": PrivacyTechnique.K_ANONYMITY,
            "k_value": k,
            "transformations": transformations,
            "utility_preservation": utility_score
        }
        
        return anonymized_df, metadata
    
    def _generalize_numeric(self, series: pd.Series, k: int) -> Tuple[pd.Series, Dict[str, Any]]:
        try:
            unique_counts = series.value_counts()
            ranges_created = []
            
            if len(unique_counts) < k:
                min_val, max_val = series.min(), series.max()
                generalized = pd.Series([f"{min_val}-{max_val}"] * len(series), index=series.index)
                return generalized, {"type": "full_range", "range": f"{min_val}-{max_val}"}
            
            sorted_values = series.sort_values()
            generalized = series.copy()
            
            for i in range(0, len(sorted_values), k):
                chunk = sorted_values.iloc[i:i+k]
                if len(chunk) > 0:
                    range_str = f"{chunk.min()}-{chunk.max()}"
                    generalized.loc[chunk.index] = range_str
                    ranges_created.append(range_str)
            
            return generalized, {"type": "range_binning", "ranges": ranges_created}
            
        except Exception:
            return series, {"type": "unchanged", "reason": "generalization_failed"}
    
    def _generalize_categorical(self, series: pd.Series, k: int) -> Tuple[pd.Series, Dict[str, Any]]:
        try:
            value_counts = series.value_counts()
            rare_values = value_counts[value_counts < k].index.tolist()
            
            if not rare_values:
                return series, {"type": "unchanged", "reason": "no_rare_values"}
            
            generalized = series.copy()
            generalized[series.isin(rare_values)] = "Other"
            
            return generalized, {"type": "suppression", "suppressed_values": rare_values}
            
        except Exception:
            return series, {"type": "unchanged", "reason": "generalization_failed"}
    
    def apply_differential_privacy(self, df: pd.DataFrame, epsilon: Optional[float] = None,
                                 numeric_columns: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        
        epsilon = epsilon or self.config.epsilon
        private_df = df.copy()
        
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        noise_added = {}
        
        for column in numeric_columns:
            if column in df.columns:
                sensitivity = df[column].std()
                noise_scale = sensitivity / epsilon
                noise = np.random.laplace(0, noise_scale, len(df))
                
                private_df[column] = df[column] + noise
                noise_added[column] = {"scale": noise_scale, "sensitivity": sensitivity}
        
        utility_score = self._calculate_utility_preservation(df, private_df, numeric_columns)
        
        metadata = {
            "technique": PrivacyTechnique.DIFFERENTIAL_PRIVACY,
            "epsilon": epsilon,
            "noise_parameters": noise_added,
            "utility_preservation": utility_score
        }
        
        return private_df, metadata
    
    def apply_l_diversity(self, df: pd.DataFrame, quasi_identifiers: List[str],
                         sensitive_attribute: str, l: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        
        l = l or self.config.l_value
        
        if sensitive_attribute not in df.columns:
            return df, {"error": "Sensitive attribute not found"}
        
        grouped = df.groupby(quasi_identifiers)
        diverse_groups = []
        
        for name, group in grouped:
            sensitive_values = group[sensitive_attribute].nunique()
            
            if sensitive_values >= l:
                diverse_groups.append(group)
            else:
                needed_diversity = l - sensitive_values
                group_suppressed = group.copy()
                group_suppressed[sensitive_attribute] = "Suppressed"
                diverse_groups.append(group_suppressed)
        
        if diverse_groups:
            diversified_df = pd.concat(diverse_groups, ignore_index=True)
        else:
            diversified_df = df.copy()
        
        utility_score = self._calculate_utility_preservation(df, diversified_df, [sensitive_attribute])
        
        metadata = {
            "technique": PrivacyTechnique.L_DIVERSITY,
            "l_value": l,
            "sensitive_attribute": sensitive_attribute,
            "utility_preservation": utility_score
        }
        
        return diversified_df, metadata
    
    def _calculate_utility_preservation(self, original_df: pd.DataFrame, 
                                      private_df: pd.DataFrame, 
                                      affected_columns: List[str]) -> float:
        
        if len(affected_columns) == 0:
            return 1.0
        
        utility_scores = []
        
        for column in affected_columns:
            if column not in original_df.columns or column not in private_df.columns:
                continue
            
            if original_df[column].dtype in ['int64', 'float64']:
                try:
                    orig_mean = original_df[column].mean()
                    priv_mean = private_df[column].mean()
                    orig_std = original_df[column].std()
                    priv_std = private_df[column].std()
                    
                    mean_preservation = 1 - abs(orig_mean - priv_mean) / (abs(orig_mean) + 1e-6)
                    std_preservation = 1 - abs(orig_std - priv_std) / (orig_std + 1e-6)
                    
                    utility_scores.append((mean_preservation + std_preservation) / 2)
                except:
                    utility_scores.append(0.5)
            else:
                try:
                    orig_counts = original_df[column].value_counts(normalize=True)
                    priv_counts = private_df[column].value_counts(normalize=True)
                    
                    common_values = set(orig_counts.index) & set(priv_counts.index)
                    distribution_similarity = sum(min(orig_counts.get(val, 0), priv_counts.get(val, 0)) 
                                                for val in common_values)
                    
                    utility_scores.append(distribution_similarity)
                except:
                    utility_scores.append(0.5)
        
        return np.mean(utility_scores) if utility_scores else 0.5
    
    def generate_synthetic_data(self, df: pd.DataFrame, num_samples: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        num_samples = num_samples or len(df)
        synthetic_df = df.copy()
        
        generation_methods = {}
        
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                mean = df[column].mean()
                std = df[column].std()
                synthetic_values = np.random.normal(mean, std, num_samples)
                
                if df[column].dtype == 'int64':
                    synthetic_values = synthetic_values.round().astype(int)
                
                generation_methods[column] = {"method": "gaussian", "mean": mean, "std": std}
                
            else:
                value_counts = df[column].value_counts(normalize=True)
                synthetic_values = np.random.choice(
                    value_counts.index, 
                    size=num_samples, 
                    p=value_counts.values
                )
                generation_methods[column] = {"method": "categorical_sampling", "distribution": value_counts.to_dict()}
            
            synthetic_df = synthetic_df.iloc[:num_samples].copy() if len(synthetic_df) > num_samples else synthetic_df
            synthetic_df[column] = synthetic_values[:len(synthetic_df)]
        
        utility_score = self._calculate_utility_preservation(df, synthetic_df, list(df.columns))
        
        metadata = {
            "technique": PrivacyTechnique.SYNTHETIC_DATA,
            "generation_methods": generation_methods,
            "num_samples": num_samples,
            "utility_preservation": utility_score
        }
        
        return synthetic_df, metadata
    
    def apply_comprehensive_privacy_protection(self, df: pd.DataFrame,
                                             quasi_identifiers: List[str] = None,
                                             sensitive_attributes: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        
        quasi_identifiers = quasi_identifiers or self._auto_detect_quasi_identifiers(df)
        sensitive_attributes = sensitive_attributes or self._auto_detect_sensitive_attributes(df)
        
        protected_df = df.copy()
        applied_techniques = []
        all_metadata = {}
        
        if self.config.target_privacy_level in [PrivacyLevel.HIGH, PrivacyLevel.MAXIMUM]:
            protected_df, dp_metadata = self.apply_differential_privacy(protected_df)
            applied_techniques.append(PrivacyTechnique.DIFFERENTIAL_PRIVACY)
            all_metadata.update(dp_metadata)
        
        if quasi_identifiers:
            protected_df, ka_metadata = self.apply_k_anonymity(protected_df, quasi_identifiers)
            applied_techniques.append(PrivacyTechnique.K_ANONYMITY)
            all_metadata.update(ka_metadata)
        
        if sensitive_attributes and len(sensitive_attributes) > 0:
            for sensitive_attr in sensitive_attributes[:1]:
                protected_df, ld_metadata = self.apply_l_diversity(
                    protected_df, quasi_identifiers, sensitive_attr
                )
                applied_techniques.append(PrivacyTechnique.L_DIVERSITY)
                all_metadata.update(ld_metadata)
                break
        
        final_utility = self._calculate_utility_preservation(df, protected_df, list(df.columns))
        
        comprehensive_metadata = {
            "applied_techniques": [tech.value for tech in applied_techniques],
            "overall_utility_preservation": final_utility,
            "privacy_level": self.config.target_privacy_level.value,
            "technique_details": all_metadata
        }
        
        return protected_df, comprehensive_metadata
    
    def evaluate_privacy_utility_tradeoff(self, original_df: pd.DataFrame, 
                                        protected_df: pd.DataFrame) -> Dict[str, Any]:
        
        utility_preservation = self._calculate_utility_preservation(
            original_df, protected_df, list(original_df.columns)
        )
        
        privacy_assessment = self.assess_privacy_risk(protected_df)
        
        tradeoff_score = (utility_preservation + privacy_assessment.privacy_score) / 2
        
        return {
            "utility_preservation": utility_preservation,
            "privacy_score": privacy_assessment.privacy_score,
            "reidentification_risk": privacy_assessment.reidentification_risk,
            "tradeoff_score": tradeoff_score,
            "recommendation": self._interpret_tradeoff(utility_preservation, privacy_assessment.privacy_score)
        }
    
    def _interpret_tradeoff(self, utility: float, privacy: float) -> str:
        if utility > 0.8 and privacy > 0.8:
            return "Excellent balance - high utility with strong privacy protection"
        elif utility > 0.6 and privacy > 0.6:
            return "Good balance - acceptable utility with adequate privacy protection"
        elif utility > privacy:
            return "Utility-favored - consider stronger privacy techniques"
        elif privacy > utility:
            return "Privacy-favored - consider utility-preserving techniques"
        else:
            return "Poor balance - both utility and privacy need improvement"