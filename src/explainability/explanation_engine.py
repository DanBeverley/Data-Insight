import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

@dataclass
class GlobalExplanation:
    feature_importance: Dict[str, float]
    feature_interactions: Optional[Dict[str, float]] = None
    partial_dependence: Optional[Dict[str, Any]] = None
    explanation_method: str = "fallback"

@dataclass
class LocalExplanation:
    instance_id: Union[str, int]
    prediction: float
    feature_contributions: Dict[str, float]
    counterfactuals: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    explanation_method: str = "fallback"

@dataclass
class BusinessInsight:
    insight_type: str
    message: str
    confidence: float
    supporting_features: List[str]
    recommendation: Optional[str] = None

class ExplanationEngine:
    def __init__(self, model, X_train: pd.DataFrame, task_type: str = "classification"):
        self.model = model
        self.X_train = X_train
        self.task_type = task_type
        self.feature_names = list(X_train.columns)
        
        self.shap_explainer = None
        self.lime_explainer = None
        
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        if SHAP_AVAILABLE:
            try:
                if hasattr(self.model, 'predict_proba'):
                    self.shap_explainer = shap.Explainer(self.model.predict_proba, self.X_train)
                else:
                    self.shap_explainer = shap.Explainer(self.model.predict, self.X_train)
            except Exception:
                self.shap_explainer = None
        
        if LIME_AVAILABLE:
            try:
                mode = 'classification' if self.task_type == 'classification' else 'regression'
                self.lime_explainer = lime_tabular.LimeTabularExplainer(
                    self.X_train.values,
                    feature_names=self.feature_names,
                    mode=mode,
                    random_state=42
                )
            except Exception:
                self.lime_explainer = None
    
    def explain_global(self, method: str = "auto") -> GlobalExplanation:
        if method == "auto":
            method = "shap" if self.shap_explainer else "fallback"
        
        if method == "shap" and self.shap_explainer:
            return self._shap_global_explanation()
        else:
            return self._fallback_global_explanation()
    
    def _shap_global_explanation(self) -> GlobalExplanation:
        try:
            shap_values = self.shap_explainer(self.X_train.sample(min(100, len(self.X_train))))
            
            if hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) == 3:
                    importance_values = np.abs(shap_values.values[:, :, 1]).mean(axis=0)
                else:
                    importance_values = np.abs(shap_values.values).mean(axis=0)
            else:
                importance_values = np.abs(shap_values).mean(axis=0)
            
            feature_importance = dict(zip(self.feature_names, importance_values))
            
            interactions = self._calculate_shap_interactions(shap_values)
            
            return GlobalExplanation(
                feature_importance=feature_importance,
                feature_interactions=interactions,
                explanation_method="shap"
            )
            
        except Exception:
            return self._fallback_global_explanation()
    
    def _calculate_shap_interactions(self, shap_values) -> Dict[str, float]:
        try:
            interactions = {}
            values = shap_values.values if hasattr(shap_values, 'values') else shap_values
            
            if len(values.shape) == 3:
                values = values[:, :, 1]
            
            for i in range(min(5, len(self.feature_names))):
                for j in range(i+1, min(5, len(self.feature_names))):
                    interaction_strength = np.corrcoef(values[:, i], values[:, j])[0, 1]
                    if not np.isnan(interaction_strength):
                        interactions[f"{self.feature_names[i]}_x_{self.feature_names[j]}"] = abs(interaction_strength)
            
            return dict(sorted(interactions.items(), key=lambda x: x[1], reverse=True)[:10])
            
        except Exception:
            return {}
    
    def _fallback_global_explanation(self) -> GlobalExplanation:
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_values = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importance_values = np.abs(self.model.coef_).flatten()
            else:
                from sklearn.inspection import permutation_importance
                perm_importance = permutation_importance(
                    self.model, self.X_train.sample(min(200, len(self.X_train))), 
                    self.model.predict(self.X_train.sample(min(200, len(self.X_train)))),
                    n_repeats=3, random_state=42
                )
                importance_values = perm_importance.importances_mean
            
            feature_importance = dict(zip(self.feature_names, importance_values))
            
            return GlobalExplanation(
                feature_importance=feature_importance,
                explanation_method="fallback"
            )
            
        except Exception:
            uniform_importance = 1.0 / len(self.feature_names)
            feature_importance = {name: uniform_importance for name in self.feature_names}
            
            return GlobalExplanation(
                feature_importance=feature_importance,
                explanation_method="uniform"
            )
    
    def explain_local(self, instance: pd.Series, method: str = "auto") -> LocalExplanation:
        if method == "auto":
            method = "shap" if self.shap_explainer else "lime" if self.lime_explainer else "fallback"
        
        instance_id = getattr(instance, 'name', 0)
        
        if method == "shap" and self.shap_explainer:
            return self._shap_local_explanation(instance, instance_id)
        elif method == "lime" and self.lime_explainer:
            return self._lime_local_explanation(instance, instance_id)
        else:
            return self._fallback_local_explanation(instance, instance_id)
    
    def _shap_local_explanation(self, instance: pd.Series, instance_id) -> LocalExplanation:
        try:
            instance_df = pd.DataFrame([instance], columns=self.feature_names)
            shap_values = self.shap_explainer(instance_df)
            
            if hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) == 3:
                    contributions = shap_values.values[0, :, 1]
                else:
                    contributions = shap_values.values[0]
            else:
                contributions = shap_values[0]
            
            feature_contributions = dict(zip(self.feature_names, contributions))
            
            prediction = self.model.predict(instance_df)[0]
            if hasattr(self.model, 'predict_proba'):
                confidence = np.max(self.model.predict_proba(instance_df)[0])
            else:
                confidence = 0.5
            
            return LocalExplanation(
                instance_id=instance_id,
                prediction=prediction,
                feature_contributions=feature_contributions,
                confidence=confidence,
                explanation_method="shap"
            )
            
        except Exception:
            return self._fallback_local_explanation(instance, instance_id)
    
    def _lime_local_explanation(self, instance: pd.Series, instance_id) -> LocalExplanation:
        try:
            explanation = self.lime_explainer.explain_instance(
                instance.values, 
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                num_features=min(10, len(self.feature_names))
            )
            
            contributions = dict(explanation.as_list())
            feature_contributions = {name: contributions.get(name, 0.0) for name in self.feature_names}
            
            prediction = self.model.predict(pd.DataFrame([instance], columns=self.feature_names))[0]
            confidence = explanation.score if hasattr(explanation, 'score') else 0.5
            
            return LocalExplanation(
                instance_id=instance_id,
                prediction=prediction,
                feature_contributions=feature_contributions,
                confidence=confidence,
                explanation_method="lime"
            )
            
        except Exception:
            return self._fallback_local_explanation(instance, instance_id)
    
    def _fallback_local_explanation(self, instance: pd.Series, instance_id) -> LocalExplanation:
        try:
            global_explanation = self.explain_global()
            
            feature_contributions = {}
            for feature, importance in global_explanation.feature_importance.items():
                if feature in instance.index:
                    feature_value = instance[feature]
                    if pd.isna(feature_value):
                        contribution = 0.0
                    else:
                        feature_mean = self.X_train[feature].mean()
                        contribution = importance * (feature_value - feature_mean)
                    feature_contributions[feature] = contribution
                else:
                    feature_contributions[feature] = 0.0
            
            prediction = self.model.predict(pd.DataFrame([instance], columns=self.feature_names))[0]
            
            return LocalExplanation(
                instance_id=instance_id,
                prediction=prediction,
                feature_contributions=feature_contributions,
                confidence=0.5,
                explanation_method="fallback"
            )
            
        except Exception:
            return LocalExplanation(
                instance_id=instance_id,
                prediction=0.0,
                feature_contributions={name: 0.0 for name in self.feature_names},
                confidence=0.0,
                explanation_method="error"
            )
    
    def generate_business_insights(self, global_explanation: GlobalExplanation, 
                                 sample_instances: Optional[pd.DataFrame] = None) -> List[BusinessInsight]:
        insights = []
        
        top_features = sorted(global_explanation.feature_importance.items(), 
                            key=lambda x: x[1], reverse=True)[:5]
        
        insights.append(BusinessInsight(
            insight_type="key_drivers",
            message=f"Top predictive factors: {', '.join([f[0] for f in top_features[:3]])}",
            confidence=0.9,
            supporting_features=[f[0] for f in top_features[:3]],
            recommendation="Focus on these key factors for maximum impact"
        ))
        
        if global_explanation.feature_interactions:
            top_interaction = max(global_explanation.feature_interactions.items(), key=lambda x: x[1])
            insights.append(BusinessInsight(
                insight_type="interaction_effect",
                message=f"Strong interaction detected between {top_interaction[0].replace('_x_', ' and ')}",
                confidence=0.7,
                supporting_features=top_interaction[0].split('_x_'),
                recommendation="Consider combined strategies for these features"
            ))
        
        if sample_instances is not None and len(sample_instances) > 0:
            insights.extend(self._analyze_prediction_patterns(sample_instances))
        
        return insights
    
    def _analyze_prediction_patterns(self, instances: pd.DataFrame) -> List[BusinessInsight]:
        insights = []
        
        try:
            predictions = self.model.predict(instances)
            
            if self.task_type == "classification":
                positive_rate = np.mean(predictions)
                if positive_rate > 0.7:
                    insights.append(BusinessInsight(
                        insight_type="prediction_pattern",
                        message=f"High positive prediction rate: {positive_rate:.1%}",
                        confidence=0.8,
                        supporting_features=[],
                        recommendation="Review data distribution for potential bias"
                    ))
                elif positive_rate < 0.3:
                    insights.append(BusinessInsight(
                        insight_type="prediction_pattern",
                        message=f"Low positive prediction rate: {positive_rate:.1%}",
                        confidence=0.8,
                        supporting_features=[],
                        recommendation="Consider model recalibration or threshold adjustment"
                    ))
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(instances)
                max_probs = np.max(probabilities, axis=1)
                avg_confidence = np.mean(max_probs)
                
                if avg_confidence < 0.6:
                    insights.append(BusinessInsight(
                        insight_type="confidence_analysis",
                        message=f"Low average prediction confidence: {avg_confidence:.1%}",
                        confidence=0.7,
                        supporting_features=[],
                        recommendation="Consider gathering more training data or feature engineering"
                    ))
                    
        except Exception:
            pass
        
        return insights
    
    def explain_prediction_difference(self, instance1: pd.Series, instance2: pd.Series) -> Dict[str, Any]:
        explanation1 = self.explain_local(instance1)
        explanation2 = self.explain_local(instance2)
        
        pred_diff = explanation1.prediction - explanation2.prediction
        
        contribution_diffs = {}
        for feature in self.feature_names:
            diff = explanation1.feature_contributions.get(feature, 0) - explanation2.feature_contributions.get(feature, 0)
            contribution_diffs[feature] = diff
        
        top_differences = sorted(contribution_diffs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        return {
            'prediction_difference': pred_diff,
            'top_contributing_differences': top_differences,
            'instance1_prediction': explanation1.prediction,
            'instance2_prediction': explanation2.prediction
        }
    
    def get_explanation_summary(self) -> Dict[str, Any]:
        return {
            'available_methods': {
                'shap': SHAP_AVAILABLE and self.shap_explainer is not None,
                'lime': LIME_AVAILABLE and self.lime_explainer is not None,
                'fallback': True
            },
            'model_type': type(self.model).__name__,
            'task_type': self.task_type,
            'feature_count': len(self.feature_names),
            'training_samples': len(self.X_train)
        }