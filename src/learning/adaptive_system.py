import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging

@dataclass
class LearningEvent:
    timestamp: datetime
    event_type: str
    context: Dict[str, Any]
    outcome: Dict[str, Any]
    success_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PatternInsight:
    pattern_id: str
    pattern_type: str
    description: str
    confidence: float
    frequency: int
    conditions: Dict[str, Any]
    recommendations: List[str]
    last_updated: datetime

class AdaptiveLearningSystem:
    """Self-learning system that improves pipeline performance over time"""
    
    def __init__(self, learning_store_path: str = "./learning_store"):
        self.learning_store_path = Path(learning_store_path)
        self.learning_store_path.mkdir(exist_ok=True)
        
        # Learning components
        self.event_history: deque = deque(maxlen=10000)
        self.pattern_library: Dict[str, PatternInsight] = {}
        self.performance_tracker: Dict[str, List[float]] = defaultdict(list)
        self.adaptation_rules: Dict[str, Dict] = {}
        
        # Configuration
        self.min_events_for_pattern = 5
        self.pattern_confidence_threshold = 0.7
        self.performance_window_size = 100
        
        # Load existing knowledge
        self._load_learning_state()
        
        self.logger = logging.getLogger("AdaptiveLearningSystem")
    
    def record_pipeline_execution(self, pipeline_results: Dict[str, Any],
                                intelligence_profile: Dict[str, Any],
                                execution_metadata: Dict[str, Any]):
        """Record pipeline execution for learning"""
        
        # Extract learning signals
        success_score = self._calculate_success_score(pipeline_results)
        
        event = LearningEvent(
            timestamp=datetime.now(),
            event_type="pipeline_execution",
            context={
                'domain': self._extract_domain_context(intelligence_profile),
                'data_characteristics': self._extract_data_characteristics(intelligence_profile),
                'feature_engineering': self._extract_fe_context(pipeline_results),
                'pipeline_config': execution_metadata.get('config', {})
            },
            outcome={
                'success_score': success_score,
                'performance_metrics': pipeline_results.get('performance_metrics', {}),
                'execution_time': execution_metadata.get('total_time', 0),
                'stages_completed': execution_metadata.get('successful_stages', 0)
            },
            success_score=success_score,
            metadata=execution_metadata
        )
        
        self.event_history.append(event)
        self._update_performance_tracking(event)
        
        # Trigger pattern discovery and adaptation
        self._discover_new_patterns()
        self._update_adaptation_rules()
        
        # Persist learning
        self._save_learning_state()
    
    def get_adaptive_recommendations(self, intelligence_profile: Dict[str, Any],
                                   current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations based on learned patterns"""
        
        context = {
            'domain': self._extract_domain_context(intelligence_profile),
            'data_characteristics': self._extract_data_characteristics(intelligence_profile)
        }
        
        recommendations = {
            'feature_engineering': self._recommend_feature_engineering(context),
            'pipeline_config': self._recommend_pipeline_config(context),
            'data_processing': self._recommend_data_processing(context),
            'model_selection': self._recommend_model_selection(context)
        }
        
        return {
            'adaptive_recommendations': recommendations,
            'confidence_scores': self._calculate_recommendation_confidence(recommendations, context),
            'learning_insights': self._generate_learning_insights()
        }
    
    def _calculate_success_score(self, pipeline_results: Dict[str, Any]) -> float:
        """Calculate overall success score for pipeline execution"""
        
        scores = []
        
        # Performance score
        performance_metrics = pipeline_results.get('performance_metrics', {})
        if 'accuracy' in performance_metrics:
            scores.append(performance_metrics['accuracy'])
        elif 'r2_score' in performance_metrics:
            scores.append(max(0, performance_metrics['r2_score']))
        
        # Execution efficiency score
        execution_time = pipeline_results.get('execution_time', 0)
        if execution_time > 0:
            efficiency_score = min(1.0, 300 / execution_time)  # 5 minutes baseline
            scores.append(efficiency_score)
        
        # Data quality score
        validation_metrics = pipeline_results.get('validation_metrics', {})
        if 'data_quality_score' in validation_metrics:
            scores.append(validation_metrics['data_quality_score'])
        
        # Stage completion score
        successful_stages = pipeline_results.get('successful_stages', 0)
        total_stages = pipeline_results.get('total_stages', 7)
        completion_score = successful_stages / total_stages
        scores.append(completion_score)
        
        return np.mean(scores) if scores else 0.5
    
    def _extract_domain_context(self, intelligence_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Extract domain context for learning"""
        domain_analysis = intelligence_profile.get('domain_analysis', {})
        detected_domains = domain_analysis.get('detected_domains', [])
        
        return {
            'primary_domain': detected_domains[0].get('domain') if detected_domains else 'unknown',
            'domain_confidence': detected_domains[0].get('confidence', 0) if detected_domains else 0,
            'domain_count': len(detected_domains)
        }
    
    def _extract_data_characteristics(self, intelligence_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data characteristics for learning"""
        column_profiles = intelligence_profile.get('column_profiles', {})
        
        semantic_types = [profile.semantic_type.value for profile in column_profiles.values()]
        type_distribution = {st: semantic_types.count(st) for st in set(semantic_types)}
        
        return {
            'column_count': len(column_profiles),
            'semantic_diversity': len(set(semantic_types)) / len(semantic_types) if semantic_types else 0,
            'dominant_semantic_types': sorted(type_distribution.items(), key=lambda x: x[1], reverse=True)[:3],
            'has_temporal': any('datetime' in st for st in semantic_types),
            'has_text': any('text' in st for st in semantic_types),
            'has_geographic': any('geo' in st or 'location' in st for st in semantic_types)
        }
    
    def _extract_fe_context(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract feature engineering context"""
        fe_results = pipeline_results.get('feature_engineering_results', {})
        
        return {
            'original_features': fe_results.get('original_features', 0),
            'engineered_features': fe_results.get('engineered_features', 0),
            'feature_expansion_ratio': fe_results.get('engineered_features', 0) / max(1, fe_results.get('original_features', 1)),
            'applied_techniques': fe_results.get('applied_techniques', [])
        }
    
    def _update_performance_tracking(self, event: LearningEvent):
        """Update performance tracking metrics"""
        
        # Track overall performance
        self.performance_tracker['overall'].append(event.success_score)
        
        # Track domain-specific performance
        domain = event.context.get('domain', {}).get('primary_domain', 'unknown')
        self.performance_tracker[f'domain_{domain}'].append(event.success_score)
        
        # Track by data characteristics
        data_chars = event.context.get('data_characteristics', {})
        if data_chars.get('has_temporal'):
            self.performance_tracker['temporal_data'].append(event.success_score)
        if data_chars.get('has_text'):
            self.performance_tracker['text_data'].append(event.success_score)
    
    def _discover_new_patterns(self):
        """Discover new patterns from recent events"""
        
        if len(self.event_history) < self.min_events_for_pattern:
            return
        
        # Group events by context similarities
        context_groups = self._group_events_by_context()
        
        for group_key, events in context_groups.items():
            if len(events) >= self.min_events_for_pattern:
                pattern = self._analyze_event_pattern(group_key, events)
                if pattern and pattern.confidence >= self.pattern_confidence_threshold:
                    self.pattern_library[pattern.pattern_id] = pattern
    
    def _group_events_by_context(self) -> Dict[str, List[LearningEvent]]:
        """Group events by similar context characteristics"""
        
        groups = defaultdict(list)
        
        for event in self.event_history:
            if event.event_type == 'pipeline_execution':
                # Create grouping key based on context
                domain = event.context.get('domain', {}).get('primary_domain', 'unknown')
                data_chars = event.context.get('data_characteristics', {})
                
                group_key = f"{domain}_{data_chars.get('column_count', 0)//10}_{data_chars.get('has_temporal', False)}_{data_chars.get('has_text', False)}"
                groups[group_key].append(event)
        
        return groups
    
    def _analyze_event_pattern(self, group_key: str, events: List[LearningEvent]) -> Optional[PatternInsight]:
        """Analyze a group of events to extract patterns"""
        
        if not events:
            return None
        
        # Calculate pattern statistics
        success_scores = [event.success_score for event in events]
        avg_success = np.mean(success_scores)
        success_std = np.std(success_scores)
        
        # Identify high-performing configurations
        high_performers = [event for event in events if event.success_score > avg_success + 0.1]
        
        if not high_performers:
            return None
        
        # Extract common characteristics of high performers
        common_configs = self._extract_common_configurations(high_performers)
        
        pattern_id = f"pattern_{group_key}_{datetime.now().strftime('%Y%m%d')}"
        
        return PatternInsight(
            pattern_id=pattern_id,
            pattern_type="high_performance_config",
            description=f"High-performing configuration for {group_key}",
            confidence=min(1.0, (avg_success - 0.5) * 2),  # Scale 0.5-1.0 to 0-1.0
            frequency=len(high_performers),
            conditions=self._extract_pattern_conditions(events[0]),
            recommendations=self._generate_pattern_recommendations(common_configs),
            last_updated=datetime.now()
        )
    
    def _extract_common_configurations(self, events: List[LearningEvent]) -> Dict[str, Any]:
        """Extract common configurations from high-performing events"""
        
        common_configs = {
            'feature_engineering': defaultdict(int),
            'pipeline_settings': defaultdict(int),
            'data_processing': defaultdict(int)
        }
        
        for event in events:
            # Feature engineering patterns
            fe_context = event.context.get('feature_engineering', {})
            for technique in fe_context.get('applied_techniques', []):
                common_configs['feature_engineering'][technique] += 1
            
            # Pipeline configuration patterns  
            config = event.context.get('pipeline_config', {})
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    common_configs['pipeline_settings'][f"{key}_{value}"] += 1
        
        # Keep only frequent patterns (>50% of events)
        threshold = len(events) * 0.5
        filtered_configs = {}
        
        for category, configs in common_configs.items():
            filtered_configs[category] = {k: v for k, v in configs.items() if v >= threshold}
        
        return filtered_configs
    
    def _extract_pattern_conditions(self, reference_event: LearningEvent) -> Dict[str, Any]:
        """Extract conditions under which pattern applies"""
        
        return {
            'domain': reference_event.context.get('domain', {}),
            'data_characteristics': reference_event.context.get('data_characteristics', {}),
            'min_success_score': 0.7
        }
    
    def _generate_pattern_recommendations(self, common_configs: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on discovered patterns"""
        
        recommendations = []
        
        # Feature engineering recommendations
        fe_configs = common_configs.get('feature_engineering', {})
        if fe_configs:
            top_techniques = sorted(fe_configs.items(), key=lambda x: x[1], reverse=True)[:3]
            recommendations.extend([f"Apply {technique}" for technique, _ in top_techniques])
        
        # Pipeline recommendations
        pipeline_configs = common_configs.get('pipeline_settings', {})
        if pipeline_configs:
            recommendations.extend([f"Use {config}" for config in list(pipeline_configs.keys())[:3]])
        
        return recommendations
    
    def _update_adaptation_rules(self):
        """Update adaptation rules based on learned patterns"""
        
        # Clear old rules
        self.adaptation_rules.clear()
        
        # Generate rules from high-confidence patterns
        for pattern in self.pattern_library.values():
            if pattern.confidence >= 0.8:
                rule_id = f"rule_{pattern.pattern_id}"
                self.adaptation_rules[rule_id] = {
                    'conditions': pattern.conditions,
                    'actions': pattern.recommendations,
                    'confidence': pattern.confidence,
                    'last_applied': None
                }
    
    def _recommend_feature_engineering(self, context: Dict[str, Any]) -> List[str]:
        """Recommend feature engineering based on learned patterns"""
        
        matching_patterns = self._find_matching_patterns(context)
        recommendations = []
        
        for pattern in matching_patterns:
            if 'feature_engineering' in pattern.recommendations[0].lower():
                recommendations.extend(pattern.recommendations)
        
        # Add domain-specific learned recommendations
        domain = context.get('domain', {}).get('primary_domain', 'unknown')
        domain_performance = self.performance_tracker.get(f'domain_{domain}', [])
        
        if domain_performance and np.mean(domain_performance[-10:]) > 0.8:
            recommendations.append(f"Apply domain-optimized feature engineering for {domain}")
        
        return list(set(recommendations))[:5]  # Top 5 unique recommendations
    
    def _recommend_pipeline_config(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend pipeline configuration based on learned patterns"""
        
        config_recommendations = {}
        matching_patterns = self._find_matching_patterns(context)
        
        for pattern in matching_patterns:
            conditions = pattern.conditions
            data_chars = conditions.get('data_characteristics', {})
            
            # Recommend caching for large datasets
            if data_chars.get('column_count', 0) > 50:
                config_recommendations['enable_caching'] = True
                config_recommendations['cache_intermediate'] = True
            
            # Recommend parallel processing for complex features
            if data_chars.get('semantic_diversity', 0) > 0.7:
                config_recommendations['parallel_processing'] = True
                config_recommendations['max_workers'] = 4
        
        return config_recommendations
    
    def _recommend_data_processing(self, context: Dict[str, Any]) -> List[str]:
        """Recommend data processing steps based on learned patterns"""
        
        recommendations = []
        domain = context.get('domain', {}).get('primary_domain', 'unknown')
        
        # Domain-specific processing recommendations
        if domain == 'finance':
            recommendations.extend([
                "Apply robust scaling for financial outliers",
                "Use time-aware validation splits",
                "Implement feature importance tracking"
            ])
        elif domain == 'ecommerce':
            recommendations.extend([
                "Handle seasonal patterns in validation",
                "Apply customer-level feature aggregation",
                "Use stratified sampling for customer segments"
            ])
        
        return recommendations
    
    def _recommend_model_selection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend model selection based on learned patterns"""
        
        data_chars = context.get('data_characteristics', {})
        
        recommendations = {
            'primary_algorithms': [],
            'ensemble_methods': [],
            'hyperparameter_focus': []
        }
        
        # Based on data characteristics
        if data_chars.get('has_temporal'):
            recommendations['primary_algorithms'].extend(['XGBoost', 'LightGBM', 'TimeSeriesForest'])
            recommendations['hyperparameter_focus'].append('learning_rate')
        
        if data_chars.get('semantic_diversity', 0) > 0.8:
            recommendations['ensemble_methods'].append('StackingClassifier')
            recommendations['hyperparameter_focus'].append('max_depth')
        
        return recommendations
    
    def _find_matching_patterns(self, context: Dict[str, Any]) -> List[PatternInsight]:
        """Find patterns that match the current context"""
        
        matching_patterns = []
        
        for pattern in self.pattern_library.values():
            if self._context_matches_pattern(context, pattern.conditions):
                matching_patterns.append(pattern)
        
        # Sort by confidence and recency
        return sorted(matching_patterns, 
                     key=lambda p: (p.confidence, p.last_updated), 
                     reverse=True)
    
    def _context_matches_pattern(self, context: Dict[str, Any], 
                               pattern_conditions: Dict[str, Any]) -> bool:
        """Check if current context matches pattern conditions"""
        
        # Domain matching
        context_domain = context.get('domain', {}).get('primary_domain', 'unknown')
        pattern_domain = pattern_conditions.get('domain', {}).get('primary_domain', 'unknown')
        
        if context_domain != pattern_domain:
            return False
        
        # Data characteristics matching
        context_chars = context.get('data_characteristics', {})
        pattern_chars = pattern_conditions.get('data_characteristics', {})
        
        # Check key characteristics
        char_matches = 0
        char_total = 0
        
        for key in ['has_temporal', 'has_text', 'has_geographic']:
            if key in pattern_chars:
                char_total += 1
                if context_chars.get(key) == pattern_chars.get(key):
                    char_matches += 1
        
        return char_matches / max(1, char_total) >= 0.7
    
    def _calculate_recommendation_confidence(self, recommendations: Dict[str, Any],
                                          context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for recommendations"""
        
        confidence_scores = {}
        matching_patterns = self._find_matching_patterns(context)
        
        for category, recs in recommendations.items():
            if matching_patterns:
                avg_confidence = np.mean([p.confidence for p in matching_patterns])
                confidence_scores[category] = min(0.95, avg_confidence + 0.1)
            else:
                confidence_scores[category] = 0.5  # Default confidence
        
        return confidence_scores
    
    def _generate_learning_insights(self) -> Dict[str, Any]:
        """Generate insights about the learning system state"""
        
        return {
            'total_patterns_learned': len(self.pattern_library),
            'high_confidence_patterns': len([p for p in self.pattern_library.values() if p.confidence > 0.8]),
            'recent_performance_trend': self._calculate_performance_trend(),
            'most_successful_domain': self._identify_best_performing_domain(),
            'learning_velocity': len(self.event_history) / max(1, (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)).days + 1)
        }
    
    def _calculate_performance_trend(self) -> str:
        """Calculate recent performance trend"""
        
        overall_scores = self.performance_tracker['overall']
        if len(overall_scores) < 10:
            return "insufficient_data"
        
        recent_avg = np.mean(overall_scores[-10:])
        older_avg = np.mean(overall_scores[-20:-10]) if len(overall_scores) >= 20 else np.mean(overall_scores[:-10])
        
        if recent_avg > older_avg + 0.05:
            return "improving"
        elif recent_avg < older_avg - 0.05:
            return "declining"
        else:
            return "stable"
    
    def _identify_best_performing_domain(self) -> str:
        """Identify the best performing domain"""
        
        domain_averages = {}
        
        for key, scores in self.performance_tracker.items():
            if key.startswith('domain_') and len(scores) >= 3:
                domain = key.replace('domain_', '')
                domain_averages[domain] = np.mean(scores)
        
        if domain_averages:
            return max(domain_averages.items(), key=lambda x: x[1])[0]
        else:
            return "unknown"
    
    def _save_learning_state(self):
        """Save learning state to persistent storage"""
        
        state = {
            'event_history': list(self.event_history),
            'pattern_library': {k: v.__dict__ for k, v in self.pattern_library.items()},
            'performance_tracker': dict(self.performance_tracker),
            'adaptation_rules': self.adaptation_rules,
            'last_updated': datetime.now().isoformat()
        }
        
        state_file = self.learning_store_path / "learning_state.json"
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save learning state: {e}")
    
    def _load_learning_state(self):
        """Load learning state from persistent storage"""
        
        state_file = self.learning_store_path / "learning_state.json"
        
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Restore event history
            events = state.get('event_history', [])
            for event_data in events:
                event = LearningEvent(**event_data)
                self.event_history.append(event)
            
            # Restore pattern library
            patterns = state.get('pattern_library', {})
            for pattern_id, pattern_data in patterns.items():
                pattern_data['last_updated'] = datetime.fromisoformat(pattern_data['last_updated'])
                self.pattern_library[pattern_id] = PatternInsight(**pattern_data)
            
            # Restore performance tracker
            perf_data = state.get('performance_tracker', {})
            for key, scores in perf_data.items():
                self.performance_tracker[key] = scores
            
            # Restore adaptation rules
            self.adaptation_rules = state.get('adaptation_rules', {})
            
            self.logger.info(f"Loaded learning state with {len(self.pattern_library)} patterns")
            
        except Exception as e:
            self.logger.warning(f"Failed to load learning state: {e}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning system summary"""
        
        return {
            'system_stats': {
                'total_executions': len(self.event_history),
                'patterns_discovered': len(self.pattern_library),
                'adaptation_rules': len(self.adaptation_rules),
                'domains_learned': len([k for k in self.performance_tracker.keys() if k.startswith('domain_')])
            },
            'performance_insights': {
                'overall_trend': self._calculate_performance_trend(),
                'best_domain': self._identify_best_performing_domain(),
                'average_success_rate': np.mean(self.performance_tracker['overall']) if self.performance_tracker['overall'] else 0
            },
            'recent_patterns': [
                {
                    'pattern_id': p.pattern_id,
                    'description': p.description,
                    'confidence': p.confidence,
                    'frequency': p.frequency
                }
                for p in sorted(self.pattern_library.values(), 
                               key=lambda x: x.last_updated, reverse=True)[:5]
            ]
        }