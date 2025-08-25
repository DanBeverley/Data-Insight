"""Production-grade Strategy Translator - Strategic to Technical Translation System"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .project_definition import (
    ProjectDefinition, Objective, Domain, Priority, RiskLevel, ProjectPhase,
    BusinessContext, TechnicalConstraints, RegulatoryConstraints
)

class TranslationStrategy(Enum):
    """Strategy for translating business objectives to technical configurations."""
    CONSERVATIVE = "conservative"  # Safe, proven approaches
    BALANCED = "balanced"         # Balance between innovation and safety
    AGGRESSIVE = "aggressive"     # Cutting-edge, high-risk high-reward
    COMPLIANCE_FIRST = "compliance_first"  # Prioritize regulatory compliance

class ModelComplexity(Enum):
    """Model complexity levels based on business requirements."""
    SIMPLE = "simple"        # Linear models, simple trees
    MODERATE = "moderate"    # Random forests, gradient boosting
    COMPLEX = "complex"      # Deep learning, ensemble methods
    EXPERIMENTAL = "experimental"  # Latest research models

@dataclass
class TechnicalConfiguration:
    """Technical configuration derived from business strategy."""
    
    # Model selection guidance
    recommended_algorithms: List[str]
    model_complexity: ModelComplexity
    ensemble_strategy: bool
    
    # Performance targets
    accuracy_target: float
    latency_target_ms: Optional[int]
    throughput_target: Optional[int]
    
    # Resource allocation
    max_training_time_hours: Optional[float]
    max_memory_gb: Optional[float]
    compute_budget_priority: str  # "cost_optimized", "performance_optimized", "balanced"
    
    # Feature engineering guidance
    feature_engineering_complexity: str  # "basic", "advanced", "automated"
    feature_selection_strategy: str
    
    # Validation strategy
    validation_rigor: str  # "standard", "extensive", "regulatory"
    cross_validation_folds: int
    test_set_percentage: float
    
    # Deployment configuration
    deployment_strategy: str  # "batch", "real_time", "hybrid"
    scaling_requirements: Dict[str, Any]
    monitoring_level: str  # "basic", "comprehensive", "regulatory"
    
    # Interpretability requirements
    interpretability_level: str  # "none", "basic", "full"
    explanation_methods: List[str]
    
    # Security and compliance
    security_level: str  # "standard", "enhanced", "maximum"
    audit_requirements: List[str]
    data_governance: Dict[str, Any]

class StrategyTranslator:
    """Production-grade strategy translator that converts business objectives into technical configurations."""
    
    def __init__(self, translation_strategy: TranslationStrategy = TranslationStrategy.BALANCED):
        self.translation_strategy = translation_strategy
        self.domain_expertise = self._initialize_domain_expertise()
        self.objective_mappings = self._initialize_objective_mappings()
        self.compliance_mappings = self._initialize_compliance_mappings()
        
        try:
            from ..learning.persistent_storage import PersistentMetaDatabase, create_dataset_characteristics
            from ..database.service import get_database_service
            self.meta_db = PersistentMetaDatabase()
            self.db_service = get_database_service()
            self.enable_historical_learning = True
        except ImportError:
            self.meta_db = None
            self.db_service = None
            self.enable_historical_learning = False
        
    def translate(self, project_definition: ProjectDefinition, 
                 dataset_characteristics=None) -> TechnicalConfiguration:
        """Translate business strategy to comprehensive technical configuration with historical learning."""
        
        validation_results = project_definition.validate_project_definition()
        if validation_results["errors"]:
            raise ValueError(f"Invalid project definition: {validation_results['errors']}")
        
        historical_insights = self._get_historical_insights(project_definition, dataset_characteristics)
        recommended_strategies = self._get_recommended_strategies(project_definition, dataset_characteristics)
        model_complexity = self._determine_model_complexity(project_definition, historical_insights)
        
        # Select recommended algorithms
        algorithms = self._select_algorithms(project_definition, model_complexity, historical_insights)
        
        # Calculate performance targets
        performance_targets = self._calculate_performance_targets(project_definition)
        
        # Determine resource allocation
        resource_config = self._determine_resource_allocation(project_definition)
        
        # Configure feature engineering
        feature_config = self._configure_feature_engineering(project_definition)
        
        # Set validation strategy
        validation_config = self._configure_validation_strategy(project_definition)
        
        # Configure deployment strategy
        deployment_config = self._configure_deployment_strategy(project_definition)
        
        # Set interpretability requirements
        interpretability_config = self._configure_interpretability(project_definition)
        
        # Configure security and compliance
        security_config = self._configure_security_compliance(project_definition)
        
        tech_config = TechnicalConfiguration(
            recommended_algorithms=algorithms,
            model_complexity=model_complexity,
            ensemble_strategy=self._should_use_ensemble(project_definition),
            
            accuracy_target=performance_targets["accuracy"],
            latency_target_ms=performance_targets.get("latency_ms"),
            throughput_target=performance_targets.get("throughput"),
            
            max_training_time_hours=resource_config["training_hours"],
            max_memory_gb=resource_config["memory_gb"],
            compute_budget_priority=resource_config["budget_priority"],
            
            feature_engineering_complexity=feature_config["complexity"],
            feature_selection_strategy=feature_config["selection_strategy"],
            
            validation_rigor=validation_config["rigor"],
            cross_validation_folds=validation_config["cv_folds"],
            test_set_percentage=validation_config["test_percentage"],
            
            deployment_strategy=deployment_config["strategy"],
            scaling_requirements=deployment_config["scaling"],
            monitoring_level=deployment_config["monitoring"],
            
            interpretability_level=interpretability_config["level"],
            explanation_methods=interpretability_config["methods"],
            
            security_level=security_config["level"],
            audit_requirements=security_config["audit"],
            data_governance=security_config["governance"]
        )
        
        # Apply experience-based adjustments if available
        tech_config = self._apply_experience_based_adjustments(tech_config, recommended_strategies)
        
        return tech_config
    
    def translate_to_pipeline_config(self, project_definition: ProjectDefinition) -> Dict[str, Any]:
        """Translate to pipeline orchestrator configuration format."""
        
        tech_config = self.translate(project_definition)
        
        return {
            # Core configuration
            'project_id': project_definition.project_id,
            'objective': project_definition.objective.value,
            'domain': project_definition.domain.value,
            'priority': project_definition.priority.value,
            'risk_level': project_definition.risk_level.value,
            
            # Algorithm selection
            'algorithm_preferences': tech_config.recommended_algorithms,
            'model_complexity': tech_config.model_complexity.value,
            'use_ensemble': tech_config.ensemble_strategy,
            
            # Performance targets
            'performance_targets': {
                'accuracy': tech_config.accuracy_target,
                'latency_ms': tech_config.latency_target_ms,
                'throughput': tech_config.throughput_target
            },
            
            # Resource constraints
            'resource_constraints': {
                'max_training_hours': tech_config.max_training_time_hours,
                'max_memory_gb': tech_config.max_memory_gb,
                'budget_priority': tech_config.compute_budget_priority
            },
            
            # Feature engineering
            'feature_engineering': {
                'complexity': tech_config.feature_engineering_complexity,
                'selection_strategy': tech_config.feature_selection_strategy,
                'automated_fe': tech_config.feature_engineering_complexity == "automated"
            },
            
            # Validation configuration
            'validation_config': {
                'rigor_level': tech_config.validation_rigor,
                'cross_validation_folds': tech_config.cross_validation_folds,
                'test_set_size': tech_config.test_set_percentage,
                'validation_metrics': self._get_validation_metrics(project_definition)
            },
            
            # Interpretability
            'interpretability': {
                'required': tech_config.interpretability_level != "none",
                'level': tech_config.interpretability_level,
                'methods': tech_config.explanation_methods
            },
            
            # Deployment
            'deployment': {
                'strategy': tech_config.deployment_strategy,
                'scaling_requirements': tech_config.scaling_requirements,
                'monitoring_level': tech_config.monitoring_level
            },
            
            # Security and compliance
            'security': {
                'level': tech_config.security_level,
                'audit_required': len(tech_config.audit_requirements) > 0,
                'compliance_rules': project_definition.regulatory_constraints.compliance_rules
            },
            
            # Business context
            'business_context': {
                'stakeholders': project_definition.business_context.stakeholders,
                'success_criteria': project_definition.business_context.success_criteria,
                'timeline_months': project_definition.business_context.timeline_months
            }
        }
    
    def _determine_model_complexity(self, project_def: ProjectDefinition) -> ModelComplexity:
        """Determine appropriate model complexity based on project characteristics."""
        
        complexity_score = 0
        
        # Objective-based complexity
        if project_def.objective in [Objective.ACCURACY, Objective.INNOVATION]:
            complexity_score += 2
        elif project_def.objective in [Objective.SPEED, Objective.COST_EFFICIENCY]:
            complexity_score -= 1
        
        # Interpretability reduces complexity
        if project_def.regulatory_constraints.interpretability_required:
            complexity_score -= 2
        
        # Domain complexity
        if project_def.domain in [Domain.FINANCE, Domain.HEALTHCARE]:
            complexity_score += 1
        
        # Risk level adjustment
        if project_def.risk_level == RiskLevel.CRITICAL:
            if self.translation_strategy == TranslationStrategy.CONSERVATIVE:
                complexity_score -= 1
        
        # Translation strategy adjustment
        if self.translation_strategy == TranslationStrategy.AGGRESSIVE:
            complexity_score += 2
        elif self.translation_strategy == TranslationStrategy.CONSERVATIVE:
            complexity_score -= 1
        
        # Map score to complexity level
        if complexity_score <= -1:
            return ModelComplexity.SIMPLE
        elif complexity_score <= 2:
            return ModelComplexity.MODERATE
        elif complexity_score <= 4:
            return ModelComplexity.COMPLEX
        else:
            return ModelComplexity.EXPERIMENTAL
    
    def _select_algorithms(self, project_def: ProjectDefinition, complexity: ModelComplexity) -> List[str]:
        """Select appropriate algorithms based on project requirements."""
        
        # Base algorithms by complexity
        algorithm_map = {
            ModelComplexity.SIMPLE: ["linear_regression", "logistic_regression", "decision_tree"],
            ModelComplexity.MODERATE: ["random_forest", "gradient_boosting", "svm", "naive_bayes"],
            ModelComplexity.COMPLEX: ["xgboost", "lightgbm", "neural_network", "deep_learning"],
            ModelComplexity.EXPERIMENTAL: ["transformer", "automl", "neural_architecture_search"]
        }
        
        base_algorithms = algorithm_map[complexity].copy()
        
        # Domain-specific adjustments
        domain_preferences = self.domain_expertise.get(project_def.domain, {})
        if "preferred_algorithms" in domain_preferences:
            # Prioritize domain-preferred algorithms
            domain_algos = [algo for algo in domain_preferences["preferred_algorithms"] if algo in base_algorithms]
            other_algos = [algo for algo in base_algorithms if algo not in domain_algos]
            base_algorithms = domain_algos + other_algos
        
        # Objective-specific adjustments
        if project_def.objective == Objective.SPEED:
            # Prioritize fast algorithms
            speed_priority = ["linear_regression", "logistic_regression", "naive_bayes", "decision_tree"]
            base_algorithms = [algo for algo in speed_priority if algo in base_algorithms] + \
                             [algo for algo in base_algorithms if algo not in speed_priority]
        
        elif project_def.objective == Objective.INTERPRETABILITY:
            # Prioritize interpretable algorithms
            interpretable = ["linear_regression", "logistic_regression", "decision_tree"]
            base_algorithms = [algo for algo in interpretable if algo in base_algorithms] + \
                             [algo for algo in base_algorithms if algo not in interpretable]
        
        # Limit to top 5 algorithms
        return base_algorithms[:5]
    
    def _calculate_performance_targets(self, project_def: ProjectDefinition) -> Dict[str, Any]:
        """Calculate performance targets based on objectives and constraints."""
        
        targets = {}
        
        # Accuracy target
        if project_def.technical_constraints.min_accuracy:
            targets["accuracy"] = project_def.technical_constraints.min_accuracy
        else:
            # Default accuracy targets by domain
            domain_defaults = {
                Domain.FINANCE: 0.85,
                Domain.HEALTHCARE: 0.90,
                Domain.MANUFACTURING: 0.88,
                Domain.RETAIL: 0.75,
                Domain.MARKETING: 0.70
            }
            targets["accuracy"] = domain_defaults.get(project_def.domain, 0.80)
        
        # Latency target
        if project_def.technical_constraints.max_latency_ms:
            targets["latency_ms"] = project_def.technical_constraints.max_latency_ms
        elif project_def.objective == Objective.SPEED:
            # Aggressive latency targets for speed-focused projects
            targets["latency_ms"] = 50
        else:
            # Domain-typical latency targets
            domain_latency = {
                Domain.FINANCE: 100,
                Domain.RETAIL: 200,
                Domain.MANUFACTURING: 500,
                Domain.HEALTHCARE: 1000
            }
            targets["latency_ms"] = domain_latency.get(project_def.domain, 300)
        
        # Throughput target (requests per second)
        if project_def.objective == Objective.SCALABILITY:
            targets["throughput"] = 1000  # High throughput target
        else:
            targets["throughput"] = 100   # Standard throughput
        
        return targets
    
    def _determine_resource_allocation(self, project_def: ProjectDefinition) -> Dict[str, Any]:
        """Determine resource allocation based on project characteristics."""
        
        resource_requirements = project_def.get_resource_requirements()
        
        # Training time allocation
        training_hours = project_def.technical_constraints.max_training_hours
        if not training_hours:
            if project_def.priority == Priority.CRITICAL:
                training_hours = 48.0  # 2 days max
            elif project_def.priority == Priority.HIGH:
                training_hours = 24.0  # 1 day max
            else:
                training_hours = 8.0   # 8 hours max
        
        # Memory allocation
        memory_gb = project_def.technical_constraints.max_memory_gb
        if not memory_gb:
            if project_def.objective == Objective.SCALABILITY:
                memory_gb = 64.0
            elif project_def.domain in [Domain.MANUFACTURING, Domain.HEALTHCARE]:
                memory_gb = 32.0  # Large datasets expected
            else:
                memory_gb = 16.0  # Standard allocation
        
        # Budget priority
        if project_def.objective == Objective.COST_EFFICIENCY:
            budget_priority = "cost_optimized"
        elif project_def.objective in [Objective.ACCURACY, Objective.INNOVATION]:
            budget_priority = "performance_optimized"
        else:
            budget_priority = "balanced"
        
        return {
            "training_hours": training_hours,
            "memory_gb": memory_gb,
            "budget_priority": budget_priority
        }
    
    def _configure_feature_engineering(self, project_def: ProjectDefinition) -> Dict[str, str]:
        """Configure feature engineering strategy."""
        
        if project_def.objective == Objective.INNOVATION:
            complexity = "automated"
            selection = "advanced"
        elif project_def.objective == Objective.SPEED:
            complexity = "basic"
            selection = "fast"
        elif project_def.regulatory_constraints.interpretability_required:
            complexity = "basic"
            selection = "interpretable"
        else:
            complexity = "advanced"
            selection = "correlation_based"
        
        return {
            "complexity": complexity,
            "selection_strategy": selection
        }
    
    def _configure_validation_strategy(self, project_def: ProjectDefinition) -> Dict[str, Any]:
        """Configure validation strategy based on risk and domain."""
        
        if project_def.risk_level == RiskLevel.CRITICAL:
            rigor = "regulatory"
            cv_folds = 10
            test_percentage = 0.3
        elif project_def.risk_level == RiskLevel.HIGH:
            rigor = "extensive"
            cv_folds = 5
            test_percentage = 0.25
        else:
            rigor = "standard"
            cv_folds = 5
            test_percentage = 0.2
        
        # Domain adjustments
        if project_def.domain in [Domain.FINANCE, Domain.HEALTHCARE]:
            rigor = "regulatory"
            test_percentage = max(test_percentage, 0.25)
        
        return {
            "rigor": rigor,
            "cv_folds": cv_folds,
            "test_percentage": test_percentage
        }
    
    def _configure_deployment_strategy(self, project_def: ProjectDefinition) -> Dict[str, Any]:
        """Configure deployment strategy."""
        
        if project_def.objective == Objective.SPEED:
            strategy = "real_time"
            monitoring = "comprehensive"
        elif project_def.domain == Domain.MANUFACTURING:
            strategy = "hybrid"
            monitoring = "comprehensive"
        else:
            strategy = "batch"
            monitoring = "basic" if project_def.risk_level == RiskLevel.LOW else "comprehensive"
        
        # Scaling requirements
        scaling = {
            "auto_scaling": project_def.objective == Objective.SCALABILITY,
            "max_instances": 10 if project_def.priority == Priority.CRITICAL else 5,
            "geographic_distribution": len(project_def.regulatory_constraints.geographic_restrictions) == 0
        }
        
        return {
            "strategy": strategy,
            "scaling": scaling,
            "monitoring": monitoring
        }
    
    def _configure_interpretability(self, project_def: ProjectDefinition) -> Dict[str, Any]:
        """Configure interpretability requirements."""
        
        if project_def.regulatory_constraints.interpretability_required:
            level = "full"
            methods = ["shap", "lime", "feature_importance", "decision_rules"]
        elif project_def.domain in [Domain.FINANCE, Domain.HEALTHCARE]:
            level = "basic"
            methods = ["feature_importance", "shap"]
        elif project_def.objective == Objective.INTERPRETABILITY:
            level = "full"
            methods = ["shap", "lime", "feature_importance"]
        else:
            level = "none"
            methods = []
        
        return {
            "level": level,
            "methods": methods
        }
    
    def _configure_security_compliance(self, project_def: ProjectDefinition) -> Dict[str, Any]:
        """Configure security and compliance requirements."""
        
        if project_def.risk_level == RiskLevel.CRITICAL:
            security_level = "maximum"
        elif project_def.domain in [Domain.FINANCE, Domain.HEALTHCARE]:
            security_level = "enhanced"
        else:
            security_level = "standard"
        
        # Audit requirements
        audit_reqs = []
        if project_def.regulatory_constraints.audit_trail_required:
            audit_reqs.extend(["model_decisions", "data_access", "performance_monitoring"])
        if project_def.domain == Domain.FINANCE:
            audit_reqs.extend(["bias_testing", "model_validation"])
        if project_def.domain == Domain.HEALTHCARE:
            audit_reqs.extend(["clinical_validation", "patient_safety"])
        
        # Data governance
        governance = {
            "data_classification": "sensitive" if project_def.regulatory_constraints.privacy_level in ["high", "maximum"] else "internal",
            "retention_policy": project_def.regulatory_constraints.data_retention_requirements,
            "anonymization_required": project_def.regulatory_constraints.anonymization_required,
            "encryption_required": project_def.regulatory_constraints.encryption_required or security_level == "maximum"
        }
        
        return {
            "level": security_level,
            "audit": list(set(audit_reqs)),  # Remove duplicates
            "governance": governance
        }
    
    def _should_use_ensemble(self, project_def: ProjectDefinition) -> bool:
        """Determine if ensemble methods should be used."""
        
        # Use ensemble for high-accuracy requirements
        if project_def.objective == Objective.ACCURACY:
            return True
        
        # Use ensemble for critical applications
        if project_def.risk_level == RiskLevel.CRITICAL:
            return True
        
        # Avoid ensemble if speed is critical
        if project_def.objective == Objective.SPEED:
            return False
        
        # Avoid ensemble if interpretability is required
        if project_def.regulatory_constraints.interpretability_required:
            return False
        
        # Default based on priority
        return project_def.priority in [Priority.HIGH, Priority.CRITICAL]
    
    def _get_validation_metrics(self, project_def: ProjectDefinition) -> List[str]:
        """Get validation metrics based on objectives."""
        
        metrics = project_def.objective.get_success_metrics()
        
        # Add secondary objective metrics
        for secondary in project_def.secondary_objectives:
            metrics.extend(secondary.get_success_metrics())
        
        # Add domain-specific metrics
        if project_def.domain == Domain.FINANCE:
            metrics.extend(["auc_roc", "precision_recall"])
        elif project_def.domain == Domain.HEALTHCARE:
            metrics.extend(["sensitivity", "specificity"])
        
        # Remove duplicates and limit to top 10
        return list(set(metrics))[:10]
    
    def _initialize_domain_expertise(self) -> Dict[Domain, Dict[str, Any]]:
        """Initialize domain-specific expertise mappings."""
        
        return {
            Domain.FINANCE: {
                "preferred_algorithms": ["xgboost", "random_forest", "logistic_regression"],
                "typical_constraints": {"interpretability_required": True, "max_latency_ms": 100},
                "regulatory_focus": ["bias_testing", "model_explainability", "audit_trail"]
            },
            Domain.HEALTHCARE: {
                "preferred_algorithms": ["random_forest", "gradient_boosting", "svm"],
                "typical_constraints": {"interpretability_required": True, "privacy_level": "maximum"},
                "regulatory_focus": ["clinical_validation", "patient_safety", "data_privacy"]
            },
            Domain.RETAIL: {
                "preferred_algorithms": ["collaborative_filtering", "deep_learning", "xgboost"],
                "typical_constraints": {"max_latency_ms": 200, "scalability_required": True},
                "business_focus": ["customer_experience", "conversion_optimization"]
            },
            Domain.MANUFACTURING: {
                "preferred_algorithms": ["time_series", "anomaly_detection", "neural_network"],
                "typical_constraints": {"real_time_processing": True, "reliability_critical": True},
                "operational_focus": ["predictive_maintenance", "quality_control"]
            }
        }
    
    def _initialize_objective_mappings(self) -> Dict[Objective, Dict[str, Any]]:
        """Initialize objective-specific technical mappings."""
        
        return {
            Objective.ACCURACY: {
                "priority_metrics": ["accuracy", "precision", "recall", "f1_score"],
                "algorithm_preferences": ["ensemble", "deep_learning", "xgboost"],
                "validation_emphasis": "extensive_testing"
            },
            Objective.SPEED: {
                "priority_metrics": ["inference_latency", "training_time", "throughput"],
                "algorithm_preferences": ["linear_models", "tree_based", "simple_nn"],
                "optimization_focus": "latency_optimization"
            },
            Objective.INTERPRETABILITY: {
                "priority_metrics": ["explanation_quality", "feature_importance"],
                "algorithm_preferences": ["linear_regression", "decision_tree", "rule_based"],
                "interpretability_methods": ["shap", "lime", "feature_importance"]
            },
            Objective.FAIRNESS: {
                "priority_metrics": ["demographic_parity", "equalized_odds", "calibration"],
                "algorithm_preferences": ["fair_classification", "bias_corrected"],
                "fairness_constraints": ["protected_attributes", "bias_mitigation"]
            },
            Objective.COMPLIANCE: {
                "priority_metrics": ["regulatory_adherence", "audit_completeness"],
                "algorithm_preferences": ["interpretable_models", "auditable_systems"],
                "compliance_requirements": ["audit_trail", "model_documentation"]
            }
        }
    
    def _initialize_compliance_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize compliance rule mappings."""
        
        return {
            "GDPR": {
                "requirements": ["data_minimization", "right_to_explanation", "consent_management"],
                "technical_implications": ["anonymization", "explainable_ai", "data_governance"]
            },
            "HIPAA": {
                "requirements": ["data_encryption", "access_controls", "audit_logs"],
                "technical_implications": ["secure_processing", "privacy_preserving", "audit_trail"]
            },
            "Basel III": {
                "requirements": ["model_validation", "stress_testing", "documentation"],
                "technical_implications": ["robust_validation", "scenario_testing", "interpretability"]
            }
        }
    
    def _get_historical_insights(self, project_definition: ProjectDefinition, 
                                dataset_characteristics=None) -> Dict[str, Any]:
        """Query historical results from similar projects to inform strategy."""
        
        if not self.enable_historical_learning or not dataset_characteristics:
            return {}
        
        try:
            similar_executions = self.meta_db.find_similar_successful_executions(
                dataset_characteristics,
                project_config=None,
                similarity_threshold=0.7,
                min_success_rating=0.8,
                limit=5
            )
            
            if not similar_executions:
                return {}
            
            successful_strategies = {}
            performance_data = []
            
            for execution in similar_executions:
                strategy = execution.project_config.strategy_applied
                if strategy not in successful_strategies:
                    successful_strategies[strategy] = {
                        'count': 0,
                        'avg_performance': 0,
                        'avg_execution_time': 0,
                        'feature_engineering_success': 0,
                        'algorithms_used': []
                    }
                
                successful_strategies[strategy]['count'] += 1
                performance_data.append(execution.success_rating)
                
                if execution.project_config.feature_engineering_enabled:
                    successful_strategies[strategy]['feature_engineering_success'] += 1
            
            for strategy_data in successful_strategies.values():
                if strategy_data['count'] > 0:
                    strategy_data['feature_engineering_success'] /= strategy_data['count']
            
            return {
                'similar_projects_found': len(similar_executions),
                'successful_strategies': successful_strategies,
                'historical_avg_performance': sum(performance_data) / len(performance_data),
                'recommended_approach': max(successful_strategies.keys(), 
                                          key=lambda k: successful_strategies[k]['count']) if successful_strategies else None
            }
            
        except Exception:
            return {}
    
    def _determine_model_complexity(self, project_def: ProjectDefinition, 
                                   historical_insights: Dict[str, Any] = None) -> ModelComplexity:
        """Determine model complexity enhanced with historical insights."""
        
        complexity_score = 0
        
        if project_def.objective in [Objective.ACCURACY, Objective.INNOVATION]:
            complexity_score += 2
        elif project_def.objective in [Objective.SPEED, Objective.COST_EFFICIENCY]:
            complexity_score -= 1
        
        if project_def.regulatory_constraints.interpretability_required:
            complexity_score -= 2
        
        if project_def.domain in [Domain.FINANCE, Domain.HEALTHCARE]:
            complexity_score += 1
        
        if project_def.risk_level == RiskLevel.CRITICAL:
            if self.translation_strategy == TranslationStrategy.CONSERVATIVE:
                complexity_score -= 1
        
        if self.translation_strategy == TranslationStrategy.AGGRESSIVE:
            complexity_score += 2
        elif self.translation_strategy == TranslationStrategy.CONSERVATIVE:
            complexity_score -= 1
        
        if historical_insights and historical_insights.get('similar_projects_found', 0) > 0:
            historical_performance = historical_insights.get('historical_avg_performance', 0)
            if historical_performance > 0.9:
                complexity_score += 1
            elif historical_performance < 0.7:
                complexity_score -= 1
        
        if complexity_score <= -1:
            return ModelComplexity.SIMPLE
        elif complexity_score <= 2:
            return ModelComplexity.MODERATE
        elif complexity_score <= 4:
            return ModelComplexity.COMPLEX
        else:
            return ModelComplexity.EXPERIMENTAL
    
    def _select_algorithms(self, project_def: ProjectDefinition, complexity: ModelComplexity,
                          historical_insights: Dict[str, Any] = None) -> List[str]:
        """Select algorithms enhanced with historical success patterns."""
        
        algorithm_map = {
            ModelComplexity.SIMPLE: ["linear_regression", "logistic_regression", "decision_tree"],
            ModelComplexity.MODERATE: ["random_forest", "gradient_boosting", "svm", "naive_bayes"],
            ModelComplexity.COMPLEX: ["xgboost", "lightgbm", "neural_network", "deep_learning"],
            ModelComplexity.EXPERIMENTAL: ["transformer", "automl", "neural_architecture_search"]
        }
        
        base_algorithms = algorithm_map[complexity].copy()
        
        if historical_insights and historical_insights.get('similar_projects_found', 0) > 0:
            successful_strategies = historical_insights.get('successful_strategies', {})
            if successful_strategies:
                historically_successful = []
                for strategy_data in successful_strategies.values():
                    historically_successful.extend(strategy_data.get('algorithms_used', []))
                
                proven_algorithms = [algo for algo in historically_successful if algo in base_algorithms]
                other_algorithms = [algo for algo in base_algorithms if algo not in proven_algorithms]
                base_algorithms = proven_algorithms + other_algorithms
        
        domain_preferences = self.domain_expertise.get(project_def.domain, {})
        if "preferred_algorithms" in domain_preferences:
            domain_algos = [algo for algo in domain_preferences["preferred_algorithms"] if algo in base_algorithms]
            other_algos = [algo for algo in base_algorithms if algo not in domain_algos]
            base_algorithms = domain_algos + other_algos
        
        if project_def.objective == Objective.SPEED:
            speed_priority = ["linear_regression", "logistic_regression", "naive_bayes", "decision_tree"]
            base_algorithms = [algo for algo in speed_priority if algo in base_algorithms] + \
                             [algo for algo in base_algorithms if algo not in speed_priority]
        
        elif project_def.objective == Objective.INTERPRETABILITY:
            interpretable = ["linear_regression", "logistic_regression", "decision_tree"]
            base_algorithms = [algo for algo in interpretable if algo in base_algorithms] + \
                             [algo for algo in base_algorithms if algo not in interpretable]
        
        return base_algorithms[:5]

    def _get_recommended_strategies(self, project_definition: ProjectDefinition, 
                                  dataset_characteristics=None) -> Dict[str, Any]:
        """Get strategy recommendations from database experience"""
        if not self.enable_historical_learning or not dataset_characteristics:
            return {}
        
        try:
            if hasattr(dataset_characteristics, 'dataset_hash'):
                dataset_chars = dataset_characteristics
            else:
                dataset_chars = create_dataset_characteristics(
                    dataset_characteristics, 
                    domain=project_definition.domain.value
                )
            
            recommendations = self.meta_db.get_recommended_strategies(
                dataset_chars,
                objective=project_definition.objective.value,
                domain=project_definition.domain.value
            )
            
            if recommendations.get('recommended_strategies'):
                self.logger.info(f"Found {len(recommendations['recommended_strategies'])} strategy recommendations from experience")
                return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting strategy recommendations: {e}")
        
        return {}

    def _apply_experience_based_adjustments(self, tech_config: TechnicalConfiguration,
                                           recommended_strategies: Dict[str, Any]) -> TechnicalConfiguration:
        """Apply experience-based adjustments to technical configuration"""
        if not recommended_strategies.get('recommended_strategies'):
            return tech_config
            
        top_strategy = recommended_strategies['recommended_strategies'][0]
        strategy_config = top_strategy.get('config', {})
        confidence = top_strategy.get('confidence', 0.0)
        
        if confidence > 0.8:
            if 'feature_engineering_enabled' in strategy_config:
                tech_config.feature_engineering_complexity = "automated" if strategy_config['feature_engineering_enabled'] else "basic"
            if 'feature_selection_enabled' in strategy_config:
                tech_config.feature_selection_strategy = "advanced" if strategy_config['feature_selection_enabled'] else "basic"
            if 'security_level' in strategy_config:
                tech_config.security_level = strategy_config['security_level']
        
        return tech_config