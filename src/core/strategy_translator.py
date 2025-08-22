from typing import Dict, Any, List
from .project_definition import ProjectDefinition, Objective, Domain, RiskLevel
from .pipeline_orchestrator import PipelineConfig

class StrategyTranslator:
    
    def __init__(self):
        self.strategy_rules = self._initialize_strategy_rules()
    
    def translate(self, project_def: ProjectDefinition) -> PipelineConfig:
        base_config = self._get_base_config(project_def)
        
        objective_config = self._apply_objective_strategy(project_def.objective, base_config)
        domain_config = self._apply_domain_strategy(project_def.domain, objective_config)
        risk_config = self._apply_risk_strategy(project_def.risk_level, domain_config)
        final_config = self._apply_constraints(project_def.constraints, risk_config)
        
        return final_config
    
    def _get_base_config(self, project_def: ProjectDefinition) -> PipelineConfig:
        return PipelineConfig(
            enable_explainability=True,
            enable_bias_detection=len(project_def.constraints.protected_attributes) > 0,
            enable_security=project_def.domain in [Domain.FINANCE, Domain.HEALTHCARE],
            enable_privacy_protection=project_def.domain in [Domain.FINANCE, Domain.HEALTHCARE],
            enable_compliance=len(project_def.constraints.compliance_rules) > 0,
            enable_adaptive_learning=True,
            enable_feature_engineering=True,
            enable_feature_selection=True
        )
    
    def _apply_objective_strategy(self, objective: Objective, config: PipelineConfig) -> PipelineConfig:
        if objective == Objective.ACCURACY:
            config.enable_feature_engineering = True
            config.enable_feature_selection = True
            config.max_workers = 8
            
        elif objective == Objective.SPEED:
            config.enable_feature_engineering = False
            config.enable_feature_selection = True
            config.max_workers = 4
            config.enable_explainability = False
            
        elif objective == Objective.INTERPRETABILITY:
            config.enable_feature_engineering = False
            config.enable_explainability = True
            config.enable_bias_detection = True
            
        elif objective == Objective.FAIRNESS:
            config.enable_bias_detection = True
            config.enable_explainability = True
            config.sensitive_attributes = getattr(config, 'sensitive_attributes', [])
            
        elif objective == Objective.COST:
            config.enable_feature_engineering = False
            config.max_workers = 2
            config.enable_adaptive_learning = False
            
        elif objective == Objective.ROBUSTNESS:
            config.enable_feature_engineering = True
            config.enable_adaptive_learning = True
            config.validation_threshold = 0.95
            
        elif objective == Objective.COMPLIANCE:
            config.enable_security = True
            config.enable_privacy_protection = True
            config.enable_compliance = True
            config.enable_bias_detection = True
            config.enable_explainability = True
        
        return config
    
    def _apply_domain_strategy(self, domain: Domain, config: PipelineConfig) -> PipelineConfig:
        domain_configs = {
            Domain.FINANCE: {
                'security_level': 'high',
                'privacy_level': 'high',
                'applicable_regulations': ['gdpr', 'pci_dss', 'sox'],
                'enable_security': True,
                'enable_privacy_protection': True,
                'enable_compliance': True
            },
            Domain.HEALTHCARE: {
                'security_level': 'high', 
                'privacy_level': 'high',
                'applicable_regulations': ['hipaa', 'gdpr'],
                'enable_security': True,
                'enable_privacy_protection': True,
                'enable_compliance': True
            },
            Domain.RETAIL: {
                'security_level': 'medium',
                'privacy_level': 'medium',
                'applicable_regulations': ['gdpr', 'ccpa'],
                'enable_feature_engineering': True
            },
            Domain.TECHNOLOGY: {
                'security_level': 'medium',
                'enable_adaptive_learning': True,
                'max_workers': 8
            }
        }
        
        domain_config = domain_configs.get(domain, {})
        for key, value in domain_config.items():
            setattr(config, key, value)
        
        return config
    
    def _apply_risk_strategy(self, risk_level: RiskLevel, config: PipelineConfig) -> PipelineConfig:
        if risk_level == RiskLevel.LOW:
            config.validation_threshold = 0.98
            config.enable_explainability = True
            config.enable_bias_detection = True
            
        elif risk_level == RiskLevel.MEDIUM:
            config.validation_threshold = 0.90
            
        elif risk_level == RiskLevel.HIGH:
            config.validation_threshold = 0.80
            config.enable_adaptive_learning = True
            
        return config
    
    def _apply_constraints(self, constraints, config: PipelineConfig) -> PipelineConfig:
        if constraints.max_latency_ms and constraints.max_latency_ms < 100:
            config.enable_feature_engineering = False
            config.enable_explainability = False
            config.max_workers = 2
            
        if constraints.interpretability_required:
            config.enable_explainability = True
            config.enable_bias_detection = True
            
        if constraints.protected_attributes:
            config.enable_bias_detection = True
            config.sensitive_attributes = constraints.protected_attributes
            
        if constraints.min_accuracy:
            config.validation_threshold = constraints.min_accuracy
            
        if constraints.compliance_rules:
            config.enable_compliance = True
            config.applicable_regulations = constraints.compliance_rules
            
        return config
    
    def _initialize_strategy_rules(self) -> Dict[str, Any]:
        return {
            'model_selection_strategies': {
                Objective.INTERPRETABILITY: ['linear_models', 'tree_models'],
                Objective.SPEED: ['linear_models', 'fast_ensemble'],
                Objective.ACCURACY: ['all_models', 'ensemble_heavy'],
                Objective.FAIRNESS: ['fair_models', 'bias_aware']
            },
            'feature_strategies': {
                Objective.SPEED: 'minimal_features',
                Objective.INTERPRETABILITY: 'interpretable_features',
                Objective.ACCURACY: 'comprehensive_features'
            }
        }
    
    def get_model_strategy(self, objective: Objective) -> List[str]:
        return self.strategy_rules['model_selection_strategies'].get(
            objective, ['balanced_models']
        )