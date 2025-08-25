"""
LLM Interface Module for DataInsight AI
Provides natural language processing capabilities for conversational AI interaction
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import openai
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class LLMInterface:
    """
    LLM Interface for intent understanding and response generation
    Supports multiple providers (OpenAI, Anthropic) with fallback capability
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.providers = {}
        self.embedding_model = None
        self.conversation_history = []
        
        self._initialize_providers()
        self._initialize_embeddings()
        
    def _initialize_providers(self):
        """Initialize available LLM providers"""
        
        # OpenAI
        openai_key = os.getenv('OPENAI_API_KEY') or self.config.get('openai_api_key')
        if openai_key:
            self.providers['openai'] = openai.OpenAI(api_key=openai_key)
            logger.info("OpenAI provider initialized")
        
        # Anthropic
        anthropic_key = os.getenv('ANTHROPIC_API_KEY') or self.config.get('anthropic_api_key')
        if anthropic_key:
            self.providers['anthropic'] = Anthropic(api_key=anthropic_key)
            logger.info("Anthropic provider initialized")
            
        if not self.providers:
            logger.warning("No LLM providers configured. Using mock responses.")
            
    def _initialize_embeddings(self):
        """Initialize sentence embedding model for semantic similarity"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model initialized")
        except Exception as e:
            logger.warning(f"Could not initialize embedding model: {e}")
            
    def extract_intent(self, user_message: str, context: Dict = None) -> Dict[str, Any]:
        """
        Extract user intent from natural language input with intelligent dataset analysis
        """
        
        dataset_analysis = self._analyze_dataset_context(context) if context else {}
        
        prompt = self._build_intent_extraction_prompt(user_message, context, dataset_analysis)
        
        try:
            response = self._call_llm(
                prompt=prompt,
                max_tokens=800,
                temperature=0.1,
                system_message="You are an expert data scientist who can automatically identify target variables and understand complex data analysis intents from natural language."
            )
            
            intent = self._parse_intent_response(response)
            intent['original_message'] = user_message
            intent['timestamp'] = datetime.now().isoformat()
            intent['dataset_analysis'] = dataset_analysis
            
            if dataset_analysis and intent.get('task_type') in ['classification', 'regression', 'timeseries', 'nlp']:
                intent = self._enhance_intent_with_target_detection(intent, dataset_analysis, user_message)
            
            return intent
            
        except Exception as e:
            logger.error(f"Intent extraction failed: {e}")
            return {
                'task_type': 'unknown',
                'confidence': 0.0,
                'parameters': {},
                'clarification_needed': True,
                'error': str(e)
            }
    
    def convert_intent_to_strategy(self, intent: Dict[str, Any], data_context: Dict = None) -> Dict[str, Any]:
        """
        Convert extracted intent to executable strategy configuration
        """
        
        prompt = self._build_strategy_conversion_prompt(intent, data_context)
        
        try:
            response = self._call_llm(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.2,
                system_message="You are an ML pipeline architect. Convert user intents into specific ML strategy configurations."
            )
            
            strategy = self._parse_strategy_response(response)
            strategy['source_intent'] = intent
            strategy['timestamp'] = datetime.now().isoformat()
            
            return strategy
            
        except Exception as e:
            logger.error(f"Strategy conversion failed: {e}")
            return {
                'task_type': intent.get('task_type', 'classification'),
                'target_column': None,
                'features': [],
                'configuration': {},
                'error': str(e)
            }
    
    def generate_follow_up_questions(self, intent: Dict, data_summary: Dict = None) -> List[str]:
        """
        Generate intelligent follow-up questions based on intent and data context
        """
        
        prompt = self._build_followup_prompt(intent, data_summary)
        
        try:
            response = self._call_llm(
                prompt=prompt,
                max_tokens=300,
                temperature=0.3,
                system_message="Generate helpful follow-up questions to clarify user intent for data analysis."
            )
            
            questions = self._parse_questions_response(response)
            return questions[:3]  # Return top 3 questions
            
        except Exception as e:
            logger.error(f"Follow-up question generation failed: {e}")
            return [
                "What specific outcome are you trying to predict or analyze?",
                "Do you have any particular columns in mind as the target variable?",
                "Are there any specific requirements or constraints for this analysis?"
            ]
    
    def generate_explanation(self, results: Dict, user_question: str = None) -> str:
        """
        Generate natural language explanation of results
        """
        
        prompt = self._build_explanation_prompt(results, user_question)
        
        try:
            response = self._call_llm(
                prompt=prompt,
                max_tokens=800,
                temperature=0.4,
                system_message="You are an expert data scientist. Explain ML results in clear, accessible language."
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return "I encountered an error while generating the explanation. Please try again."
    
    def _call_llm(self, prompt: str, max_tokens: int = 500, temperature: float = 0.3, 
                  system_message: str = None) -> str:
        """
        Call LLM with fallback between providers
        """
        
        # Try OpenAI first
        if 'openai' in self.providers:
            try:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})
                
                response = self.providers['openai'].chat.completions.create(
                    model="gpt-4o-mini",  # Cost-effective model
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"OpenAI call failed: {e}, trying Anthropic...")
        
        # Try Anthropic as fallback
        if 'anthropic' in self.providers:
            try:
                full_prompt = prompt
                if system_message:
                    full_prompt = f"{system_message}\n\n{prompt}"
                
                response = self.providers['anthropic'].messages.create(
                    model="claude-3-haiku-20240307",  # Fast, cost-effective model
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                
                return response.content[0].text
                
            except Exception as e:
                logger.warning(f"Anthropic call failed: {e}")
        
        # Mock response if no providers available
        return self._generate_mock_response(prompt)
    
    def _analyze_dataset_context(self, context: Dict) -> Dict[str, Any]:
        """Analyze dataset context to understand column types and potential targets"""
        
        if not context or 'columns' not in context:
            return {}
        
        columns = context.get('columns', [])
        dtypes = context.get('dtypes', {})
        sample_data = context.get('sample_data', {})
        
        analysis = {
            'total_columns': len(columns),
            'column_analysis': {},
            'potential_targets': [],
            'categorical_features': [],
            'numerical_features': [],
            'text_features': [],
            'datetime_features': []
        }
        
        for col in columns:
            col_dtype = str(dtypes.get(col, 'object'))
            col_analysis = {
                'dtype': col_dtype,
                'is_categorical': False,
                'is_numerical': False,
                'is_target_candidate': False,
                'unique_ratio_estimate': 0.0
            }
            
            # Analyze by column name patterns
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['target', 'label', 'y', 'class', 'category', 'outcome', 'result']):
                col_analysis['is_target_candidate'] = True
                analysis['potential_targets'].append(col)
            
            if any(keyword in col_lower for keyword in ['churn', 'fraud', 'spam', 'sentiment', 'rating', 'status', 'type']):
                col_analysis['is_target_candidate'] = True
                analysis['potential_targets'].append(col)
                
            # Analyze by data type
            if 'int' in col_dtype or 'float' in col_dtype:
                col_analysis['is_numerical'] = True
                analysis['numerical_features'].append(col)
            elif 'object' in col_dtype or 'string' in col_dtype:
                if sample_data and col in sample_data:
                    sample_values = list(sample_data[col].values())
                    if any(isinstance(v, str) and len(str(v)) > 50 for v in sample_values):
                        analysis['text_features'].append(col)
                    else:
                        col_analysis['is_categorical'] = True
                        analysis['categorical_features'].append(col)
                        
                        # Check if binary categorical (potential target)
                        unique_values = set(str(v) for v in sample_values if v is not None)
                        if len(unique_values) == 2:
                            col_analysis['is_target_candidate'] = True
                            if col not in analysis['potential_targets']:
                                analysis['potential_targets'].append(col)
                                
            elif 'datetime' in col_dtype:
                analysis['datetime_features'].append(col)
            
            analysis['column_analysis'][col] = col_analysis
        
        return analysis
    
    def _enhance_intent_with_target_detection(self, intent: Dict, dataset_analysis: Dict, user_message: str) -> Dict[str, Any]:
        """Enhance intent with automatic target column detection"""
        
        potential_targets = dataset_analysis.get('potential_targets', [])
        
        if not potential_targets:
            return intent
        
        # Find best target match based on user message and column analysis
        best_target = self._find_best_target_match(user_message, potential_targets, dataset_analysis)
        
        if best_target:
            intent['target_variable'] = best_target
            intent['confidence'] = min(1.0, intent.get('confidence', 0.7) + 0.2)
            intent['clarification_needed'] = False
            intent['auto_detected_target'] = True
            
            # Remove target detection from clarifications
            clarifications = intent.get('clarifications', [])
            intent['clarifications'] = [c for c in clarifications if 'target' not in c.lower()]
        
        return intent
    
    def _find_best_target_match(self, user_message: str, potential_targets: List[str], dataset_analysis: Dict) -> str:
        """Find best target column match based on user intent and column characteristics"""
        
        if len(potential_targets) == 1:
            return potential_targets[0]
        
        message_lower = user_message.lower()
        
        # Score each potential target
        scored_targets = []
        for target in potential_targets:
            score = 0.0
            target_lower = target.lower()
            
            # Keyword matching
            if any(keyword in target_lower for keyword in ['churn', 'fraud', 'spam'] if keyword in message_lower):
                score += 0.5
            if any(keyword in target_lower for keyword in ['target', 'label', 'outcome', 'result']):
                score += 0.3
            if 'predict' in message_lower and any(word in target_lower for word in message_lower.split()):
                score += 0.4
                
            # Binary targets often good for classification
            col_analysis = dataset_analysis.get('column_analysis', {}).get(target, {})
            if col_analysis.get('is_categorical') and col_analysis.get('unique_ratio_estimate', 0) < 0.1:
                score += 0.2
            
            scored_targets.append((target, score))
        
        # Return highest scoring target if significantly better
        scored_targets.sort(key=lambda x: x[1], reverse=True)
        if scored_targets and scored_targets[0][1] > 0.3:
            return scored_targets[0][0]
        
        # Default to first potential target
        return potential_targets[0] if potential_targets else None

    def _build_intent_extraction_prompt(self, message: str, context: Dict = None, dataset_analysis: Dict = None) -> str:
        """Build prompt for intent extraction with dataset context"""
        
        context_str = ""
        if context:
            context_str = f"\nData context: {json.dumps(context, indent=2)}"
        
        dataset_str = ""
        if dataset_analysis:
            dataset_str = f"\nDataset analysis: {json.dumps(dataset_analysis, indent=2)}"
        
        return f"""
Analyze this user message and extract the ML/data analysis intent. Use the dataset context to automatically identify the most appropriate target column.

User message: "{message}"{context_str}{dataset_str}

Return a JSON response with this structure:
{{
    "task_type": "classification|regression|clustering|timeseries|nlp|eda|supervised|unsupervised",
    "confidence": 0.0-1.0,
    "target_variable": "column_name or null",
    "features_mentioned": ["list", "of", "columns"],
    "parameters": {{
        "metric": "accuracy|rmse|f1|auc|etc",
        "method": "specific_algorithm",
        "constraints": ["any", "constraints"]
    }},
    "clarification_needed": true/false,
    "clarifications": ["what needs clarification"],
    "reasoning": "why this target was selected"
}}

IMPORTANT: 
- Analyze the dataset_analysis to automatically identify the best target column
- For classification: look for categorical columns, especially binary ones, or columns with names like "churn", "fraud", "category", "class"
- For regression: look for numerical columns that could be predicted like "price", "revenue", "score", "rating"
- For clustering: set target_variable to null
- For timeseries: look for datetime columns and value columns to predict
- For NLP: look for text columns and classification/sentiment targets
- Always try to automatically determine the target rather than asking the user
"""
    
    def _build_strategy_conversion_prompt(self, intent: Dict, data_context: Dict = None) -> str:
        """Build prompt for strategy conversion"""
        
        data_str = ""
        if data_context:
            data_str = f"\nData context:\n{json.dumps(data_context, indent=2)}"
        
        return f"""
Convert this user intent into a specific ML pipeline strategy.

Intent: {json.dumps(intent, indent=2)}{data_str}

Return a JSON configuration with this structure:
{{
    "task_type": "classification|regression|clustering|timeseries|nlp",
    "target_column": "column_name",
    "feature_columns": ["list", "of", "feature", "columns"],
    "configuration": {{
        "enable_feature_generation": true/false,
        "enable_feature_selection": true/false,
        "enable_intelligence": true/false,
        "algorithms": ["list", "of", "preferred", "algorithms"],
        "validation_strategy": "cross_validation|holdout|time_series",
        "metrics": ["primary", "secondary", "metrics"]
    }},
    "preprocessing": {{
        "handle_missing": "drop|impute|advanced",
        "encoding": "automatic|manual",
        "scaling": "standard|robust|none"
    }}
}}

Generate a complete, executable strategy based on the user's intent.
"""
    
    def _build_followup_prompt(self, intent: Dict, data_summary: Dict = None) -> str:
        """Build prompt for follow-up questions"""
        
        data_str = ""
        if data_summary:
            data_str = f"\nData summary:\n{json.dumps(data_summary, indent=2)}"
        
        return f"""
Generate 2-3 intelligent follow-up questions to clarify the user's intent.

Current intent: {json.dumps(intent, indent=2)}{data_str}

Return a JSON list of questions:
["Question 1?", "Question 2?", "Question 3?"]

Questions should:
1. Address unclear aspects of the intent
2. Help specify target variables or features
3. Clarify constraints or requirements
4. Be specific to the data context when available

Focus on practical, actionable clarifications.
"""
    
    def _build_explanation_prompt(self, results: Dict, user_question: str = None) -> str:
        """Build prompt for result explanation"""
        
        question_context = ""
        if user_question:
            question_context = f"\nUser's specific question: {user_question}"
        
        return f"""
Explain these ML results in clear, accessible language.

Results: {json.dumps(results, indent=2)}{question_context}

Provide a natural language explanation that:
1. Summarizes the key findings
2. Explains what the metrics mean in practical terms  
3. Highlights important insights or patterns
4. Suggests next steps or actions if relevant
5. Uses business-friendly language, avoiding technical jargon

Keep the explanation concise but comprehensive.
"""
    
    def _parse_intent_response(self, response: str) -> Dict[str, Any]:
        """Parse structured intent response"""
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                # Fallback parsing
                return self._extract_fallback_intent(response)
        except:
            return self._extract_fallback_intent(response)
    
    def _parse_strategy_response(self, response: str) -> Dict[str, Any]:
        """Parse structured strategy response"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                return self._extract_fallback_strategy(response)
        except:
            return self._extract_fallback_strategy(response)
    
    def _parse_questions_response(self, response: str) -> List[str]:
        """Parse follow-up questions response"""
        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                # Extract questions from text
                lines = response.strip().split('\n')
                questions = []
                for line in lines:
                    if '?' in line:
                        # Clean up the question
                        question = line.strip()
                        if question.startswith(('-', '*', '•')):
                            question = question[1:].strip()
                        if question:
                            questions.append(question)
                return questions[:3]
        except:
            return ["Could you provide more details about your analysis goals?"]
    
    def _extract_fallback_intent(self, response: str) -> Dict[str, Any]:
        """Extract intent using keyword matching as fallback"""
        response_lower = response.lower()
        
        # Determine task type
        task_type = "classification"
        if any(word in response_lower for word in ["predict", "regression", "continuous"]):
            task_type = "regression"
        elif any(word in response_lower for word in ["cluster", "group", "segment"]):
            task_type = "clustering"
        elif any(word in response_lower for word in ["time", "forecast", "trend"]):
            task_type = "timeseries"
        elif any(word in response_lower for word in ["text", "nlp", "sentiment"]):
            task_type = "nlp"
        elif any(word in response_lower for word in ["explore", "analyze", "understand"]):
            task_type = "eda"
        
        return {
            "task_type": task_type,
            "confidence": 0.6,
            "target_variable": None,
            "features_mentioned": [],
            "parameters": {},
            "clarification_needed": True,
            "clarifications": ["Please specify your target variable"]
        }
    
    def _extract_fallback_strategy(self, response: str) -> Dict[str, Any]:
        """Extract strategy using defaults as fallback"""
        return {
            "task_type": "classification",
            "target_column": None,
            "feature_columns": [],
            "configuration": {
                "enable_feature_generation": True,
                "enable_feature_selection": True,
                "enable_intelligence": True,
                "algorithms": ["random_forest", "logistic_regression"],
                "validation_strategy": "cross_validation",
                "metrics": ["accuracy", "f1_score"]
            },
            "preprocessing": {
                "handle_missing": "impute",
                "encoding": "automatic",
                "scaling": "standard"
            }
        }
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock response when no providers available"""
        if "intent" in prompt.lower():
            return '''
            {
                "task_type": "classification",
                "confidence": 0.7,
                "target_variable": null,
                "features_mentioned": [],
                "parameters": {},
                "clarification_needed": true,
                "clarifications": ["Please specify your target variable"]
            }
            '''
        elif "strategy" in prompt.lower():
            return '''
            {
                "task_type": "classification",
                "target_column": null,
                "feature_columns": [],
                "configuration": {
                    "enable_feature_generation": true,
                    "enable_feature_selection": true,
                    "enable_intelligence": true
                }
            }
            '''
        elif "question" in prompt.lower():
            return '["What specific outcome would you like to predict?", "Which column should be the target variable?"]'
        else:
            return "I understand you want to analyze your data. Could you provide more specific details about your goals?"
    
    def get_conversation_context(self) -> Dict[str, Any]:
        """Get current conversation context for RAG"""
        return {
            'history': self.conversation_history[-5:],  # Last 5 messages
            'timestamp': datetime.now().isoformat()
        }
    
    def add_to_conversation(self, role: str, content: str, metadata: Dict = None):
        """Add message to conversation history"""
        self.conversation_history.append({
            'role': role,
            'content': content,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 20 messages to manage memory
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]