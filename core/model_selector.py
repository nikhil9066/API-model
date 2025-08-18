"""
core/model_selector.py
Smart model selection based on dataset characteristics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass

@dataclass
class ModelSuggestion:
    """Model suggestion with reasoning"""
    model_name: str
    confidence_score: float
    reasoning: str
    expected_performance: str
    training_time: str
    complexity: str

@dataclass
class DatasetCharacteristics:
    """Dataset characteristics for model selection"""
    size_category: str
    feature_category: str
    target_distribution: str
    missing_data_percentage: float
    outlier_percentage: float
    feature_correlation: float
    skewness_level: str
    complexity_score: float

class ModelSelector:
    """Smart model selection based on dataset analysis"""
    
    def __init__(self, config: Dict, job_id: str, progress_tracker=None, state_manager=None):
        self.config = config
        self.job_id = job_id
        self.progress = progress_tracker
        self.state_manager = state_manager
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.modeling_config = config.get('modeling', {})
        self.intelligence_config = config.get('intelligence', {})
        
        # Available models by category
        self.available_models = self.modeling_config.get('available_models', {})
        
    def analyze_dataset_and_suggest_models(self, df: pd.DataFrame, target_variable: str) -> Tuple[DatasetCharacteristics, List[ModelSuggestion]]:
        """Analyze dataset characteristics and suggest best models"""
        
        self.logger.info(f"Analyzing dataset characteristics for model selection")
        
        if self.progress:
            self.progress.start_sub_step("model_selection", "dataset_analysis", "Analyzing dataset characteristics")
        
        # Analyze dataset characteristics
        characteristics = self._analyze_dataset_characteristics(df, target_variable)
        
        if self.progress:
            self.progress.complete_sub_step("model_selection", "dataset_analysis")
            self.progress.start_sub_step("model_selection", "model_suggestions", "Generating model suggestions")
        
        # Generate model suggestions
        suggestions = self._generate_model_suggestions(characteristics)
        
        if self.progress:
            self.progress.complete_sub_step("model_selection", "model_suggestions")
        
        # Update state manager
        if self.state_manager:
            self.state_manager.update_model_suggestions(
                self.job_id,
                [self._suggestion_to_dict(s) for s in suggestions],
                self._characteristics_to_dict(characteristics)
            )
        
        self.logger.info(f"Generated {len(suggestions)} model suggestions")
        return characteristics, suggestions
    
    def _analyze_dataset_characteristics(self, df: pd.DataFrame, target_variable: str) -> DatasetCharacteristics:
        """Analyze dataset to determine characteristics for model selection"""
        
        # Basic dataset info
        n_rows, n_cols = df.shape
        n_features = n_cols - 1  # Exclude target
        
        # Size categorization
        size_categories = self.intelligence_config.get('dataset_profiling', {}).get('size_categories', {
            'small': [0, 1000],
            'medium': [1000, 10000],
            'large': [10000, 100000],
            'xlarge': [100000, 1000000]
        })
        
        size_category = 'xlarge'
        for category, (min_size, max_size) in size_categories.items():
            if min_size <= n_rows < max_size:
                size_category = category
                break
        
        # Feature categorization
        feature_categories = self.intelligence_config.get('dataset_profiling', {}).get('feature_categories', {
            'low': [1, 10],
            'medium': [10, 50],
            'high': [50, 200],
            'very_high': [200, 1000]
        })
        
        feature_category = 'very_high'
        for category, (min_feat, max_feat) in feature_categories.items():
            if min_feat <= n_features < max_feat:
                feature_category = category
                break
        
        # Target distribution analysis
        target_data = df[target_variable]
        target_skewness = abs(target_data.skew())
        target_kurtosis = target_data.kurtosis()
        
        if target_skewness < 0.5 and abs(target_kurtosis) < 3:
            target_distribution = 'normal'
        elif target_skewness >= 2:
            target_distribution = 'highly_skewed'
        elif target_skewness >= 1:
            target_distribution = 'moderately_skewed'
        else:
            target_distribution = 'slightly_skewed'
        
        # Missing data analysis
        missing_percentage = (df.isnull().sum().sum() / df.size) * 100
        
        # Outlier analysis (simple IQR method)
        Q1 = target_data.quantile(0.25)
        Q3 = target_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = target_data[(target_data < lower_bound) | (target_data > upper_bound)]
        outlier_percentage = (len(outliers) / len(target_data)) * 100
        
        # Feature correlation analysis
        numeric_features = df.select_dtypes(include=[np.number]).drop(columns=[target_variable])
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr()
            # Average absolute correlation between features
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            avg_correlation = upper_triangle.stack().abs().mean()
        else:
            avg_correlation = 0.0
        
        # Overall skewness level
        feature_skewness = numeric_features.skew().abs()
        highly_skewed = (feature_skewness > 1).sum()
        moderately_skewed = ((feature_skewness > 0.5) & (feature_skewness <= 1)).sum()
        
        if highly_skewed > len(feature_skewness) * 0.5:
            skewness_level = 'high'
        elif moderately_skewed > len(feature_skewness) * 0.3:
            skewness_level = 'moderate'
        else:
            skewness_level = 'low'
        
        # Complexity score calculation
        complexity_factors = {
            'size_factor': min(n_rows / 100000, 1.0) * 0.3,
            'feature_factor': min(n_features / 100, 1.0) * 0.25,
            'missing_factor': missing_percentage / 100 * 0.15,
            'correlation_factor': avg_correlation * 0.15,
            'skewness_factor': (highly_skewed / max(len(feature_skewness), 1)) * 0.15
        }
        complexity_score = sum(complexity_factors.values())
        
        return DatasetCharacteristics(
            size_category=size_category,
            feature_category=feature_category,
            target_distribution=target_distribution,
            missing_data_percentage=missing_percentage,
            outlier_percentage=outlier_percentage,
            feature_correlation=avg_correlation,
            skewness_level=skewness_level,
            complexity_score=complexity_score
        )
    
    def _generate_model_suggestions(self, characteristics: DatasetCharacteristics) -> List[ModelSuggestion]:
        """Generate model suggestions based on dataset characteristics"""
        
        suggestions = []
        
        # Define model characteristics and scoring
        model_profiles = {
            'linear_regression': {
                'best_for': ['small', 'medium'],
                'handles_skewness': False,
                'handles_outliers': False,
                'training_speed': 'very_fast',
                'interpretability': 'high',
                'complexity': 'low'
            },
            'ridge': {
                'best_for': ['small', 'medium', 'large'],
                'handles_skewness': False,
                'handles_outliers': True,
                'training_speed': 'fast',
                'interpretability': 'high',
                'complexity': 'low'
            },
            'lasso': {
                'best_for': ['medium', 'large'],
                'handles_skewness': False,
                'handles_outliers': True,
                'training_speed': 'fast',
                'interpretability': 'high',
                'complexity': 'low',
                'feature_selection': True
            },
            'elastic_net': {
                'best_for': ['medium', 'large'],
                'handles_skewness': False,
                'handles_outliers': True,
                'training_speed': 'fast',
                'interpretability': 'high',
                'complexity': 'low',
                'feature_selection': True
            },
            'random_forest': {
                'best_for': ['small', 'medium', 'large'],
                'handles_skewness': True,
                'handles_outliers': True,
                'training_speed': 'medium',
                'interpretability': 'medium',
                'complexity': 'medium',
                'non_linear': True
            },
            'gradient_boosting': {
                'best_for': ['medium', 'large'],
                'handles_skewness': True,
                'handles_outliers': True,
                'training_speed': 'slow',
                'interpretability': 'medium',
                'complexity': 'high',
                'non_linear': True
            },
            'xgboost': {
                'best_for': ['medium', 'large', 'xlarge'],
                'handles_skewness': True,
                'handles_outliers': True,
                'training_speed': 'medium',
                'interpretability': 'low',
                'complexity': 'high',
                'non_linear': True
            },
            'lightgbm': {
                'best_for': ['large', 'xlarge'],
                'handles_skewness': True,
                'handles_outliers': True,
                'training_speed': 'fast',
                'interpretability': 'low',
                'complexity': 'high',
                'non_linear': True
            },
            'svr': {
                'best_for': ['small', 'medium'],
                'handles_skewness': True,
                'handles_outliers': False,
                'training_speed': 'slow',
                'interpretability': 'low',
                'complexity': 'high',
                'non_linear': True
            },
            'knn': {
                'best_for': ['small', 'medium'],
                'handles_skewness': True,
                'handles_outliers': False,
                'training_speed': 'fast',
                'interpretability': 'medium',
                'complexity': 'low',
                'non_linear': True
            },
            'neural_network': {
                'best_for': ['medium', 'large', 'xlarge'],
                'handles_skewness': True,
                'handles_outliers': True,
                'training_speed': 'slow',
                'interpretability': 'very_low',
                'complexity': 'very_high',
                'non_linear': True
            }
        }
        
        # Score each model
        model_scores = {}
        
        for model_name, profile in model_profiles.items():
            score = 0.0
            reasoning_parts = []
            
            # Size compatibility
            if characteristics.size_category in profile['best_for']:
                score += 0.3
                reasoning_parts.append(f"good for {characteristics.size_category} datasets")
            else:
                score -= 0.1
            
            # Skewness handling
            if characteristics.skewness_level in ['moderate', 'high']:
                if profile.get('handles_skewness', False):
                    score += 0.2
                    reasoning_parts.append("handles skewed data well")
                else:
                    score -= 0.15
            
            # Outlier handling
            if characteristics.outlier_percentage > 5:
                if profile.get('handles_outliers', False):
                    score += 0.15
                    reasoning_parts.append("robust to outliers")
                else:
                    score -= 0.1
            
            # Feature count consideration
            if characteristics.feature_category in ['high', 'very_high']:
                if profile.get('feature_selection', False):
                    score += 0.1
                    reasoning_parts.append("includes feature selection")
                if model_name in ['lasso', 'elastic_net', 'random_forest']:
                    score += 0.1
                    reasoning_parts.append("handles many features well")
            
            # Non-linearity for complex data
            if characteristics.complexity_score > 0.6:
                if profile.get('non_linear', False):
                    score += 0.15
                    reasoning_parts.append("captures non-linear relationships")
            
            # Target distribution considerations
            if characteristics.target_distribution in ['highly_skewed', 'moderately_skewed']:
                if model_name in ['random_forest', 'gradient_boosting', 'xgboost']:
                    score += 0.1
                    reasoning_parts.append("handles skewed targets")
            
            # Performance vs speed trade-offs
            if characteristics.size_category == 'xlarge':
                if profile['training_speed'] in ['fast', 'medium']:
                    score += 0.1
                    reasoning_parts.append("efficient for large datasets")
            
            model_scores[model_name] = {
                'score': max(0, score),  # Ensure non-negative
                'reasoning': '; '.join(reasoning_parts),
                'profile': profile
            }
        
        # Sort models by score and get top suggestions
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        top_k = self.modeling_config.get('top_k_models', 3)
        
        for i, (model_name, model_info) in enumerate(sorted_models[:top_k]):
            confidence = min(0.95, max(0.3, model_info['score']))  # Scale to 30-95%
            
            suggestion = ModelSuggestion(
                model_name=model_name,
                confidence_score=confidence,
                reasoning=model_info['reasoning'] or f"Good general choice for {characteristics.size_category} datasets",
                expected_performance=self._get_performance_expectation(confidence),
                training_time=model_info['profile']['training_speed'],
                complexity=model_info['profile']['complexity']
            )
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _get_performance_expectation(self, confidence: float) -> str:
        """Get performance expectation based on confidence score"""
        if confidence >= 0.8:
            return "excellent"
        elif confidence >= 0.7:
            return "very good"
        elif confidence >= 0.6:
            return "good"
        elif confidence >= 0.5:
            return "moderate"
        else:
            return "uncertain"
    
    def get_all_available_models(self) -> List[str]:
        """Get list of all available models"""
        all_models = []
        for category, models in self.available_models.items():
            all_models.extend(models)
        return all_models
    
    def get_suggested_models(self, suggestions: List[ModelSuggestion]) -> List[str]:
        """Get list of suggested model names"""
        return [suggestion.model_name for suggestion in suggestions]
    
    def _suggestion_to_dict(self, suggestion: ModelSuggestion) -> Dict:
        """Convert ModelSuggestion to dictionary for JSON serialization"""
        return {
            'model_name': suggestion.model_name,
            'confidence_score': suggestion.confidence_score,
            'reasoning': suggestion.reasoning,
            'expected_performance': suggestion.expected_performance,
            'training_time': suggestion.training_time,
            'complexity': suggestion.complexity
        }
    
    def _characteristics_to_dict(self, characteristics: DatasetCharacteristics) -> Dict:
        """Convert DatasetCharacteristics to dictionary for JSON serialization"""
        return {
            'size_category': characteristics.size_category,
            'feature_category': characteristics.feature_category,
            'target_distribution': characteristics.target_distribution,
            'missing_data_percentage': characteristics.missing_data_percentage,
            'outlier_percentage': characteristics.outlier_percentage,
            'feature_correlation': characteristics.feature_correlation,
            'skewness_level': characteristics.skewness_level,
            'complexity_score': characteristics.complexity_score
        }