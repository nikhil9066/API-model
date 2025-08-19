"""
core/model_trainer.py
Minimal working model trainer - guaranteed to work
"""

import pandas as pd
import numpy as np
import pickle
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass

# Core scikit-learn models (always available)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelResult:
    """Result structure for individual model training"""
    model_name: str
    model: Any
    train_score: float
    test_score: float
    cv_score: float
    cv_std: float
    training_time: float
    hyperparameters: Dict
    feature_importance: Optional[np.ndarray] = None

class ModelTrainer:
    """Minimal but comprehensive model trainer"""
    
    def __init__(self, config: Dict, job_id: str, progress_tracker=None, state_manager=None):
        self.config = config
        self.job_id = job_id
        self.progress = progress_tracker
        self.state_manager = state_manager
        self.logger = logging.getLogger(__name__)
        
        # Simple configuration
        self.test_size = 0.2
        self.cv_folds = 5
        self.random_state = 42
        
        # Results storage
        self.results: List[ModelResult] = []
        self.best_model: Optional[ModelResult] = None
        
        print("✅ ModelTrainer initialized successfully")
        
    def train_all_models(self, df: pd.DataFrame, target_variable: str, 
                        suggested_models: Optional[List[str]] = None,
                        train_all: bool = True) -> Dict[str, Any]:
        """Train all available models"""
        
        print(f"🚀 Starting model training for job {self.job_id}")
        
        if self.progress:
            self.progress.start_step("model_training", "Training models")
        
        # Prepare data
        X, y, X_train, X_test, y_train, y_test = self._prepare_data(df, target_variable)
        
        # Get models to train
        models = self._get_core_models()
        
        # Train each model
        total_models = len(models)
        
        for i, (model_name, model_config) in enumerate(models.items(), 1):
            print(f"Training {model_name} ({i}/{total_models})...")
            
            if self.progress:
                progress_pct = (i / total_models) * 80
                self.progress.update_step_progress(
                    "model_training", 
                    f"Training {model_name} ({i}/{total_models})",
                    {'current_model': model_name, 'progress': progress_pct}
                )
            
            try:
                result = self._train_single_model(
                    model_name, model_config, 
                    X_train, X_test, y_train, y_test, X, y
                )
                
                if result:
                    self.results.append(result)
                    print(f"✅ {model_name}: R² = {result.test_score:.4f}")
                
            except Exception as e:
                print(f"❌ {model_name} failed: {str(e)}")
                continue
        
        # Find best model
        if self.results:
            self.best_model = max(self.results, key=lambda x: x.test_score)
            print(f"🏆 Best model: {self.best_model.model_name} (R² = {self.best_model.test_score:.4f})")
        
        # Generate summary
        summary = self._generate_training_summary()
        
        if self.progress:
            self.progress.complete_step("model_training")
        
        return summary
    
    def _prepare_data(self, df: pd.DataFrame, target_variable: str) -> Tuple:
        """Prepare data for training"""
        # Separate features and target
        X = df.drop(columns=[target_variable])
        y = df[target_variable]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"📊 Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
        
        return X, y, X_train, X_test, y_train, y_test
    
    def _get_core_models(self) -> Dict[str, Dict]:
        """Get core scikit-learn models that always work"""
        return {
            'LinearRegression': {
                'model': LinearRegression(),
                'needs_scaling': False
            },
            'Ridge': {
                'model': Ridge(random_state=self.random_state),
                'needs_scaling': True
            },
            'Lasso': {
                'model': Lasso(random_state=self.random_state),
                'needs_scaling': True
            },
            'DecisionTree': {
                'model': DecisionTreeRegressor(random_state=self.random_state, max_depth=10),
                'needs_scaling': False
            },
            'RandomForest': {
                'model': RandomForestRegressor(random_state=self.random_state, n_estimators=100),
                'needs_scaling': False
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=self.random_state, n_estimators=100),
                'needs_scaling': False
            },
            'SVR': {
                'model': SVR(kernel='rbf'),
                'needs_scaling': True
            },
            'KNeighbors': {
                'model': KNeighborsRegressor(n_neighbors=5),
                'needs_scaling': True
            }
        }
    
    def _train_single_model(self, model_name: str, model_config: Dict,
                           X_train, X_test, y_train, y_test, X, y) -> Optional[ModelResult]:
        """Train a single model"""
        
        start_time = time.time()
        
        try:
            model = model_config['model']
            needs_scaling = model_config.get('needs_scaling', False)
            
            # Create pipeline with optional scaling
            if needs_scaling:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
            else:
                pipeline = Pipeline([
                    ('model', model)
                ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            train_score = pipeline.score(X_train, y_train)
            test_score = pipeline.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X, y, cv=self.cv_folds, scoring='r2')
            cv_score = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Feature importance (if available)
            feature_importance = None
            try:
                if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                    feature_importance = pipeline.named_steps['model'].feature_importances_
                elif hasattr(pipeline.named_steps['model'], 'coef_'):
                    feature_importance = np.abs(pipeline.named_steps['model'].coef_)
            except:
                pass
            
            training_time = time.time() - start_time
            
            return ModelResult(
                model_name=model_name,
                model=pipeline,
                train_score=train_score,
                test_score=test_score,
                cv_score=cv_score,
                cv_std=cv_std,
                training_time=training_time,
                hyperparameters={},
                feature_importance=feature_importance
            )
            
        except Exception as e:
            self.logger.error(f"Failed to train {model_name}: {str(e)}")
            return None
    
    def _generate_training_summary(self) -> Dict[str, Any]:
        """Generate training summary"""
        if not self.results:
            return {'error': 'No models were successfully trained'}
        
        # Sort results by test score
        sorted_results = sorted(self.results, key=lambda x: x.test_score, reverse=True)
        
        summary = {
            'total_models_trained': len(self.results),
            'best_model': {
                'model_name': self.best_model.model_name,
                'name': self.best_model.model_name,  # For compatibility
                'test_score': self.best_model.test_score,
                'train_score': self.best_model.train_score,
                'cv_score': self.best_model.cv_score,
                'training_time': self.best_model.training_time
            } if self.best_model else None,
            'all_models_performance': {
                result.model_name: {
                    'test_score': result.test_score,
                    'train_score': result.train_score,
                    'cv_score': result.cv_score,
                    'training_time': result.training_time
                }
                for result in self.results
            }
        }
        
        return summary
    
    def get_model_comparison_data(self) -> pd.DataFrame:
        """Get model comparison data for visualization"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Model': result.model_name,
                'Train Score': result.train_score,
                'Test Score': result.test_score,
                'CV Score': result.cv_score,
                'CV Std': result.cv_std,
                'Training Time': result.training_time
            })
        
        return pd.DataFrame(data).sort_values('Test Score', ascending=False)

# Test the trainer
def test_model_trainer():
    """Test the ModelTrainer with sample data"""
    
    # Create sample data
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(5, 2, 100),
        'feature3': np.random.uniform(0, 10, 100),
        'target': np.random.normal(10, 3, 100)
    }
    df = pd.DataFrame(data)
    
    # Test trainer
    config = {}
    trainer = ModelTrainer(config, "test_job")
    
    print("Testing ModelTrainer...")
    summary = trainer.train_all_models(df, 'target')
    
    print(f"Training completed!")
    print(f"Models trained: {summary.get('total_models_trained', 0)}")
    if summary.get('best_model'):
        print(f"Best model: {summary['best_model']['name']}")
        print(f"Best score: {summary['best_model']['test_score']:.4f}")

if __name__ == "__main__":
    test_model_trainer()