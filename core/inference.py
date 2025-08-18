"""
core/inference.py
Real-time inference engine for trained models
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, List, Any, Optional, Union
import logging
from dataclasses import dataclass

@dataclass
class PredictionResult:
    """Result structure for predictions"""
    predictions: pd.Series
    feature_names: List[str]
    model_used: str
    preprocessing_applied: bool
    feature_engineering_applied: bool
    prediction_metadata: Dict[str, Any]

class InferenceEngine:
    """Real-time inference engine for trained models"""
    
    def __init__(self, config: Dict, state_manager=None):
        self.config = config
        self.state_manager = state_manager
        self.logger = logging.getLogger(__name__)
        
        # Cache for loaded models and preprocessors
        self._model_cache = {}
        self._preprocessor_cache = {}
        self._feature_engineering_cache = {}
    
    def predict(self, job_id: str, new_data: Union[pd.DataFrame, str], 
                output_file: Optional[str] = None) -> PredictionResult:
        """Make predictions using a trained model"""
        
        self.logger.info(f"Making predictions for job {job_id}")
        
        # Load new data if file path provided
        if isinstance(new_data, str):
            new_data = self._load_data(new_data)
        
        # Get job state and model information
        if not self.state_manager:
            raise ValueError("State manager required for inference")
        
        job_state = self.state_manager.get_job_state(job_id)
        if not job_state:
            raise ValueError(f"Job {job_id} not found")
        
        inference_config = job_state.get('inference_setup', {})
        if not inference_config.get('model_ready', False):
            raise ValueError(f"Model for job {job_id} is not ready for inference")
        
        # Get model information
        best_model_info = job_state.get('model_results', {}).get('best_model', {})
        model_name = best_model_info.get('model_name', 'unknown')
        
        # Get expected feature names and target variable
        expected_features = inference_config.get('feature_names', [])
        target_variable = inference_config.get('target_variable')
        
        self.logger.info(f"Using model: {model_name}")
        self.logger.info(f"Input data shape: {new_data.shape}")
        
        # Step 1: Prepare data (remove target if present, basic validation)
        prepared_data = self._prepare_initial_data(new_data, target_variable)
        
        # Step 2: Apply preprocessing pipeline
        preprocessing_applied = False
        preprocessor = self._load_preprocessor(job_id, inference_config)
        if preprocessor:
            try:
                prepared_data = self._apply_preprocessing(prepared_data, preprocessor, target_variable)
                preprocessing_applied = True
                self.logger.info("Applied preprocessing pipeline to new data")
            except Exception as e:
                self.logger.warning(f"Preprocessing failed: {str(e)}, continuing with original data")
        
        # Step 3: Apply feature engineering pipeline
        feature_engineering_applied = False
        feature_engineer = self._load_feature_engineering_pipeline(job_id)
        if feature_engineer:
            try:
                prepared_data = self._apply_feature_engineering(prepared_data, feature_engineer, target_variable)
                feature_engineering_applied = True
                self.logger.info("Applied feature engineering pipeline to new data")
            except Exception as e:
                self.logger.warning(f"Feature engineering failed: {str(e)}, continuing without it")
        
        # Step 4: Final data preparation for model
        final_data = self._prepare_final_data_for_model(prepared_data, expected_features, target_variable)
        
        # Step 5: Load model and make predictions
        model = self._load_model(job_id, inference_config)
        
        try:
            predictions = model.predict(final_data)
            
            # Convert to pandas Series
            if hasattr(new_data, 'index'):
                predictions_series = pd.Series(predictions, index=new_data.index, name='predictions')
            else:
                predictions_series = pd.Series(predictions, name='predictions')
            
            self.logger.info(f"Generated {len(predictions)} predictions successfully")
            
        except Exception as e:
            self.logger.error(f"Model prediction failed: {str(e)}")
            raise ValueError(f"Failed to make predictions: {str(e)}")
        
        # Create result
        result = PredictionResult(
            predictions=predictions_series,
            feature_names=list(final_data.columns) if hasattr(final_data, 'columns') else expected_features,
            model_used=model_name,
            preprocessing_applied=preprocessing_applied,
            feature_engineering_applied=feature_engineering_applied,
            prediction_metadata={
                'job_id': job_id,
                'input_shape': new_data.shape,
                'final_shape': final_data.shape if hasattr(final_data, 'shape') else (len(predictions), 0),
                'prediction_count': len(predictions),
                'target_variable': target_variable,
                'expected_features': expected_features
            }
        )
        
        # Save predictions if output file specified
        if output_file:
            self._save_predictions(result, new_data, output_file)
        
        return result
    
    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file"""
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                return pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif file_extension == '.parquet':
                return pd.read_parquet(file_path)
            elif file_extension == '.json':
                return pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            raise ValueError(f"Failed to load data from {file_path}: {str(e)}")
    
    def _prepare_initial_data(self, data: pd.DataFrame, target_variable: str) -> pd.DataFrame:
        """Initial data preparation - remove target variable if present"""
        
        prepared_data = data.copy()
        
        # Remove target variable if present
        if target_variable and target_variable in prepared_data.columns:
            prepared_data = prepared_data.drop(columns=[target_variable])
            self.logger.info(f"Removed target variable '{target_variable}' from prediction data")
        
        return prepared_data
    
    def _load_model(self, job_id: str, inference_config: Dict) -> Any:
        """Load trained model"""
        
        # Check cache first
        if job_id in self._model_cache:
            return self._model_cache[job_id]
        
        model_path = inference_config.get('model_path')
        if not model_path:
            # Fallback to best model path
            if self.state_manager:
                model_path = self.state_manager.get_best_model_path(job_id)
        
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found for job {job_id}")
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Cache the model
            self._model_cache[job_id] = model
            
            self.logger.info(f"Loaded model from {model_path}")
            return model
            
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
    
    def _load_preprocessor(self, job_id: str, inference_config: Dict) -> Optional[Any]:
        """Load preprocessing pipeline"""
        
        # Check cache first
        cache_key = f"{job_id}_preprocessor"
        if cache_key in self._preprocessor_cache:
            return self._preprocessor_cache[cache_key]
        
        preprocessor_path = inference_config.get('preprocessing_pipeline_path')
        if not preprocessor_path:
            # Fallback to default preprocessor path
            if self.state_manager:
                preprocessor_path = self.state_manager.get_preprocessor_path(job_id)
        
        if not preprocessor_path or not os.path.exists(preprocessor_path):
            self.logger.info("No preprocessing pipeline found")
            return None
        
        try:
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            
            # Cache the preprocessor
            self._preprocessor_cache[cache_key] = preprocessor
            
            self.logger.info(f"Loaded preprocessor from {preprocessor_path}")
            return preprocessor
            
        except Exception as e:
            self.logger.warning(f"Failed to load preprocessor: {str(e)}")
            return None
    
    def _load_feature_engineering_pipeline(self, job_id: str) -> Optional[Any]:
        """Load feature engineering pipeline"""
        
        # Check cache first
        cache_key = f"{job_id}_feature_engineering"
        if cache_key in self._feature_engineering_cache:
            return self._feature_engineering_cache[cache_key]
        
        # Construct feature engineering pipeline path
        if self.state_manager:
            job_path = self.state_manager.get_job_path(job_id)
            fe_pipeline_path = os.path.join(job_path, 'feature_engineering_pipeline.pkl')
        else:
            self.logger.warning("State manager not available for feature engineering pipeline")
            return None
        
        if not os.path.exists(fe_pipeline_path):
            self.logger.info("No feature engineering pipeline found")
            return None
        
        try:
            with open(fe_pipeline_path, 'rb') as f:
                fe_pipeline_info = pickle.load(f)
            
            # Extract the actual pipeline
            if isinstance(fe_pipeline_info, dict) and 'pipeline' in fe_pipeline_info:
                fe_pipeline = fe_pipeline_info['pipeline']
            else:
                fe_pipeline = fe_pipeline_info
            
            # Cache the feature engineering pipeline
            self._feature_engineering_cache[cache_key] = fe_pipeline
            
            self.logger.info(f"Loaded feature engineering pipeline from {fe_pipeline_path}")
            return fe_pipeline
            
        except Exception as e:
            self.logger.warning(f"Failed to load feature engineering pipeline: {str(e)}")
            return None
    
    def _apply_preprocessing(self, data: pd.DataFrame, preprocessor: Any, target_variable: str) -> pd.DataFrame:
        """Apply preprocessing pipeline to data"""
        
        try:
            # The preprocessor might expect the target variable for certain operations
            # but we'll handle it gracefully
            transformed_data = preprocessor.transform(data)
            
            # Convert back to DataFrame if needed
            if isinstance(transformed_data, np.ndarray):
                # Try to preserve column names
                if hasattr(data, 'columns'):
                    n_cols = transformed_data.shape[1] if len(transformed_data.shape) > 1 else 1
                    if n_cols == len(data.columns):
                        transformed_data = pd.DataFrame(transformed_data, columns=data.columns, index=data.index)
                    else:
                        # Number of columns changed, create generic names
                        col_names = [f'feature_{i}' for i in range(n_cols)]
                        transformed_data = pd.DataFrame(transformed_data, columns=col_names, index=data.index)
                else:
                    transformed_data = pd.DataFrame(transformed_data, index=data.index)
            
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Preprocessing transformation failed: {str(e)}")
            raise
    
    def _apply_feature_engineering(self, data: pd.DataFrame, feature_engineer: Any, target_variable: str) -> pd.DataFrame:
        """Apply feature engineering pipeline to data"""
        
        try:
            # Add a dummy target variable if the feature engineering expects it
            data_with_dummy_target = data.copy()
            if target_variable and target_variable not in data_with_dummy_target.columns:
                # Add dummy target for feature engineering (will be removed later)
                data_with_dummy_target[target_variable] = 0
            
            # Apply feature engineering
            transformed_data = feature_engineer.transform(data_with_dummy_target)
            
            # Remove dummy target if we added it
            if target_variable and target_variable in transformed_data.columns and target_variable not in data.columns:
                transformed_data = transformed_data.drop(columns=[target_variable])
            
            # Convert back to DataFrame if needed
            if isinstance(transformed_data, np.ndarray):
                # Create generic column names
                n_cols = transformed_data.shape[1] if len(transformed_data.shape) > 1 else 1
                col_names = [f'engineered_feature_{i}' for i in range(n_cols)]
                transformed_data = pd.DataFrame(transformed_data, columns=col_names, index=data.index)
            
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Feature engineering transformation failed: {str(e)}")
            raise
    
    def _prepare_final_data_for_model(self, data: pd.DataFrame, expected_features: List[str], 
                                    target_variable: str) -> pd.DataFrame:
        """Final data preparation before model prediction"""
        
        final_data = data.copy()
        
        # Remove target variable if somehow still present
        if target_variable and target_variable in final_data.columns:
            final_data = final_data.drop(columns=[target_variable])
        
        # Handle expected features
        if expected_features:
            # Check for missing features
            missing_features = set(expected_features) - set(final_data.columns)
            extra_features = set(final_data.columns) - set(expected_features)
            
            if missing_features:
                self.logger.warning(f"Missing expected features: {list(missing_features)}")
                # Add missing features with default values (zeros)
                for feature in missing_features:
                    final_data[feature] = 0.0
                    self.logger.warning(f"Added missing feature '{feature}' with default value 0")
            
            if extra_features:
                self.logger.info(f"Extra features will be ignored: {list(extra_features)}")
            
            # Reorder columns to match expected order
            try:
                final_data = final_data[expected_features]
            except KeyError as e:
                self.logger.warning(f"Could not reorder features: {e}")
                # Use available features only
                available_features = [f for f in expected_features if f in final_data.columns]
                final_data = final_data[available_features]
        
        # Ensure all data is numeric
        for col in final_data.columns:
            if not pd.api.types.is_numeric_dtype(final_data[col]):
                try:
                    final_data[col] = pd.to_numeric(final_data[col], errors='coerce')
                except:
                    self.logger.warning(f"Could not convert column '{col}' to numeric")
        
        # Handle missing values
        if final_data.isnull().any().any():
            missing_count = final_data.isnull().sum().sum()
            self.logger.warning(f"Found {missing_count} missing values, filling with 0")
            final_data = final_data.fillna(0)
        
        # Handle infinite values
        if np.isinf(final_data.select_dtypes(include=[np.number])).any().any():
            self.logger.warning("Found infinite values, replacing with 0")
            final_data = final_data.replace([np.inf, -np.inf], 0)
        
        return final_data
    
    def _save_predictions(self, result: PredictionResult, original_data: pd.DataFrame, 
                         output_file: str):
        """Save predictions to file"""
        
        try:
            # Create output dataframe
            output_df = original_data.copy()
            
            # Remove target variable if present
            target_variable = result.prediction_metadata.get('target_variable')
            if target_variable and target_variable in output_df.columns:
                output_df = output_df.drop(columns=[target_variable])
            
            # Add predictions
            output_df['predictions'] = result.predictions.values
            
            # Add metadata columns
            output_df['model_used'] = result.model_used
            output_df['job_id'] = result.prediction_metadata['job_id']
            output_df['preprocessing_applied'] = result.preprocessing_applied
            output_df['feature_engineering_applied'] = result.feature_engineering_applied
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Save to file
            file_extension = os.path.splitext(output_file)[1].lower()
            
            if file_extension == '.csv':
                output_df.to_csv(output_file, index=False)
            elif file_extension in ['.xlsx', '.xls']:
                output_df.to_excel(output_file, index=False)
            elif file_extension == '.parquet':
                output_df.to_parquet(output_file, index=False)
            else:
                # Default to CSV
                if not output_file.endswith('.csv'):
                    output_file += '.csv'
                output_df.to_csv(output_file, index=False)
            
            self.logger.info(f"Saved predictions to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save predictions: {str(e)}")
            # Don't raise error here, just log it
    
    def batch_predict(self, job_ids: List[str], data_files: List[str], 
                     output_dir: str = "predictions") -> Dict[str, PredictionResult]:
        """Make batch predictions for multiple models"""
        
        if len(job_ids) != len(data_files):
            raise ValueError("Number of job IDs must match number of data files")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        for job_id, data_file in zip(job_ids, data_files):
            try:
                output_file = os.path.join(output_dir, f"predictions_{job_id}.csv")
                result = self.predict(job_id, data_file, output_file)
                results[job_id] = result
                
                self.logger.info(f"Completed batch prediction for {job_id}")
                
            except Exception as e:
                self.logger.error(f"Batch prediction failed for {job_id}: {str(e)}")
                continue
        
        return results
    
    def compare_model_predictions(self, job_ids: List[str], data_file: str) -> pd.DataFrame:
        """Compare predictions from multiple models on the same data"""
        
        # Load data once
        data = self._load_data(data_file) if isinstance(data_file, str) else data_file
        
        comparison_df = data.copy()
        
        # Remove any target variables
        for job_id in job_ids:
            if self.state_manager:
                job_state = self.state_manager.get_job_state(job_id)
                if job_state:
                    target_var = job_state.get('inference_setup', {}).get('target_variable')
                    if target_var and target_var in comparison_df.columns:
                        comparison_df = comparison_df.drop(columns=[target_var])
        
        # Get predictions from each model
        for job_id in job_ids:
            try:
                result = self.predict(job_id, data.copy())
                model_name = result.model_used
                comparison_df[f'pred_{model_name}_{job_id}'] = result.predictions.values
                
            except Exception as e:
                self.logger.error(f"Failed to get predictions from {job_id}: {str(e)}")
                continue
        
        return comparison_df
    
    def get_prediction_intervals(self, job_id: str, data: Union[pd.DataFrame, str], 
                               confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """Get prediction intervals (if supported by the model)"""
        
        # Load model
        if not self.state_manager:
            raise ValueError("State manager required for prediction intervals")
        
        job_state = self.state_manager.get_job_state(job_id)
        if not job_state:
            raise ValueError(f"Job {job_id} not found")
        
        inference_config = job_state.get('inference_setup', {})
        model = self._load_model(job_id, inference_config)
        
        # Prepare data
        if isinstance(data, str):
            data = self._load_data(data)
        
        expected_features = inference_config.get('feature_names', [])
        target_variable = inference_config.get('target_variable')
        prepared_data = self._prepare_initial_data(data, target_variable)
        
        # Apply preprocessing
        preprocessor = self._load_preprocessor(job_id, inference_config)
        if preprocessor:
            try:
                prepared_data = self._apply_preprocessing(prepared_data, preprocessor, target_variable)
            except Exception as e:
                self.logger.warning(f"Preprocessing failed for intervals: {str(e)}")
        
        # Apply feature engineering
        feature_engineer = self._load_feature_engineering_pipeline(job_id)
        if feature_engineer:
            try:
                prepared_data = self._apply_feature_engineering(prepared_data, feature_engineer, target_variable)
            except Exception as e:
                self.logger.warning(f"Feature engineering failed for intervals: {str(e)}")
        
        # Final data preparation
        prepared_data = self._prepare_final_data_for_model(prepared_data, expected_features, target_variable)
        
        # Get the actual model from pipeline if needed
        actual_model = model
        if hasattr(model, 'steps'):
            actual_model = model.steps[-1][1]
        
        # Check if model supports prediction intervals
        if hasattr(actual_model, 'predict') and hasattr(actual_model, 'predict_proba'):
            # Some models support uncertainty estimation
            predictions = model.predict(prepared_data)
            
            # For models without built-in intervals, use bootstrap or cross-validation approach
            # This is a simplified implementation
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            # Simple approach: assume normal distribution around predictions
            # In practice, you'd want more sophisticated methods
            std_error = np.std(predictions) * 0.1  # Rough estimate
            
            from scipy import stats as scipy_stats
            z_score = scipy_stats.norm.ppf(1 - alpha / 2)
            
            lower_bound = predictions - z_score * std_error
            upper_bound = predictions + z_score * std_error
            
            return {
                'predictions': predictions,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'confidence_level': confidence_level
            }
        
        else:
            # Model doesn't support intervals
            predictions = model.predict(prepared_data)
            return {
                'predictions': predictions,
                'lower_bound': None,
                'upper_bound': None,
                'confidence_level': confidence_level,
                'message': 'Model does not support prediction intervals'
            }
    
    def validate_model_performance(self, job_id: str, test_data: Union[pd.DataFrame, str]) -> Dict[str, float]:
        """Validate model performance on new test data"""
        
        # Load test data
        if isinstance(test_data, str):
            test_data = self._load_data(test_data)
        
        # Get job state
        if not self.state_manager:
            raise ValueError("State manager required for validation")
        
        job_state = self.state_manager.get_job_state(job_id)
        if not job_state:
            raise ValueError(f"Job {job_id} not found")
        
        # Get target variable
        target_variable = job_state.get('inference_setup', {}).get('target_variable')
        if not target_variable or target_variable not in test_data.columns:
            raise ValueError(f"Target variable '{target_variable}' not found in test data")
        
        # Separate features and target
        y_true = test_data[target_variable]
        X_test = test_data.drop(columns=[target_variable])
        
        # Make predictions
        result = self.predict(job_id, X_test)
        y_pred = result.predictions
        
        # Calculate metrics
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        metrics = {
            'r2_score': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'sample_count': len(y_true)
        }
        
        self.logger.info(f"Validation completed for {job_id}: R² = {metrics['r2_score']:.4f}")
        return metrics
    
    def clear_cache(self):
        """Clear model and preprocessor cache"""
        self._model_cache.clear()
        self._preprocessor_cache.clear()
        self._feature_engineering_cache.clear()
        self.logger.info("Cleared inference cache")
    
    def get_model_info(self, job_id: str) -> Dict[str, Any]:
        """Get information about a trained model"""
        
        if not self.state_manager:
            raise ValueError("State manager required")
        
        job_state = self.state_manager.get_job_state(job_id)
        if not job_state:
            raise ValueError(f"Job {job_id} not found")
        
        model_results = job_state.get('model_results', {})
        best_model = model_results.get('best_model', {})
        inference_setup = job_state.get('inference_setup', {})
        
        return {
            'job_id': job_id,
            'model_name': best_model.get('model_name'),
            'model_ready': inference_setup.get('model_ready', False),
            'feature_count': len(inference_setup.get('feature_names', [])),
            'target_variable': inference_setup.get('target_variable'),
            'training_score': best_model.get('test_score'),
            'model_file': best_model.get('model_file'),
            'feature_names': inference_setup.get('feature_names', []),
            'preprocessing_available': bool(inference_setup.get('preprocessing_pipeline_path')),
            'feature_engineering_available': os.path.exists(
                os.path.join(self.state_manager.get_job_path(job_id), 'feature_engineering_pipeline.pkl')
            ) if self.state_manager else False
        }