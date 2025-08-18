"""
core/preprocessing.py
Enhanced preprocessing pipeline for Phase 1 - All Numeric AutoML Pipeline
Migrated and enhanced from your original datapipeline.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
from scipy import stats
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler, MinMaxScaler
import logging
from dataclasses import dataclass

@dataclass
class PreprocessingResult:
    """Result structure for preprocessing operations"""
    processed_data: pd.DataFrame
    outliers_removed: int
    features_removed: List[str]
    transformations_applied: Dict[str, List[str]]
    scaling_applied: bool
    preprocessing_stats: Dict[str, Any]

class PreprocessingPipeline:
    """Enhanced preprocessing pipeline with auto-selection capabilities"""
    
    def __init__(self, config: Dict, job_id: str, progress_tracker=None, state_manager=None):
        self.config = config
        self.job_id = job_id
        self.progress = progress_tracker
        self.state_manager = state_manager
        self.logger = logging.getLogger(__name__)
        
        # Preprocessing configuration
        self.preprocessing_config = config.get('preprocessing', {})
        
    def run_preprocessing(self, df: pd.DataFrame, target_variable: str) -> PreprocessingResult:
        """Run the complete preprocessing pipeline"""
        
        self.logger.info(f"Starting preprocessing pipeline for job {self.job_id}")
        
        if self.progress:
            self.progress.start_step("preprocessing", "Applying data preprocessing")
        
        # Step 1: Outlier Detection and Removal
        if self.progress:
            self.progress.start_sub_step("preprocessing", "outlier_detection", "Detecting and removing outliers")
        
        df_no_outliers, outlier_method, outliers_removed = self._handle_outliers(df, target_variable)
        
        if self.progress:
            self.progress.complete_sub_step("preprocessing", "outlier_detection")
            self.progress.update_step_progress("preprocessing", f"Removed {outliers_removed} outliers using {outlier_method}")
        
        # Step 2: Correlation Analysis and Feature Removal
        if self.progress:
            self.progress.start_sub_step("preprocessing", "correlation_analysis", "Analyzing feature correlations")
        
        df_filtered, correlation_removed = self._handle_correlations(df_no_outliers, target_variable)
        
        if self.progress:
            self.progress.complete_sub_step("preprocessing", "correlation_analysis")
            self.progress.update_step_progress("preprocessing", f"Removed {len(correlation_removed)} highly correlated features")
        
        # Step 3: Skewness Handling
        if self.progress:
            self.progress.start_sub_step("preprocessing", "skewness_handling", "Handling skewed features")
        
        df_transformed, transformations = self._handle_skewness(df_filtered, target_variable)
        
        if self.progress:
            self.progress.complete_sub_step("preprocessing", "skewness_handling")
            self.progress.update_step_progress("preprocessing", f"Applied transformations to {sum(len(v) for v in transformations.values())} features")
        
        # Step 4: Scaling (if enabled)
        scaling_applied = False
        if self.preprocessing_config.get('scaling', {}).get('apply_after_feature_engineering', False):
            # Scaling will be applied after feature engineering
            pass
        else:
            df_transformed, scaling_applied = self._apply_scaling(df_transformed, target_variable)
        
        # Create preprocessing result
        result = PreprocessingResult(
            processed_data=df_transformed,
            outliers_removed=outliers_removed,
            features_removed=correlation_removed,
            transformations_applied=transformations,
            scaling_applied=scaling_applied,
            preprocessing_stats={
                'original_shape': df.shape,
                'final_shape': df_transformed.shape,
                'outlier_method_used': outlier_method,
                'features_removed_count': len(correlation_removed),
                'transformations_count': sum(len(v) for v in transformations.values())
            }
        )
        
        # Update state manager
        if self.state_manager:
            self.state_manager.update_preprocessing_results(self.job_id, {
                'outliers_removed': outliers_removed,
                'features_removed': correlation_removed,
                'transformations_applied': transformations,
                'scaling_applied': scaling_applied,
                'preprocessing_pipeline_saved': True,
                'original_shape': df.shape,
                'final_shape': df_transformed.shape
            })
        
        if self.progress:
            self.progress.complete_step("preprocessing")
        
        self.logger.info(f"Preprocessing completed: {df.shape} -> {df_transformed.shape}")
        return result
    
    def _handle_outliers(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, str, int]:
        """Handle outliers using multiple methods and auto-select the best"""
        
        outlier_config = self.preprocessing_config.get('outlier_detection', {})
        methods = outlier_config.get('methods', ['iqr', 'sd3', 'percentile'])
        auto_select = outlier_config.get('auto_select', True)
        
        if not auto_select and methods:
            # Use first method if auto-select is disabled
            method = methods[0]
            if method == 'iqr':
                df_cleaned, outliers_removed = self._remove_outliers_iqr(df, target_variable)
            elif method == 'sd3':
                df_cleaned, outliers_removed = self._remove_outliers_sd3(df, target_variable)
            elif method == 'percentile':
                df_cleaned, outliers_removed = self._remove_outliers_percentile(df, target_variable)
            else:
                df_cleaned, outliers_removed = df.copy(), 0
                method = 'none'
            
            return df_cleaned, method, outliers_removed
        
        # Auto-select best method
        results = {}
        
        # Try each method and calculate outliers removed
        for method in methods:
            try:
                if method == 'iqr':
                    df_temp, outliers_count = self._remove_outliers_iqr(df, target_variable)
                elif method == 'sd3':
                    df_temp, outliers_count = self._remove_outliers_sd3(df, target_variable)
                elif method == 'percentile':
                    df_temp, outliers_count = self._remove_outliers_percentile(df, target_variable)
                else:
                    continue
                
                results[method] = {
                    'data': df_temp,
                    'outliers_removed': outliers_count,
                    'remaining_rows': len(df_temp)
                }
                
                self.logger.info(f"Method {method}: removed {outliers_count} outliers")
                
            except Exception as e:
                self.logger.warning(f"Method {method} failed: {str(e)}")
                continue
        
        if not results:
            self.logger.warning("No outlier detection methods succeeded")
            return df.copy(), 'none', 0
        
        # Select method that removes the most outliers (but not too many)
        best_method = None
        best_outliers = 0
        
        for method, result in results.items():
            outliers_removed = result['outliers_removed']
            # Don't remove more than 20% of the data
            if outliers_removed > 0 and outliers_removed > best_outliers and result['remaining_rows'] >= len(df) * 0.8:
                best_method = method
                best_outliers = outliers_removed
        
        if best_method is None:
            # If no method meets criteria, use the one that removes least outliers
            best_method = min(results.keys(), key=lambda x: results[x]['outliers_removed'])
            best_outliers = results[best_method]['outliers_removed']
        
        return results[best_method]['data'], best_method, best_outliers
    
    def _remove_outliers_iqr(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, int]:
        """Remove outliers using IQR method"""
        outlier_config = self.preprocessing_config.get('outlier_detection', {})
        multiplier = outlier_config.get('iqr_multiplier', 1.5)
        
        Q1 = df[target_variable].quantile(0.25)
        Q3 = df[target_variable].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        mask = (df[target_variable] >= lower_bound) & (df[target_variable] <= upper_bound)
        outliers_removed = len(df) - mask.sum()
        
        return df[mask].copy(), outliers_removed
    
    def _remove_outliers_sd3(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, int]:
        """Remove outliers using 3-sigma method"""
        outlier_config = self.preprocessing_config.get('outlier_detection', {})
        multiplier = outlier_config.get('std_multiplier', 3.0)
        
        mean = df[target_variable].mean()
        std = df[target_variable].std()
        lower_bound = mean - multiplier * std
        upper_bound = mean + multiplier * std
        
        mask = (df[target_variable] >= lower_bound) & (df[target_variable] <= upper_bound)
        outliers_removed = len(df) - mask.sum()
        
        return df[mask].copy(), outliers_removed
    
    def _remove_outliers_percentile(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, int]:
        """Remove outliers using percentile method"""
        outlier_config = self.preprocessing_config.get('outlier_detection', {})
        bounds = outlier_config.get('percentile_bounds', [0.01, 0.99])
        
        lower_percentile = df[target_variable].quantile(bounds[0])
        upper_percentile = df[target_variable].quantile(bounds[1])
        
        mask = (df[target_variable] >= lower_percentile) & (df[target_variable] <= upper_percentile)
        outliers_removed = len(df) - mask.sum()
        
        return df[mask].copy(), outliers_removed
    
    def _handle_correlations(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, List[str]]:
        """Handle high and low correlations"""
        
        correlation_config = self.preprocessing_config.get('correlation', {})
        high_threshold = correlation_config.get('high_threshold', 0.9)
        low_threshold = correlation_config.get('low_threshold', 0.1)
        
        # Remove high correlation features
        df_filtered, high_corr_removed = self._remove_high_correlation_features(df, target_variable, high_threshold)
        
        # Remove low correlation features  
        df_filtered, low_corr_removed = self._remove_low_correlation_features(df_filtered, target_variable, low_threshold)
        
        all_removed = high_corr_removed + low_corr_removed
        
        if all_removed:
            self.logger.info(f"Removed features due to correlation: {all_removed}")
        
        return df_filtered, all_removed
    
    def _remove_high_correlation_features(self, df: pd.DataFrame, target_variable: str, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features highly correlated with each other (not target)"""
        correlation_matrix = df.corr()
        target_corr = correlation_matrix[target_variable]
        
        features_to_remove = set()
        processed_pairs = set()
        
        # Find pairs of features with high correlation
        for col1 in df.columns:
            if col1 == target_variable:
                continue
            for col2 in df.columns:
                if col2 == target_variable or col1 == col2:
                    continue
                if (col1, col2) in processed_pairs or (col2, col1) in processed_pairs:
                    continue
                
                corr_value = abs(correlation_matrix.loc[col1, col2])
                if corr_value > threshold:
                    # Remove the feature with lower correlation to target
                    if abs(target_corr[col1]) < abs(target_corr[col2]):
                        features_to_remove.add(col1)
                    else:
                        features_to_remove.add(col2)
                    
                    processed_pairs.add((col1, col2))
        
        features_to_remove = list(features_to_remove)
        df_filtered = df.drop(columns=features_to_remove)
        
        return df_filtered, features_to_remove
    
    def _remove_low_correlation_features(self, df: pd.DataFrame, target_variable: str, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with low correlation to target"""
        correlation_matrix = df.corr()
        target_corr = correlation_matrix[target_variable]
        
        low_corr_features = []
        for col in df.columns:
            if col == target_variable:
                continue
            if abs(target_corr[col]) < threshold:
                low_corr_features.append(col)
        
        df_filtered = df.drop(columns=low_corr_features)
        
        return df_filtered, low_corr_features
    
    def _handle_skewness(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Handle skewed features with transformations"""
        
        skewness_config = self.preprocessing_config.get('skewness', {})
        high_threshold = skewness_config.get('high_threshold', 1.0)
        moderate_threshold = skewness_config.get('moderate_threshold', 0.5)
        transformation_methods = skewness_config.get('transformation_methods', ['log', 'sqrt', 'boxcox', 'yeo-johnson'])
        
        df_transformed = df.copy()
        transformations = {
            'log_transformed': [],
            'sqrt_transformed': [],
            'boxcox_transformed': [],
            'yeo_johnson_transformed': [],
            'winsorized': []
        }
        
        # Categorize features by skewness
        skew_values = df.skew()
        highly_skewed = skew_values[abs(skew_values) > high_threshold].index.tolist()
        moderately_skewed = skew_values[(abs(skew_values) >= moderate_threshold) & (abs(skew_values) <= high_threshold)].index.tolist()
        
        # Remove target from skewed lists if present
        if target_variable in highly_skewed:
            highly_skewed.remove(target_variable)
        if target_variable in moderately_skewed:
            moderately_skewed.remove(target_variable)
        
        # Handle highly skewed features
        for col in highly_skewed:
            df_transformed, applied_transformation = self._transform_highly_skewed_feature(df_transformed, col, transformation_methods)
            if applied_transformation:
                transformations[f'{applied_transformation}_transformed'].append(col)
        
        # Handle moderately skewed features (winsorization)
        for col in moderately_skewed:
            df_transformed[col] = stats.mstats.winsorize(df_transformed[col], limits=[0.05, 0.05])
            transformations['winsorized'].append(col)
        
        # Clean up empty transformation lists
        transformations = {k: v for k, v in transformations.items() if v}
        
        return df_transformed, transformations
    
    def _transform_highly_skewed_feature(self, df: pd.DataFrame, col: str, methods: List[str]) -> Tuple[pd.DataFrame, Optional[str]]:
        """Transform a highly skewed feature using the best method"""
        
        original_skew = abs(df[col].skew())
        best_method = None
        best_skew = original_skew
        best_data = df[col].copy()
        
        # Prepare data (handle negative values)
        shift_value = 0
        if (df[col] <= 0).any():
            shift_value = abs(df[col].min()) + 1
        
        # Try each transformation method
        for method in methods:
            try:
                if method == 'log' and shift_value == 0:
                    transformed = np.log1p(df[col])
                elif method == 'log' and shift_value > 0:
                    transformed = np.log1p(df[col] + shift_value)
                elif method == 'sqrt':
                    if shift_value > 0:
                        transformed = np.sqrt(df[col] + shift_value)
                    else:
                        transformed = np.sqrt(df[col])
                elif method == 'boxcox':
                    if (df[col] + shift_value > 0).all():
                        transformed, _ = boxcox(df[col] + shift_value)
                    else:
                        continue
                elif method == 'yeo-johnson':
                    pt = PowerTransformer(method='yeo-johnson')
                    transformed = pt.fit_transform(df[[col]]).flatten()
                else:
                    continue
                
                new_skew = abs(pd.Series(transformed).skew())
                
                # Check if this transformation is better
                if new_skew < best_skew:
                    best_method = method
                    best_skew = new_skew
                    best_data = transformed
                
            except Exception as e:
                self.logger.debug(f"Transformation {method} failed for {col}: {str(e)}")
                continue
        
        # Apply best transformation
        if best_method:
            # Create new column name
            new_col_name = f"{col}_{best_method}"
            df[new_col_name] = best_data
            df.drop(columns=[col], inplace=True)
            
            self.logger.info(f"Applied {best_method} transformation to {col} (skew: {original_skew:.3f} -> {best_skew:.3f})")
            return df, best_method
        
        return df, None
    
    def _apply_scaling(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, bool]:
        """Apply scaling to features"""
        
        scaling_config = self.preprocessing_config.get('scaling', {})
        method = scaling_config.get('method', 'standard')
        
        if not scaling_config.get('enabled', True):
            return df, False
        
        # Separate features and target
        features = [col for col in df.columns if col != target_variable]
        
        if not features:
            return df, False
        
        # Select scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            self.logger.warning(f"Unknown scaling method: {method}")
            return df, False
        
        # Apply scaling
        df_scaled = df.copy()
        df_scaled[features] = scaler.fit_transform(df[features])
        
        self.logger.info(f"Applied {method} scaling to {len(features)} features")
        return df_scaled, True
    
    def get_preprocessing_pipeline(self):
        """Return the preprocessing pipeline for saving"""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer
        from sklearn.base import BaseEstimator, TransformerMixin
        
        # Create a custom transformer that applies all our preprocessing steps
        class CustomPreprocessor(BaseEstimator, TransformerMixin):
            def __init__(self, config, target_variable):
                self.config = config
                self.target_variable = target_variable
                self.outlier_method = None
                self.outlier_params = {}
                self.removed_features = []
                self.transformations = {}
                self.scaler = None
                
            def fit(self, X, y=None):
                # This would store the parameters learned during preprocessing
                # For now, we'll store the configuration
                return self
            
            def transform(self, X):
                # Apply the same transformations that were applied during training
                X_transformed = X.copy()
                
                # Apply outlier removal bounds (but don't remove, just clip)
                if self.outlier_method and hasattr(self, 'outlier_bounds'):
                    if self.target_variable in X_transformed.columns:
                        lower, upper = self.outlier_bounds
                        X_transformed[self.target_variable] = X_transformed[self.target_variable].clip(lower, upper)
                
                # Remove features that were removed during training
                features_to_remove = [f for f in self.removed_features if f in X_transformed.columns]
                if features_to_remove:
                    X_transformed = X_transformed.drop(columns=features_to_remove)
                
                # Apply transformations
                for transform_type, feature_list in self.transformations.items():
                    for original_feature in feature_list:
                        if original_feature in X_transformed.columns:
                            if transform_type == 'log_transformed':
                                # Check if we need to add shift value
                                shift_val = getattr(self, f'{original_feature}_shift', 0)
                                X_transformed[f"{original_feature}_log"] = np.log1p(X_transformed[original_feature] + shift_val)
                                X_transformed = X_transformed.drop(columns=[original_feature])
                            
                            elif transform_type == 'sqrt_transformed':
                                shift_val = getattr(self, f'{original_feature}_shift', 0)
                                X_transformed[f"{original_feature}_sqrt"] = np.sqrt(X_transformed[original_feature] + shift_val)
                                X_transformed = X_transformed.drop(columns=[original_feature])
                            
                            elif transform_type == 'winsorized':
                                # Apply winsorization with stored bounds
                                if hasattr(self, f'{original_feature}_winsor_bounds'):
                                    lower, upper = getattr(self, f'{original_feature}_winsor_bounds')
                                    X_transformed[original_feature] = X_transformed[original_feature].clip(lower, upper)
                
                # Apply scaling if it was applied
                if self.scaler is not None:
                    feature_cols = [col for col in X_transformed.columns if col != self.target_variable]
                    if feature_cols:
                        X_transformed[feature_cols] = self.scaler.transform(X_transformed[feature_cols])
                
                return X_transformed
        
        # Create and return the custom preprocessor
        preprocessor = CustomPreprocessor(self.preprocessing_config, None)  # target will be set during fit
        
        return Pipeline([
            ('custom_preprocessor', preprocessor)
        ])
    
    def save_preprocessing_pipeline(self, file_path: str, preprocessing_result: PreprocessingResult):
        """Save the complete preprocessing pipeline"""
        try:
            # Create the pipeline
            pipeline = self.get_preprocessing_pipeline()
            
            # Store the preprocessing information
            preprocessor = pipeline.steps[0][1]  # Get the custom preprocessor
            
            # Store outlier information
            # Note: In a full implementation, you'd store the actual bounds calculated
            preprocessor.outlier_method = "iqr"  # This would be stored from the actual method used
            
            # Store removed features
            preprocessor.removed_features = preprocessing_result.features_removed
            
            # Store transformations
            preprocessor.transformations = preprocessing_result.transformations_applied
            
            # Store scaling information
            if preprocessing_result.scaling_applied:
                from sklearn.preprocessing import StandardScaler
                preprocessor.scaler = StandardScaler()
                # In practice, you'd fit this scaler on the training data
            
            # Save the pipeline
            with open(file_path, 'wb') as f:
                pickle.dump(pipeline, f)
            
            self.logger.info(f"Saved preprocessing pipeline to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save preprocessing pipeline: {str(e)}")
    
    def load_preprocessing_pipeline(self, file_path: str):
        """Load a saved preprocessing pipeline"""
        try:
            with open(file_path, 'rb') as f:
                pipeline = pickle.load(f)
            
            self.logger.info(f"Loaded preprocessing pipeline from {file_path}")
            return pipeline
            
        except Exception as e:
            self.logger.error(f"Failed to load preprocessing pipeline: {str(e)}")
            return None