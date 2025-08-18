"""
core/preprocessing.py
Enhanced preprocessing pipeline for Phase 1
Migrated and enhanced from your original datapipeline.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class PreprocessingResults:
    """Results from preprocessing pipeline"""
    processed_df: pd.DataFrame
    preprocessing_summary: Dict[str, Any]
    removed_outliers: int
    removed_features: List[str]
    transformations_applied: Dict[str, List[str]]
    scaling_applied: str

class PreprocessingPipeline:
    """Enhanced preprocessing pipeline for numeric data"""
    
    def __init__(self, config: Dict, progress_tracker=None):
        self.config = config
        self.progress_tracker = progress_tracker
        self.logger = logging.getLogger(__name__)
        
        # Initialize preprocessing components
        self.outlier_config = config.get('preprocessing', {}).get('outlier_detection', {})
        self.correlation_config = config.get('preprocessing', {}).get('correlation', {})
        self.skewness_config = config.get('preprocessing', {}).get('skewness', {})
        self.scaling_config = config.get('preprocessing', {}).get('scaling', {})
        
    def fit_transform(self, df: pd.DataFrame, target_variable: str) -> PreprocessingResults:
        """
        Complete preprocessing pipeline
        """
        self.logger.info("Starting preprocessing pipeline")
        
        # Initialize results tracking
        preprocessing_summary = {
            'original_shape': df.shape,
            'steps_applied': [],
            'outlier_method_used': None,
            'features_removed': [],
            'transformations': {},
            'scaling_method': None
        }
        
        processed_df = df.copy()
        
        # Step 1: Remove constant and near-constant columns
        if self.progress_tracker:
            self.progress_tracker.start_sub_step("preprocessing", "constant_removal", "Removing constant columns")
        
        processed_df, constant_removed = self._remove_constant_columns(processed_df, target_variable)
        preprocessing_summary['constant_columns_removed'] = constant_removed
        preprocessing_summary['steps_applied'].append('constant_removal')
        
        if self.progress_tracker:
            self.progress_tracker.complete_sub_step("preprocessing", "constant_removal")
        
        # Step 2: Outlier detection and removal
        if self.progress_tracker:
            self.progress_tracker.start_sub_step("preprocessing", "outlier_detection", "Detecting and removing outliers")
        
        processed_df, outliers_removed, method_used = self._handle_outliers(processed_df, target_variable)
        preprocessing_summary['outliers_removed'] = outliers_removed
        preprocessing_summary['outlier_method_used'] = method_used
        preprocessing_summary['steps_applied'].append('outlier_removal')
        
        if self.progress_tracker:
            self.progress_tracker.complete_sub_step("preprocessing", "outlier_detection")
        
        # Step 3: Correlation-based feature removal
        if self.progress_tracker:
            self.progress_tracker.start_sub_step("preprocessing", "correlation_analysis", "Analyzing feature correlations")
        
        processed_df, removed_features = self._handle_correlations(processed_df, target_variable)
        preprocessing_summary['features_removed'] = removed_features
        preprocessing_summary['steps_applied'].append('correlation_filtering')
        
        if self.progress_tracker:
            self.progress_tracker.complete_sub_step("preprocessing", "correlation_analysis")
        
        # Step 4: Skewness handling
        if self.progress_tracker:
            self.progress_tracker.start_sub_step("preprocessing", "skewness_handling", "Handling skewed features")
        
        processed_df, transformations_applied = self._handle_skewness(processed_df, target_variable)
        preprocessing_summary['transformations'] = transformations_applied
        preprocessing_summary['steps_applied'].append('skewness_handling')
        
        if self.progress_tracker:
            self.progress_tracker.complete_sub_step("preprocessing", "skewness_handling")
        
        preprocessing_summary['final_shape'] = processed_df.shape
        
        # Create results object
        results = PreprocessingResults(
            processed_df=processed_df,
            preprocessing_summary=preprocessing_summary,
            removed_outliers=outliers_removed,
            removed_features=removed_features,
            transformations_applied=transformations_applied,
            scaling_applied=None  # Scaling will be done during model training
        )
        
        self.logger.info(f"Preprocessing completed. Shape: {df.shape} → {processed_df.shape}")
        return results
    
    def _remove_constant_columns(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, List[str]]:
        """Remove constant and near-constant columns"""
        constant_cols = []
        
        for col in df.columns:
            if col == target_variable:
                continue
                
            # Check for constant values
            if df[col].nunique() <= 1:
                constant_cols.append(col)
                continue
            
            # Check for near-constant (variance threshold)
            if df[col].var() < 1e-10:
                constant_cols.append(col)
        
        if constant_cols:
            df_cleaned = df.drop(columns=constant_cols)
            self.logger.info(f"Removed {len(constant_cols)} constant columns: {constant_cols}")
        else:
            df_cleaned = df.copy()
        
        return df_cleaned, constant_cols
    
    def _handle_outliers(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, int, str]:
        """
        Enhanced outlier detection using multiple methods
        Migrated from your original remove_outliers logic
        """
        methods = {
            'iqr': self._remove_outliers_iqr,
            'sd3': self._remove_outliers_sd3,
            'percentile': self._remove_outliers_percentile
        }
        
        if self.outlier_config.get('auto_select', True):
            # Try all methods and select the one that removes the most outliers
            method_results = {}
            original_len = len(df)
            
            for method_name, method_func in methods.items():
                try:
                    df_cleaned = method_func(df, target_variable)
                    outliers_removed = original_len - len(df_cleaned)
                    method_results[method_name] = (df_cleaned, outliers_removed)
                    
                    if self.progress_tracker:
                        self.progress_tracker.update_step_progress(
                            "preprocessing", 
                            f"{method_name.upper()}: {outliers_removed} outliers detected"
                        )
                except Exception as e:
                    self.logger.warning(f"Method {method_name} failed: {e}")
                    method_results[method_name] = (df, 0)
            
            # Select method that removes the most outliers
            best_method = max(method_results.keys(), key=lambda k: method_results[k][1])
            df_cleaned, outliers_removed = method_results[best_method]
            
            self.logger.info(f"Selected {best_method} method: removed {outliers_removed} outliers")
            
        else:
            # Use specified method
            method_name = self.outlier_config.get('methods', ['iqr'])[0]
            method_func = methods.get(method_name, methods['iqr'])
            original_len = len(df)
            df_cleaned = method_func(df, target_variable)
            outliers_removed = original_len - len(df_cleaned)
            best_method = method_name
        
        return df_cleaned, outliers_removed, best_method
    
    def _remove_outliers_iqr(self, df: pd.DataFrame, target_variable: str) -> pd.DataFrame:
        """IQR method for outlier removal"""
        multiplier = self.outlier_config.get('iqr_multiplier', 1.5)
        
        Q1 = df[target_variable].quantile(0.25)
        Q3 = df[target_variable].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        mask = (df[target_variable] >= lower_bound) & (df[target_variable] <= upper_bound)
        return df[mask].copy()
    
    def _remove_outliers_sd3(self, df: pd.DataFrame, target_variable: str) -> pd.DataFrame:
        """Standard deviation method for outlier removal"""
        multiplier = self.outlier_config.get('std_multiplier', 3.0)
        
        mean = df[target_variable].mean()
        std_dev = df[target_variable].std()
        lower_bound = mean - multiplier * std_dev
        upper_bound = mean + multiplier * std_dev
        
        mask = (df[target_variable] >= lower_bound) & (df[target_variable] <= upper_bound)
        return df[mask].copy()
    
    def _remove_outliers_percentile(self, df: pd.DataFrame, target_variable: str) -> pd.DataFrame:
        """Percentile method for outlier removal"""
        bounds = self.outlier_config.get('percentile_bounds', [0.01, 0.99])
        
        lower_percentile = df[target_variable].quantile(bounds[0])
        upper_percentile = df[target_variable].quantile(bounds[1])
        
        mask = (df[target_variable] >= lower_percentile) & (df[target_variable] <= upper_percentile)
        return df[mask].copy()
    
    def _handle_correlations(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Handle highly correlated features
        Migrated from your correlation filtering logic
        """
        removed_features = []
        
        # Remove high correlation with other features
        high_threshold = self.correlation_config.get('high_threshold', 0.9)
        correlation_matrix = df.corr()
        
        # Find highly correlated feature pairs
        high_corr_features = set()
        target_corr = correlation_matrix[target_variable].abs()
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                
                if col1 == target_variable or col2 == target_variable:
                    continue
                
                if abs(correlation_matrix.iloc[i, j]) > high_threshold:
                    # Remove the feature with lower correlation to target
                    if target_corr[col1] < target_corr[col2]:
                        high_corr_features.add(col1)
                    else:
                        high_corr_features.add(col2)
        
        # Remove low correlation with target
        low_threshold = self.correlation_config.get('low_threshold', 0.1)
        low_corr_features = set()
        
        for col in df.columns:
            if col != target_variable and abs(target_corr[col]) < low_threshold:
                low_corr_features.add(col)
        
        # Combine all features to remove
        all_features_to_remove = high_corr_features.union(low_corr_features)
        removed_features = list(all_features_to_remove)
        
        if removed_features:
            df_filtered = df.drop(columns=removed_features)
            self.logger.info(f"Removed {len(removed_features)} features due to correlation: {removed_features}")
            
            if self.progress_tracker:
                self.progress_tracker.update_step_progress(
                    "preprocessing",
                    f"Removed {len(removed_features)} correlated features"
                )
        else:
            df_filtered = df.copy()
        
        return df_filtered, removed_features
    
    def _handle_skewness(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Handle skewed features with multiple transformation methods
        Migrated from your skewness handling logic
        """
        high_threshold = self.skewness_config.get('high_threshold', 1.0)
        moderate_threshold = self.skewness_config.get('moderate_threshold', 0.5)
        
        transformations_applied = {
            'log_transformed': [],
            'sqrt_transformed': [],
            'boxcox_transformed': [],
            'yeo_johnson_transformed': [],
            'winsorized': []
        }
        
        df_transformed = df.copy()
        
        # Categorize features by skewness
        skew_values = df_transformed.select_dtypes(include=[np.number]).skew()
        
        highly_skewed = [col for col in skew_values.index 
                        if col != target_variable and abs(skew_values[col]) > high_threshold]
        moderately_skewed = [col for col in skew_values.index 
                           if col != target_variable and moderate_threshold <= abs(skew_values[col]) <= high_threshold]
        
        # Handle highly skewed features
        for col in highly_skewed:
            df_transformed, transformation = self._transform_highly_skewed_feature(df_transformed, col)
            if transformation:
                transformations_applied[transformation].append(col)
        
        # Handle moderately skewed features with winsorization
        for col in moderately_skewed:
            df_transformed[col] = stats.mstats.winsorize(df_transformed[col], limits=[0.05, 0.05])
            transformations_applied['winsorized'].append(col)
        
        if self.progress_tracker:
            total_transformed = sum(len(features) for features in transformations_applied.values())
            self.progress_tracker.update_step_progress(
                "preprocessing",
                f"Applied transformations to {total_transformed} features"
            )
        
        return df_transformed, transformations_applied
    
    def _transform_highly_skewed_feature(self, df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Apply the best transformation for a highly skewed feature
        """
        original_data = df[col].copy()
        original_skew = abs(original_data.skew())
        
        # Ensure positive values for log and box-cox
        shift_val = 0
        if (original_data <= 0).any():
            shift_val = abs(original_data.min()) + 1
            shifted_data = original_data + shift_val
        else:
            shifted_data = original_data
        
        transformations = {}
        
        # Try different transformations
        try:
            # Log transformation
            if (shifted_data > 0).all():
                log_transformed = np.log1p(shifted_data)
                transformations['log_transformed'] = (log_transformed, abs(log_transformed.skew()))
        except:
            pass
        
        try:
            # Square root transformation
            if (shifted_data >= 0).all():
                sqrt_transformed = np.sqrt(shifted_data)
                transformations['sqrt_transformed'] = (sqrt_transformed, abs(sqrt_transformed.skew()))
        except:
            pass
        
        try:
            # Box-Cox transformation
            if (shifted_data > 0).all():
                from scipy.stats import boxcox
                boxcox_transformed, _ = boxcox(shifted_data)
                transformations['boxcox_transformed'] = (boxcox_transformed, abs(pd.Series(boxcox_transformed).skew()))
        except:
            pass
        
        try:
            # Yeo-Johnson transformation (handles negative values)
            pt = PowerTransformer(method='yeo-johnson')
            yj_transformed = pt.fit_transform(original_data.values.reshape(-1, 1)).flatten()
            transformations['yeo_johnson_transformed'] = (yj_transformed, abs(pd.Series(yj_transformed).skew()))
        except:
            pass
        
        # Select the best transformation (lowest skewness)
        if transformations:
            best_method = min(transformations.keys(), key=lambda k: transformations[k][1])
            best_data, best_skew = transformations[best_method]
            
            # Only apply if it significantly improves skewness
            if best_skew < original_skew * 0.8:  # At least 20% improvement
                df_copy = df.copy()
                df_copy[col] = best_data
                return df_copy, best_method
        
        # Return original if no transformation helps
        return df, None
    
    def get_preprocessing_pipeline_for_inference(self, df: pd.DataFrame, target_variable: str):
        """
        Create a preprocessing pipeline that can be saved and reused for inference
        """
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        
        # This would create a sklearn pipeline for consistent preprocessing
        # Implementation depends on the specific transformations applied
        pass