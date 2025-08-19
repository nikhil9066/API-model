"""
core/preprocessing.py
Minimal preprocessing pipeline that works
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import logging

@dataclass
class PreprocessingResult:
    """Result of preprocessing pipeline"""
    processed_data: pd.DataFrame
    outliers_removed: int
    features_removed: List[str]
    transformations_applied: List[str]
    scaler: Any = None

class PreprocessingPipeline:
    """Minimal preprocessing pipeline"""
    
    def __init__(self, config: Dict, job_id: str, progress_tracker=None, state_manager=None):
        self.config = config
        self.job_id = job_id
        self.progress = progress_tracker
        self.state_manager = state_manager
        self.logger = logging.getLogger(__name__)
        
        print("✅ PreprocessingPipeline initialized")
    
    def run_preprocessing(self, df: pd.DataFrame, target_variable: str) -> PreprocessingResult:
        """Run complete preprocessing pipeline"""
        
        self.logger.info(f"Starting preprocessing for {len(df)} samples, {len(df.columns)} features")
        
        # Initialize results
        outliers_removed = 0
        features_removed = []
        transformations_applied = []
        
        # Step 1: Remove constant and near-constant columns
        if self.progress:
            self.progress.start_sub_step("preprocessing", "constant_removal", "Removing constant columns")
        
        df_clean, removed_constants = self._remove_constant_columns(df, target_variable)
        features_removed.extend(removed_constants)
        
        if removed_constants:
            transformations_applied.append(f"Removed {len(removed_constants)} constant columns")
            self.logger.info(f"Removed constant columns: {removed_constants}")
        
        if self.progress:
            self.progress.complete_sub_step("preprocessing", "constant_removal")
        
        # Step 2: Remove outliers (simplified)
        if self.progress:
            self.progress.start_sub_step("preprocessing", "outlier_removal", "Removing outliers")
        
        df_no_outliers, outliers_count = self._remove_outliers_simple(df_clean, target_variable)
        outliers_removed = outliers_count
        
        if outliers_removed > 0:
            transformations_applied.append(f"Removed {outliers_removed} outlier rows")
            self.logger.info(f"Removed {outliers_removed} outliers")
        
        if self.progress:
            self.progress.complete_sub_step("preprocessing", "outlier_removal")
        
        # Step 3: Remove highly correlated features
        if self.progress:
            self.progress.start_sub_step("preprocessing", "correlation_removal", "Removing correlated features")
        
        df_decorr, removed_corr = self._remove_highly_correlated(df_no_outliers, target_variable)
        features_removed.extend(removed_corr)
        
        if removed_corr:
            transformations_applied.append(f"Removed {len(removed_corr)} highly correlated features")
            self.logger.info(f"Removed correlated features: {removed_corr[:5]}...")  # Show first 5
        
        if self.progress:
            self.progress.complete_sub_step("preprocessing", "correlation_removal")
        
        # Step 4: Handle missing values
        if self.progress:
            self.progress.start_sub_step("preprocessing", "missing_values", "Handling missing values")
        
        df_final = self._handle_missing_values(df_decorr)
        transformations_applied.append("Handled missing values")
        
        if self.progress:
            self.progress.complete_sub_step("preprocessing", "missing_values")
        
        self.logger.info(f"Preprocessing completed: {len(df_final)} samples, {len(df_final.columns)} features")
        
        return PreprocessingResult(
            processed_data=df_final,
            outliers_removed=outliers_removed,
            features_removed=features_removed,
            transformations_applied=transformations_applied
        )
    
    def _remove_constant_columns(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, List[str]]:
        """Remove constant and near-constant columns"""
        removed_columns = []
        
        for col in df.columns:
            if col == target_variable:
                continue
                
            # Check if column is constant or near-constant
            nunique = df[col].nunique()
            if nunique <= 1:
                removed_columns.append(col)
            elif nunique == 2 and len(df) > 100:
                # Check if one value dominates (>95%)
                value_counts = df[col].value_counts()
                if value_counts.iloc[0] / len(df) > 0.95:
                    removed_columns.append(col)
        
        df_clean = df.drop(columns=removed_columns)
        return df_clean, removed_columns
    
    def _remove_outliers_simple(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, int]:
        """Remove outliers using simple percentile method"""
        
        original_length = len(df)
        
        # Focus on numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_variable]
        
        if not numeric_cols:
            return df, 0
        
        # Use percentile method for outlier detection
        outlier_mask = pd.Series([False] * len(df))
        
        for col in numeric_cols[:10]:  # Limit to first 10 numeric columns for speed
            try:
                Q1 = df[col].quantile(0.01)  # Use 1st and 99th percentiles
                Q3 = df[col].quantile(0.99)
                
                # Mark rows with extreme outliers
                col_outliers = (df[col] < Q1) | (df[col] > Q3)
                outlier_mask = outlier_mask | col_outliers
                
            except Exception:
                continue
        
        # Remove rows that are outliers in multiple columns
        df_clean = df[~outlier_mask]
        outliers_removed = original_length - len(df_clean)
        
        return df_clean, outliers_removed
    
    def _remove_highly_correlated(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features"""
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_variable]
        
        if len(numeric_cols) < 2:
            return df, []
        
        # Calculate correlation matrix
        try:
            corr_matrix = df[numeric_cols].corr().abs()
            
            # Find highly correlated pairs
            removed_features = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:  # 95% correlation threshold
                        col_to_remove = corr_matrix.columns[j]
                        if col_to_remove not in removed_features:
                            removed_features.append(col_to_remove)
            
            # Remove highly correlated features
            df_clean = df.drop(columns=removed_features)
            return df_clean, removed_features
            
        except Exception as e:
            self.logger.warning(f"Correlation analysis failed: {e}")
            return df, []
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with simple imputation"""
        
        df_clean = df.copy()
        
        # For numeric columns, fill with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode().iloc[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)
        
        return df_clean
    
    def save_preprocessing_pipeline(self, file_path: str, result: PreprocessingResult) -> bool:
        """Save preprocessing pipeline (simplified)"""
        try:
            # Just save the result metadata
            metadata = {
                'outliers_removed': result.outliers_removed,
                'features_removed': result.features_removed,
                'transformations_applied': result.transformations_applied,
                'final_shape': result.processed_data.shape
            }
            
            import json
            with open(file_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to save preprocessing pipeline: {e}")
            return False

# Test function
def test_preprocessing():
    """Test preprocessing pipeline"""
    
    # Create sample data with issues
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'constant_col': [1] * 100,  # Constant
        'corr_col': None,  # Will be highly correlated
        'target': np.random.normal(10, 2, 100)
    }
    
    # Make correlated column
    data['corr_col'] = data['feature1'] + np.random.normal(0, 0.01, 100)
    
    # Add some outliers
    data['feature1'][0] = 100  # Extreme outlier
    data['feature2'][1] = -100  # Extreme outlier
    
    df = pd.DataFrame(data)
    
    # Test preprocessing
    config = {}
    pipeline = PreprocessingPipeline(config, "test_job")
    
    print("Testing PreprocessingPipeline...")
    print(f"Original shape: {df.shape}")
    
    result = pipeline.run_preprocessing(df, 'target')
    
    print(f"Final shape: {result.processed_data.shape}")
    print(f"Outliers removed: {result.outliers_removed}")
    print(f"Features removed: {result.features_removed}")
    print(f"Transformations: {result.transformations_applied}")

if __name__ == "__main__":
    test_preprocessing()