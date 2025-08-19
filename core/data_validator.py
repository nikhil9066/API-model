"""
core/data_validator.py
Comprehensive data validation for AutoML pipeline
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import logging

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    dataset_profile: Dict[str, Any]

@dataclass
class DatasetProfile:
    """Profile of a dataset"""
    num_rows: int
    num_cols: int
    file_size_mb: float
    memory_usage_mb: float
    column_types: Dict[str, str]
    missing_data: Dict[str, float]
    numeric_summaries: Dict[str, Dict[str, float]]
    categorical_summaries: Dict[str, Dict[str, Any]]
    correlation_issues: List[str]
    outlier_candidates: List[str]
    constant_columns: List[str]
    high_cardinality_columns: List[str]
    id_like_columns: List[str]

class DataValidator:
    """
    Comprehensive data validator for AutoML pipeline
    Validates file format, data quality, and provides recommendations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validator with configuration
        
        Args:
            config: Configuration dictionary with validation parameters
        """
        self.config = config
        self.max_file_size_mb = config.get('max_file_size_mb', 100)
        self.supported_formats = config.get('supported_formats', ['.csv', '.xlsx', '.json'])
        self.max_missing_percentage = config.get('max_missing_percentage', 50)
        self.min_samples = config.get('min_samples', 10)
        
        logger.info(f"DataValidator initialized with config: {config}")
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Validate file and return comprehensive validation result
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        suggestions = []
        
        try:
            # Step 1: File existence and format validation
            if not self._validate_file_existence(file_path):
                errors.append(f"File not found: {file_path}")
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    suggestions=suggestions,
                    dataset_profile={}
                )
            
            # Step 2: File size validation
            file_size_mb = self._get_file_size_mb(file_path)
            if file_size_mb > self.max_file_size_mb:
                warnings.append(f"Large file: {file_size_mb:.1f}MB (max recommended: {self.max_file_size_mb}MB)")
            
            # Step 3: File format validation
            if not self._validate_file_format(file_path):
                errors.append(f"Unsupported file format. Supported: {self.supported_formats}")
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    suggestions=suggestions,
                    dataset_profile={}
                )
            
            # Step 4: Load and validate data
            try:
                df = self._load_dataframe(file_path)
            except Exception as e:
                errors.append(f"Failed to load file: {str(e)}")
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    suggestions=suggestions,
                    dataset_profile={}
                )
            
            # Step 5: Data quality validation
            data_errors, data_warnings, data_suggestions = self._validate_data_quality(df)
            errors.extend(data_errors)
            warnings.extend(data_warnings)
            suggestions.extend(data_suggestions)
            
            # Step 6: Create dataset profile
            dataset_profile = self._create_dataset_profile(df, file_size_mb)
            
            # Determine if validation passed
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                dataset_profile=dataset_profile
            )
            
        except Exception as e:
            logger.error(f"Validation failed with error: {str(e)}")
            errors.append(f"Validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                dataset_profile={}
            )
    
    def _validate_file_existence(self, file_path: str) -> bool:
        """Check if file exists"""
        return os.path.exists(file_path)
    
    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB"""
        return os.path.getsize(file_path) / (1024 * 1024)
    
    def _validate_file_format(self, file_path: str) -> bool:
        """Validate file format"""
        file_extension = Path(file_path).suffix.lower()
        return file_extension in self.supported_formats
    
    def _load_dataframe(self, file_path: str) -> pd.DataFrame:
        """
        Load file into pandas DataFrame
        
        Args:
            file_path: Path to the file
            
        Returns:
            pandas DataFrame
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.csv':
            # Try different encodings and separators
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                for sep in [',', ';', '\t', '|']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                        if len(df.columns) > 1:  # Valid if more than 1 column
                            return df
                    except Exception:
                        continue
            
            # Fallback: try with default parameters
            return pd.read_csv(file_path)
            
        elif file_extension == '.xlsx':
            return pd.read_excel(file_path)
            
        elif file_extension == '.json':
            return pd.read_json(file_path)
            
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _validate_data_quality(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """
        Validate data quality and return errors, warnings, suggestions
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (errors, warnings, suggestions)
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Basic structure validation
        if df.empty:
            errors.append("Dataset is empty")
            return errors, warnings, suggestions
        
        if len(df) < self.min_samples:
            errors.append(f"Too few samples: {len(df)} (minimum: {self.min_samples})")
        
        if len(df.columns) < 2:
            errors.append("Need at least 2 columns (features + target)")
        
        # Check for excessive missing data
        missing_percentages = (df.isnull().sum() / len(df)) * 100
        highly_missing = missing_percentages[missing_percentages > self.max_missing_percentage]
        
        if len(highly_missing) > 0:
            warnings.append(f"Columns with >50% missing data: {list(highly_missing.index)}")
        
        # Check for constant columns
        constant_columns = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_columns.append(col)
        
        if constant_columns:
            warnings.append(f"Constant columns detected: {constant_columns}")
            suggestions.append("Consider removing constant columns as they provide no predictive value")
        
        # Check for potential ID columns
        id_like_columns = []
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.95 and len(df) > 20:  # >95% unique values
                id_like_columns.append(col)
        
        if id_like_columns:
            warnings.append(f"Potential ID columns detected: {id_like_columns}")
            suggestions.append("ID columns should typically not be used as features or targets")
        
        # Check data types
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            warnings.append(f"Non-numeric columns detected: {non_numeric_cols}")
            suggestions.append("Phase 1 is optimized for numeric data. Consider using Phase 2 for mixed data types")
        
        # Check for highly skewed data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        highly_skewed = []
        for col in numeric_cols:
            try:
                skewness = abs(df[col].skew())
                if skewness > 2:  # Highly skewed
                    highly_skewed.append(col)
            except:
                pass
        
        if highly_skewed:
            warnings.append(f"Highly skewed columns: {highly_skewed}")
            suggestions.append("Consider log transformation or other normalization techniques")
        
        # Check for potential outliers
        outlier_cols = []
        for col in numeric_cols:
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                    outlier_cols.append(col)
            except:
                pass
        
        if outlier_cols:
            warnings.append(f"Columns with many outliers: {outlier_cols}")
            suggestions.append("Outlier detection and removal may improve model performance")
        
        return errors, warnings, suggestions
    
    def _create_dataset_profile(self, df: pd.DataFrame, file_size_mb: float) -> Dict[str, Any]:
        """
        Create comprehensive dataset profile
        
        Args:
            df: DataFrame to profile
            file_size_mb: File size in MB
            
        Returns:
            Dictionary with dataset profile information
        """
        try:
            # Basic info
            num_rows, num_cols = df.shape
            memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            
            # Column types
            column_types = {col: str(df[col].dtype) for col in df.columns}
            
            # Missing data analysis
            missing_data = {}
            for col in df.columns:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                missing_data[col] = round(missing_pct, 2)
            
            # Numeric summaries
            numeric_summaries = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                try:
                    numeric_summaries[col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'median': float(df[col].median()),
                        'skewness': float(df[col].skew()),
                        'unique_count': int(df[col].nunique())
                    }
                except:
                    numeric_summaries[col] = {'error': 'Could not compute statistics'}
            
            # Categorical summaries
            categorical_summaries = {}
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                try:
                    value_counts = df[col].value_counts().head(10)
                    categorical_summaries[col] = {
                        'unique_count': int(df[col].nunique()),
                        'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                        'top_values': value_counts.to_dict()
                    }
                except:
                    categorical_summaries[col] = {'error': 'Could not compute statistics'}
            
            # Identify problem columns
            constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
            
            high_cardinality_columns = [
                col for col in df.columns 
                if df[col].nunique() > min(50, len(df) * 0.1)
            ]
            
            id_like_columns = [
                col for col in df.columns 
                if df[col].nunique() / len(df) > 0.95 and len(df) > 20
            ]
            
            # Correlation issues (for numeric columns)
            correlation_issues = []
            if len(numeric_cols) > 1:
                try:
                    corr_matrix = df[numeric_cols].corr()
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = abs(corr_matrix.iloc[i, j])
                            if corr_val > 0.9:  # Highly correlated
                                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                                correlation_issues.append(f"{col1} <-> {col2}: {corr_val:.3f}")
                except:
                    pass
            
            # Outlier candidates
            outlier_candidates = []
            for col in numeric_cols:
                try:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outlier_pct = (len(outliers) / len(df)) * 100
                    if outlier_pct > 5:  # More than 5% outliers
                        outlier_candidates.append(f"{col}: {outlier_pct:.1f}%")
                except:
                    pass
            
            return {
                'num_rows': num_rows,
                'num_cols': num_cols,
                'file_size_mb': round(file_size_mb, 2),
                'memory_usage_mb': round(memory_usage_mb, 2),
                'column_types': column_types,
                'missing_data': missing_data,
                'numeric_summaries': numeric_summaries,
                'categorical_summaries': categorical_summaries,
                'constant_columns': constant_columns,
                'high_cardinality_columns': high_cardinality_columns,
                'id_like_columns': id_like_columns,
                'correlation_issues': correlation_issues,
                'outlier_candidates': outlier_candidates
            }
            
        except Exception as e:
            logger.error(f"Failed to create dataset profile: {str(e)}")
            return {
                'error': f"Failed to create profile: {str(e)}",
                'num_rows': len(df) if df is not None else 0,
                'num_cols': len(df.columns) if df is not None else 0
            }
    
    def validate_target_column(self, df: pd.DataFrame, target_column: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate if a column is suitable as a target variable
        
        Args:
            df: DataFrame containing the data
            target_column: Name of the target column
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check if column exists
        if target_column not in df.columns:
            errors.append(f"Target column '{target_column}' not found in dataset")
            return False, errors, warnings
        
        target_series = df[target_column]
        
        # Check for excessive missing values
        missing_pct = (target_series.isnull().sum() / len(target_series)) * 100
        if missing_pct > 50:
            errors.append(f"Target column has {missing_pct:.1f}% missing values (too high)")
        elif missing_pct > 20:
            warnings.append(f"Target column has {missing_pct:.1f}% missing values")
        
        # Check if column is constant
        if target_series.nunique() <= 1:
            errors.append("Target column has no variance (constant values)")
            return False, errors, warnings
        
        # Check if it looks like an ID column
        unique_ratio = target_series.nunique() / len(target_series)
        if unique_ratio > 0.95 and len(target_series) > 20:
            warnings.append("Target column appears to be an ID (too many unique values)")
        
        # Check data type
        if not pd.api.types.is_numeric_dtype(target_series):
            warnings.append("Target column is not numeric - Phase 1 is optimized for numeric targets")
        
        # Check for extreme values
        if pd.api.types.is_numeric_dtype(target_series):
            try:
                Q1 = target_series.quantile(0.25)
                Q3 = target_series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # 3 IQR for extreme outliers
                upper_bound = Q3 + 3 * IQR
                extreme_outliers = target_series[(target_series < lower_bound) | (target_series > upper_bound)]
                
                if len(extreme_outliers) > 0:
                    warnings.append(f"Target has {len(extreme_outliers)} extreme outliers")
                
                # Check skewness
                skewness = abs(target_series.skew())
                if skewness > 2:
                    warnings.append(f"Target is highly skewed (skewness: {skewness:.2f})")
                    
            except:
                pass
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings

# Example usage and testing
def test_data_validator():
    """Test the DataValidator with sample data"""
    
    # Create test configuration
    config = {
        'max_file_size_mb': 100,
        'supported_formats': ['.csv', '.xlsx', '.json'],
        'max_missing_percentage': 50,
        'min_samples': 10
    }
    
    # Initialize validator
    validator = DataValidator(config)
    
    # Create sample test data
    import tempfile
    import os
    
    # Create a temporary CSV file for testing
    test_data = {
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.exponential(2, 100),  # Skewed
        'feature3': [1] * 100,  # Constant
        'id_column': range(100),  # ID-like
        'target': np.random.normal(10, 3, 100)
    }
    
    # Add some missing values
    test_data['feature1'][10:20] = np.nan
    
    df = pd.DataFrame(test_data)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        # Test validation
        result = validator.validate_file(temp_file)
        
        print("Validation Result:")
        print(f"Valid: {result.is_valid}")
        print(f"Errors: {result.errors}")
        print(f"Warnings: {result.warnings}")
        print(f"Suggestions: {result.suggestions}")
        print(f"Dataset Profile Keys: {list(result.dataset_profile.keys())}")
        
        # Test target validation
        if result.is_valid:
            df_loaded = validator._load_dataframe(temp_file)
            
            # Test good target
            valid, errors, warnings = validator.validate_target_column(df_loaded, 'target')
            print(f"\nTarget 'target' validation: {valid}")
            print(f"Errors: {errors}")
            print(f"Warnings: {warnings}")
            
            # Test bad target (constant)
            valid, errors, warnings = validator.validate_target_column(df_loaded, 'feature3')
            print(f"\nTarget 'feature3' validation: {valid}")
            print(f"Errors: {errors}")
            print(f"Warnings: {warnings}")
        
    finally:
        # Clean up
        os.unlink(temp_file)

if __name__ == "__main__":
    test_data_validator()