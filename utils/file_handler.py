"""
utils/file_handler.py
Comprehensive file handling utilities for AutoML pipeline
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class FileHandler:
    """
    Comprehensive file handler for loading, saving, and processing various file formats
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FileHandler with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Supported file formats
        self.supported_formats = {
            '.csv': self._load_csv,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.json': self._load_json,
            '.parquet': self._load_parquet,
            '.pkl': self._load_pickle,
            '.pickle': self._load_pickle
        }
        
        # Default encoding options
        self.encoding_options = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        print("✅ FileHandler initialized successfully")
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats into pandas DataFrame
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            pandas DataFrame
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}. "
                           f"Supported formats: {list(self.supported_formats.keys())}")
        
        self.logger.info(f"Loading file: {file_path}")
        
        try:
            # Load using appropriate method
            loader = self.supported_formats[file_extension]
            df = loader(file_path)
            
            # Basic validation
            if df.empty:
                raise ValueError("Loaded DataFrame is empty")
            
            self.logger.info(f"Successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Basic cleanup
            df = self._basic_cleanup(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {str(e)}")
            raise
    
    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load CSV file with multiple fallback options"""
        
        # Try different combinations of separators and encodings
        separators = [',', ';', '\t', '|']
        
        for encoding in self.encoding_options:
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                    
                    # Check if parsing was successful (more than 1 column usually indicates success)
                    if len(df.columns) > 1 or (len(df.columns) == 1 and len(df) > 0):
                        self.logger.info(f"CSV loaded with encoding={encoding}, separator='{sep}'")
                        return df
                        
                except Exception as e:
                    continue
        
        # Final fallback - try with default pandas parameters
        try:
            df = pd.read_csv(file_path)
            self.logger.info("CSV loaded with default parameters")
            return df
        except Exception as e:
            raise ValueError(f"Could not load CSV file: {str(e)}")
    
    def _load_excel(self, file_path: Path) -> pd.DataFrame:
        """Load Excel file (.xlsx, .xls)"""
        try:
            # Try to load the first sheet
            df = pd.read_excel(file_path, sheet_name=0)
            self.logger.info("Excel file loaded successfully")
            return df
            
        except Exception as e:
            # Try with openpyxl engine for .xlsx files
            if file_path.suffix.lower() == '.xlsx':
                try:
                    df = pd.read_excel(file_path, engine='openpyxl')
                    self.logger.info("Excel file loaded with openpyxl engine")
                    return df
                except:
                    pass
            
            # Try with xlrd engine for .xls files
            if file_path.suffix.lower() == '.xls':
                try:
                    df = pd.read_excel(file_path, engine='xlrd')
                    self.logger.info("Excel file loaded with xlrd engine")
                    return df
                except:
                    pass
            
            raise ValueError(f"Could not load Excel file: {str(e)}")
    
    def _load_json(self, file_path: Path) -> pd.DataFrame:
        """Load JSON file"""
        try:
            # Try loading as pandas JSON
            df = pd.read_json(file_path)
            self.logger.info("JSON file loaded successfully")
            return df
            
        except Exception as e:
            # Try loading as regular JSON and converting
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert to DataFrame
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    raise ValueError("JSON data must be a list or dictionary")
                
                self.logger.info("JSON file loaded and converted to DataFrame")
                return df
                
            except Exception as e2:
                raise ValueError(f"Could not load JSON file: {str(e2)}")
    
    def _load_parquet(self, file_path: Path) -> pd.DataFrame:
        """Load Parquet file"""
        try:
            df = pd.read_parquet(file_path)
            self.logger.info("Parquet file loaded successfully")
            return df
        except Exception as e:
            raise ValueError(f"Could not load Parquet file: {str(e)}")
    
    def _load_pickle(self, file_path: Path) -> pd.DataFrame:
        """Load Pickle file"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, pd.DataFrame):
                self.logger.info("Pickle file loaded successfully")
                return data
            else:
                raise ValueError("Pickle file does not contain a pandas DataFrame")
                
        except Exception as e:
            raise ValueError(f"Could not load Pickle file: {str(e)}")
    
    def _basic_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic cleanup on loaded DataFrame"""
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all')  # Remove rows where all values are NaN
        df = df.dropna(axis=1, how='all')  # Remove columns where all values are NaN
        
        # Strip whitespace from column names
        df.columns = df.columns.astype(str).str.strip()
        
        # Remove duplicate column names by adding suffix
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup 
                                                             for i in range(sum(cols == dup))]
        df.columns = cols
        
        # Strip whitespace from string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].astype(str).str.strip()
            # Replace 'nan' strings with actual NaN
            df[col] = df[col].replace(['nan', 'NaN', 'None', ''], np.nan)
        
        return df
    
    def save_data(self, df: pd.DataFrame, file_path: str, **kwargs) -> bool:
        """
        Save DataFrame to various file formats
        
        Args:
            df: DataFrame to save
            file_path: Path where to save the file
            **kwargs: Additional arguments for pandas save methods
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            file_extension = file_path.suffix.lower()
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_extension == '.csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif file_extension in ['.xlsx', '.xls']:
                df.to_excel(file_path, index=False, **kwargs)
            elif file_extension == '.json':
                df.to_json(file_path, **kwargs)
            elif file_extension == '.parquet':
                df.to_parquet(file_path, **kwargs)
            elif file_extension in ['.pkl', '.pickle']:
                with open(file_path, 'wb') as f:
                    pickle.dump(df, f)
            else:
                raise ValueError(f"Unsupported save format: {file_extension}")
            
            self.logger.info(f"Data saved successfully to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save data to {file_path}: {str(e)}")
            return False
    
    def save_json(self, data: Dict[str, Any], file_path: str, **kwargs) -> bool:
        """
        Save dictionary/object as JSON file
        
        Args:
            data: Data to save as JSON
            file_path: Path where to save the file
            **kwargs: Additional arguments for json.dump
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy types to native Python types for JSON serialization
            data = self._convert_numpy_types(data)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, **kwargs)
            
            self.logger.info(f"JSON saved successfully to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON to {file_path}: {str(e)}")
            return False
    
    def load_json(self, file_path: str) -> Dict[str, Any]:
        """
        Load JSON file as dictionary
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dict: Loaded JSON data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"JSON loaded successfully from: {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load JSON from {file_path}: {str(e)}")
            raise
    
    def save_pickle(self, data: Any, file_path: str) -> bool:
        """
        Save any object using pickle
        
        Args:
            data: Object to save
            file_path: Path where to save the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info(f"Pickle saved successfully to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save pickle to {file_path}: {str(e)}")
            return False
    
    def load_pickle(self, file_path: str) -> Any:
        """
        Load pickle file
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            Any: Loaded object
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            self.logger.info(f"Pickle loaded successfully from: {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load pickle from {file_path}: {str(e)}")
            raise
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return obj
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict: File information
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {'error': 'File not found'}
            
            stat = file_path.stat()
            
            info = {
                'filename': file_path.name,
                'extension': file_path.suffix.lower(),
                'size_bytes': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified_time': stat.st_mtime,
                'is_supported': file_path.suffix.lower() in self.supported_formats
            }
            
            return info
            
        except Exception as e:
            return {'error': str(e)}
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate if file can be loaded and get basic info
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict: Validation result with file info
        """
        result = {
            'is_valid': False,
            'file_info': {},
            'data_info': {},
            'errors': []
        }
        
        try:
            # Get file info
            file_info = self.get_file_info(file_path)
            result['file_info'] = file_info
            
            if 'error' in file_info:
                result['errors'].append(file_info['error'])
                return result
            
            if not file_info['is_supported']:
                result['errors'].append(f"Unsupported file format: {file_info['extension']}")
                return result
            
            # Try to load and get basic data info
            df = self.load_data(file_path)
            
            result['data_info'] = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'missing_data': df.isnull().sum().to_dict(),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            }
            
            result['is_valid'] = True
            
        except Exception as e:
            result['errors'].append(str(e))
        
        return result

# Test function
def test_file_handler():
    """Test the FileHandler with sample data"""
    
    # Create sample data
    import tempfile
    
    sample_data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [1.1, 2.2, 3.3, 4.4, 5.5],
        'target': [10, 20, 30, 40, 50]
    }
    df = pd.DataFrame(sample_data)
    
    # Test FileHandler
    config = {}
    handler = FileHandler(config)
    
    print("Testing FileHandler...")
    
    # Test saving and loading CSV
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        temp_csv = f.name
    
    try:
        # Save CSV
        success = handler.save_data(df, temp_csv)
        print(f"Save CSV: {'✅' if success else '❌'}")
        
        # Load CSV
        df_loaded = handler.load_data(temp_csv)
        print(f"Load CSV: ✅ - Shape: {df_loaded.shape}")
        
        # Validate file
        validation = handler.validate_file(temp_csv)
        print(f"Validate CSV: {'✅' if validation['is_valid'] else '❌'}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        if os.path.exists(temp_csv):
            os.unlink(temp_csv)

if __name__ == "__main__":
    test_file_handler()