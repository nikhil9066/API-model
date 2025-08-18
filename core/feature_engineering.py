"""
core/feature_engineering.py
Automated feature engineering with Auto-sklearn integration
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import logging
from dataclasses import dataclass

# Auto-sklearn imports with error handling
try:
    from autosklearn.experimental.askl2 import AutoSklearnRegressor
    from autosklearn.pipeline.components.feature_preprocessing import *
    AUTOSKLEARN_AVAILABLE = True
except ImportError:
    AUTOSKLEARN_AVAILABLE = False
    logging.warning("Auto-sklearn not available. Using basic feature engineering.")

@dataclass
class FeatureEngineeringResult:
    """Result structure for feature engineering operations"""
    engineered_data: pd.DataFrame
    feature_names: List[str]
    features_created: Dict[str, int]
    feature_importance: Dict[str, float]
    original_feature_count: int
    final_feature_count: int
    engineering_stats: Dict[str, Any]

class FeatureEngineer:
    """Automated feature engineering with multiple strategies"""
    
    def __init__(self, config: Dict, job_id: str, progress_tracker=None, state_manager=None):
        self.config = config
        self.job_id = job_id
        self.progress = progress_tracker
        self.state_manager = state_manager
        self.logger = logging.getLogger(__name__)
        
        # Feature engineering configuration
        self.fe_config = config.get('feature_engineering', {})
        
    def run_feature_engineering(self, df: pd.DataFrame, target_variable: str) -> FeatureEngineeringResult:
        """Run the complete feature engineering pipeline"""
        
        self.logger.info(f"Starting feature engineering for job {self.job_id}")
        original_feature_count = len(df.columns) - 1  # Exclude target
        
        if self.progress:
            self.progress.start_step("feature_engineering", "Automated feature engineering")
        
        df_engineered = df.copy()
        features_created = {}
        
        # Step 1: Auto-sklearn Feature Engineering
        if self.fe_config.get('auto_sklearn', {}).get('enabled', True) and AUTOSKLEARN_AVAILABLE:
            if self.progress:
                self.progress.start_sub_step("feature_engineering", "auto_sklearn", "Auto-sklearn feature generation")
            
            df_engineered, auto_features = self._apply_auto_sklearn_features(df_engineered, target_variable)
            features_created['auto_sklearn'] = auto_features
            
            if self.progress:
                self.progress.complete_sub_step("feature_engineering", "auto_sklearn")
                self.progress.update_step_progress("feature_engineering", f"Created {auto_features} auto-sklearn features")
        
        # Step 2: Polynomial Features
        if self.fe_config.get('polynomial_features', {}).get('enabled', True):
            if self.progress:
                self.progress.start_sub_step("feature_engineering", "polynomial_features", "Creating polynomial features")
            
            df_engineered, poly_features = self._create_polynomial_features(df_engineered, target_variable)
            features_created['polynomial'] = poly_features
            
            if self.progress:
                self.progress.complete_sub_step("feature_engineering", "polynomial_features")
                self.progress.update_step_progress("feature_engineering", f"Created {poly_features} polynomial features")
        
        # Step 3: Interaction Features
        if self.fe_config.get('interaction_features', {}).get('enabled', True):
            if self.progress:
                self.progress.start_sub_step("feature_engineering", "interaction_features", "Creating interaction features")
            
            df_engineered, interaction_features = self._create_interaction_features(df_engineered, target_variable)
            features_created['interaction'] = interaction_features
            
            if self.progress:
                self.progress.complete_sub_step("feature_engineering", "interaction_features")
                self.progress.update_step_progress("feature_engineering", f"Created {interaction_features} interaction features")
        
        # Step 4: Statistical Features
        if self.fe_config.get('statistical_features', {}).get('enabled', True):
            df_engineered, stat_features = self._create_statistical_features(df_engineered, target_variable)
            features_created['statistical'] = stat_features
        
        # Step 5: Feature Selection
        if self.fe_config.get('feature_selection', {}).get('enabled', True):
            if self.progress:
                self.progress.start_sub_step("feature_engineering", "feature_selection", "Selecting best features")
            
            df_engineered, feature_importance = self._apply_feature_selection(df_engineered, target_variable)
            
            if self.progress:
                self.progress.complete_sub_step("feature_engineering", "feature_selection")
                self.progress.update_step_progress("feature_engineering", f"Selected {len(df_engineered.columns)-1} best features")
        else:
            feature_importance = {}
        
        # Create result
        final_feature_count = len(df_engineered.columns) - 1  # Exclude target
        
        result = FeatureEngineeringResult(
            engineered_data=df_engineered,
            feature_names=[col for col in df_engineered.columns if col != target_variable],
            features_created=features_created,
            feature_importance=feature_importance,
            original_feature_count=original_feature_count,
            final_feature_count=final_feature_count,
            engineering_stats={
                'total_features_created': sum(features_created.values()),
                'feature_expansion_ratio': final_feature_count / original_feature_count if original_feature_count > 0 else 1,
                'methods_applied': list(features_created.keys())
            }
        )
        
        # Update state manager
        if self.state_manager:
            self.state_manager.update_feature_engineering_results(self.job_id, {
                'features_created': sum(features_created.values()),
                'original_feature_count': original_feature_count,
                'final_feature_count': final_feature_count,
                'feature_names': result.feature_names,
                'methods_used': list(features_created.keys()),
                'feature_selection_applied': self.fe_config.get('feature_selection', {}).get('enabled', True)
            })
        
        if self.progress:
            self.progress.complete_step("feature_engineering")
        
        self.logger.info(f"Feature engineering completed: {original_feature_count} -> {final_feature_count} features")
        return result
    
    def _apply_auto_sklearn_features(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, int]:
        """Apply auto-sklearn feature engineering"""
        
        if not AUTOSKLEARN_AVAILABLE:
            self.logger.warning("Auto-sklearn not available, skipping auto feature engineering")
            return df, 0
        
        auto_config = self.fe_config.get('auto_sklearn', {})
        time_budget = auto_config.get('time_budget', 300)  # 5 minutes
        memory_limit = auto_config.get('memory_limit', 2048)  # 2GB
        
        try:
            # Separate features and target
            X = df.drop(columns=[target_variable])
            y = df[target_variable]
            
            # Initialize auto-sklearn
            automl = AutoSklearnRegressor(
                time_left_for_this_task=time_budget,
                memory_limit=memory_limit,
                include={'feature_preprocessor': auto_config.get('include_preprocessors', ['polynomial', 'select_percentile'])},
                n_jobs=1,  # Avoid multiprocessing issues
                delete_tmp_folder_after_terminate=True
            )
            
            # Fit auto-sklearn (this will automatically create features)
            automl.fit(X, y)
            
            # Transform the data to get engineered features
            X_transformed = automl.transform(X)
            
            # Create new dataframe with engineered features
            feature_names = [f"auto_feature_{i}" for i in range(X_transformed.shape[1])]
            df_engineered = pd.DataFrame(X_transformed, columns=feature_names, index=df.index)
            df_engineered[target_variable] = y.values
            
            features_created = X_transformed.shape[1] - X.shape[1]
            features_created = max(0, features_created)  # Ensure non-negative
            
            self.logger.info(f"Auto-sklearn created {features_created} features")
            return df_engineered, features_created
            
        except Exception as e:
            self.logger.error(f"Auto-sklearn feature engineering failed: {str(e)}")
            return df, 0
    
    def _create_polynomial_features(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, int]:
        """Create polynomial features"""
        
        poly_config = self.fe_config.get('polynomial_features', {})
        max_degree = poly_config.get('max_degree', 3)
        interaction_only = poly_config.get('interaction_only', False)
        include_bias = poly_config.get('include_bias', False)
        
        # Separate features and target
        X = df.drop(columns=[target_variable])
        y = df[target_variable]
        
    def _create_polynomial_features(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, int]:
        """Create polynomial features"""
        
        poly_config = self.fe_config.get('polynomial_features', {})
        max_degree = poly_config.get('max_degree', 3)
        interaction_only = poly_config.get('interaction_only', False)
        include_bias = poly_config.get('include_bias', False)
        
        # Separate features and target
        X = df.drop(columns=[target_variable])
        y = df[target_variable]
        
        # Limit features to prevent explosion
        if X.shape[1] > 10:
            # Use top 10 features by correlation with target
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            top_features = correlations.head(10).index.tolist()
            X_selected = X[top_features]
        else:
            X_selected = X
        
        try:
            # Create polynomial features
            poly = PolynomialFeatures(
                degree=max_degree,
                interaction_only=interaction_only,
                include_bias=include_bias
            )
            
            X_poly = poly.fit_transform(X_selected)
            
            # Get feature names
            feature_names = poly.get_feature_names_out(X_selected.columns)
            
            # Create dataframe with polynomial features
            df_poly = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
            
            # Add back original features not used in polynomial expansion
            if X.shape[1] > 10:
                unused_features = [col for col in X.columns if col not in top_features]
                for col in unused_features:
                    df_poly[col] = X[col].values
            
            # Add target variable
            df_poly[target_variable] = y.values
            
            features_created = len(feature_names) - X_selected.shape[1]
            
            self.logger.info(f"Created {features_created} polynomial features (degree {max_degree})")
            return df_poly, features_created
            
        except Exception as e:
            self.logger.error(f"Polynomial feature creation failed: {str(e)}")
            return df, 0
    
    def _create_interaction_features(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, int]:
        """Create interaction features"""
        
        interaction_config = self.fe_config.get('interaction_features', {})
        max_interactions = interaction_config.get('max_interactions', 2)
        
        # Separate features and target
        X = df.drop(columns=[target_variable])
        y = df[target_variable]
        
        # Limit to top correlated features to prevent explosion
        if X.shape[1] > 8:
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            top_features = correlations.head(8).index.tolist()
            X_selected = X[top_features]
        else:
            X_selected = X
            top_features = X.columns.tolist()
        
        df_interactions = df.copy()
        features_created = 0
        
        try:
            # Create pairwise interactions
            for i, col1 in enumerate(top_features):
                for j, col2 in enumerate(top_features[i+1:], i+1):
                    if features_created >= max_interactions * 10:  # Limit total interactions
                        break
                    
                    # Multiplication interaction
                    interaction_name = f"{col1}_x_{col2}"
                    df_interactions[interaction_name] = X_selected[col1] * X_selected[col2]
                    features_created += 1
                    
                    # Division interaction (if denominator is not zero)
                    if not (X_selected[col2] == 0).any():
                        div_interaction_name = f"{col1}_div_{col2}"
                        df_interactions[div_interaction_name] = X_selected[col1] / X_selected[col2]
                        features_created += 1
                    
                    # Ratio interaction
                    ratio_interaction_name = f"{col1}_ratio_{col2}"
                    df_interactions[ratio_interaction_name] = X_selected[col1] / (X_selected[col1] + X_selected[col2] + 1e-8)
                    features_created += 1
            
            self.logger.info(f"Created {features_created} interaction features")
            return df_interactions, features_created
            
        except Exception as e:
            self.logger.error(f"Interaction feature creation failed: {str(e)}")
            return df, 0
    
    def _create_statistical_features(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, int]:
        """Create statistical features"""
        
        stat_config = self.fe_config.get('statistical_features', {})
        include_ratios = stat_config.get('include_ratios', True)
        include_differences = stat_config.get('include_differences', True)
        
        # Separate features and target
        X = df.drop(columns=[target_variable])
        y = df[target_variable]
        
        df_stats = df.copy()
        features_created = 0
        
        try:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                # Row-wise statistics
                df_stats['feature_sum'] = X[numeric_cols].sum(axis=1)
                df_stats['feature_mean'] = X[numeric_cols].mean(axis=1)
                df_stats['feature_std'] = X[numeric_cols].std(axis=1)
                df_stats['feature_min'] = X[numeric_cols].min(axis=1)
                df_stats['feature_max'] = X[numeric_cols].max(axis=1)
                df_stats['feature_range'] = df_stats['feature_max'] - df_stats['feature_min']
                features_created += 6
                
                # Additional statistical features
                if include_ratios:
                    df_stats['max_to_mean_ratio'] = df_stats['feature_max'] / (df_stats['feature_mean'] + 1e-8)
                    df_stats['min_to_mean_ratio'] = df_stats['feature_min'] / (df_stats['feature_mean'] + 1e-8)
                    features_created += 2
                
                if include_differences:
                    df_stats['max_min_diff'] = df_stats['feature_max'] - df_stats['feature_min']
                    df_stats['sum_mean_diff'] = df_stats['feature_sum'] - df_stats['feature_mean'] * len(numeric_cols)
                    features_created += 2
            
            self.logger.info(f"Created {features_created} statistical features")
            return df_stats, features_created
            
        except Exception as e:
            self.logger.error(f"Statistical feature creation failed: {str(e)}")
            return df, 0
    
    def _apply_feature_selection(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Apply feature selection to keep only the best features"""
        
        selection_config = self.fe_config.get('feature_selection', {})
        method = selection_config.get('method', 'auto')
        k_best = selection_config.get('k_best', 20)
        
        # Separate features and target
        X = df.drop(columns=[target_variable])
        y = df[target_variable]
        
        if X.shape[1] <= k_best:
            # If we have fewer features than k_best, return all with importance scores
            try:
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                rf.fit(X, y)
                feature_importance = dict(zip(X.columns, rf.feature_importances_))
                return df, feature_importance
            except:
                return df, {}
        
        try:
            feature_importance = {}
            
            if method == 'auto' or method == 'select_k_best':
                # Use SelectKBest with f_regression
                selector = SelectKBest(score_func=f_regression, k=min(k_best, X.shape[1]))
                X_selected = selector.fit_transform(X, y)
                
                # Get selected feature names
                selected_features = X.columns[selector.get_support()].tolist()
                
                # Get feature scores
                scores = selector.scores_
                feature_importance = dict(zip(X.columns, scores))
                
            elif method == 'rfe':
                # Use Recursive Feature Elimination
                estimator = LinearRegression()
                selector = RFE(estimator, n_features_to_select=min(k_best, X.shape[1]))
                X_selected = selector.fit_transform(X, y)
                
                # Get selected feature names
                selected_features = X.columns[selector.get_support()].tolist()
                
                # Get feature importance from the estimator
                if hasattr(selector.estimator_, 'coef_'):
                    importance_scores = np.abs(selector.estimator_.coef_)
                    feature_importance = dict(zip(selected_features, importance_scores))
                
            else:
                # Fallback to random forest feature importance
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                # Get feature importances
                importances = rf.feature_importances_
                feature_importance = dict(zip(X.columns, importances))
                
                # Select top k features
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                selected_features = [feat[0] for feat in sorted_features[:min(k_best, len(sorted_features))]]
            
            # Create dataframe with selected features
            df_selected = df[selected_features + [target_variable]].copy()
            
            self.logger.info(f"Selected {len(selected_features)} features using {method}")
            return df_selected, feature_importance
            
        except Exception as e:
            self.logger.error(f"Feature selection failed: {str(e)}")
            return df, {}
    
    def get_feature_names(self, df: pd.DataFrame, target_variable: str) -> List[str]:
        """Get list of feature names (excluding target)"""
        return [col for col in df.columns if col != target_variable]
    
    def save_feature_engineering_pipeline(self, file_path: str, feature_result: FeatureEngineeringResult):
        """Save the feature engineering pipeline for reuse"""
        from sklearn.pipeline import Pipeline
        from sklearn.base import BaseEstimator, TransformerMixin
        
        class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
            def __init__(self, config, feature_result):
                self.config = config
                self.feature_result = feature_result
                self.polynomial_transformer = None
                self.feature_selector = None
                self.original_features = None
                self.final_features = feature_result.feature_names
                
            def fit(self, X, y=None):
                self.original_features = X.columns.tolist()
                
                # If polynomial features were created, recreate the transformer
                if self.config.get('polynomial_features', {}).get('enabled', True):
                    from sklearn.preprocessing import PolynomialFeatures
                    poly_config = self.config.get('polynomial_features', {})
                    self.polynomial_transformer = PolynomialFeatures(
                        degree=poly_config.get('max_degree', 3),
                        interaction_only=poly_config.get('interaction_only', False),
                        include_bias=poly_config.get('include_bias', False)
                    )
                    
                    # Fit on subset of features (top 10 by correlation)
                    if len(X.columns) > 10:
                        # In practice, you'd store which features were used
                        feature_subset = X.columns[:10]  # Simplified
                    else:
                        feature_subset = X.columns
                    
                    self.polynomial_transformer.fit(X[feature_subset])
                
                return self
            
            def transform(self, X):
                X_transformed = X.copy()
                
                # Apply polynomial features if they were created
                if self.polynomial_transformer is not None:
                    try:
                        # Get the features that were used for polynomial expansion
                        if len(X.columns) > 10:
                            feature_subset = X.columns[:10]  # Simplified
                        else:
                            feature_subset = X.columns
                        
                        X_poly = self.polynomial_transformer.transform(X[feature_subset])
                        feature_names = self.polynomial_transformer.get_feature_names_out(feature_subset)
                        
                        # Create dataframe with polynomial features
                        df_poly = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
                        
                        # Add back unused features
                        unused_features = [col for col in X.columns if col not in feature_subset]
                        for col in unused_features:
                            df_poly[col] = X[col]
                        
                        X_transformed = df_poly
                        
                    except Exception as e:
                        # If polynomial transformation fails, continue with original
                        pass
                
                # Apply interaction features (simplified implementation)
                if self.config.get('interaction_features', {}).get('enabled', True):
                    try:
                        # Create simple interactions between top features
                        numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns[:5]  # Top 5
                        
                        for i, col1 in enumerate(numeric_cols):
                            for col2 in numeric_cols[i+1:]:
                                if f"{col1}_x_{col2}" not in X_transformed.columns:
                                    X_transformed[f"{col1}_x_{col2}"] = X_transformed[col1] * X_transformed[col2]
                    except Exception as e:
                        pass
                
                # Apply statistical features
                if self.config.get('statistical_features', {}).get('enabled', True):
                    try:
                        numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) >= 2:
                            X_transformed['feature_sum'] = X_transformed[numeric_cols].sum(axis=1)
                            X_transformed['feature_mean'] = X_transformed[numeric_cols].mean(axis=1)
                            X_transformed['feature_std'] = X_transformed[numeric_cols].std(axis=1)
                    except Exception as e:
                        pass
                
                # Apply feature selection (keep only final features)
                if self.final_features:
                    available_features = [f for f in self.final_features if f in X_transformed.columns]
                    X_transformed = X_transformed[available_features]
                
                return X_transformed
        
        try:
            # Create the feature engineering pipeline
            fe_pipeline = FeatureEngineeringPipeline(self.fe_config, feature_result)
            
            # Wrap in sklearn Pipeline
            pipeline = Pipeline([
                ('feature_engineering', fe_pipeline)
            ])
            
            # Save pipeline information
            pipeline_info = {
                'pipeline': pipeline,
                'config': self.fe_config,
                'feature_names': feature_result.feature_names,
                'original_feature_count': feature_result.original_feature_count,
                'final_feature_count': feature_result.final_feature_count,
                'features_created': feature_result.features_created
            }
            
            # Save to file
            with open(file_path, 'wb') as f:
                pickle.dump(pipeline_info, f)
            
            self.logger.info(f"Saved feature engineering pipeline to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save feature engineering pipeline: {str(e)}")
    
    def load_feature_engineering_pipeline(self, file_path: str):
        """Load a saved feature engineering pipeline"""
        try:
            with open(file_path, 'rb') as f:
                pipeline_info = pickle.load(f)
            
            self.logger.info(f"Loaded feature engineering pipeline from {file_path}")
            return pipeline_info['pipeline']
            
        except Exception as e:
            self.logger.error(f"Failed to load feature engineering pipeline: {str(e)}")
            return None
    
    def apply_saved_pipeline(self, df: pd.DataFrame, pipeline_file: str) -> pd.DataFrame:
        """Apply a saved feature engineering pipeline to new data"""
        pipeline = self.load_feature_engineering_pipeline(pipeline_file)
        
        if pipeline is None:
            self.logger.warning("Could not load pipeline, returning original data")
            return df
        
        try:
            # Fit and transform the data
            transformed_data = pipeline.fit_transform(df)
            
            if isinstance(transformed_data, np.ndarray):
                # Convert back to DataFrame if needed
                transformed_data = pd.DataFrame(transformed_data, index=df.index)
            
            self.logger.info("Applied saved feature engineering pipeline")
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Failed to apply saved pipeline: {str(e)}")
            return df