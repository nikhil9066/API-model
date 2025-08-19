"""
utils/visualization.py
Visualization utilities for model comparison and results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")

class Visualizer:
    """Comprehensive visualization utilities for AutoML results"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set default figure size and DPI
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        
    def plot_model_comparison(self, results_df: pd.DataFrame, save_path: str = None, 
                            title: str = "Model Performance Comparison") -> str:
        """Create comprehensive model comparison plot"""
        
        if results_df.empty:
            raise ValueError("Results dataframe is empty")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Sort by test score for better visualization
        results_sorted = results_df.sort_values('Test Score', ascending=True)
        
        # 1. Horizontal bar chart of test scores
        ax1 = axes[0, 0]
        bars = ax1.barh(results_sorted['Model'], results_sorted['Test Score'], 
                       color=plt.cm.viridis(np.linspace(0, 1, len(results_sorted))))
        ax1.set_xlabel('Test Score (R²)')
        ax1.set_title('Test Performance Comparison')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, results_sorted['Test Score'])):
            ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontweight='bold')
        
        # 2. Train vs Test scores scatter plot
        ax2 = axes[0, 1]
        scatter = ax2.scatter(results_sorted['Train Score'], results_sorted['Test Score'], 
                            s=100, alpha=0.7, c=range(len(results_sorted)), cmap='viridis')
        
        # Add diagonal line (perfect fit)
        min_score = min(results_sorted['Train Score'].min(), results_sorted['Test Score'].min())
        max_score = max(results_sorted['Train Score'].max(), results_sorted['Test Score'].max())
        ax2.plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.5, label='Perfect Fit')
        
        ax2.set_xlabel('Train Score (R²)')
        ax2.set_ylabel('Test Score (R²)')
        ax2.set_title('Train vs Test Performance')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Add model labels
        for i, model in enumerate(results_sorted['Model']):
            ax2.annotate(model, (results_sorted['Train Score'].iloc[i], 
                               results_sorted['Test Score'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
        
        # 3. Cross-validation scores with error bars
        ax3 = axes[1, 0]
        if 'CV Score' in results_sorted.columns and 'CV Std' in results_sorted.columns:
            x_pos = range(len(results_sorted))
            bars = ax3.bar(x_pos, results_sorted['CV Score'], 
                          yerr=results_sorted['CV Std'], capsize=5,
                          color=plt.cm.plasma(np.linspace(0, 1, len(results_sorted))))
            
            ax3.set_xlabel('Models')
            ax3.set_ylabel('CV Score (R²)')
            ax3.set_title('Cross-Validation Performance')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(results_sorted['Model'], rotation=45, ha='right')
            ax3.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (bar, score, std) in enumerate(zip(bars, results_sorted['CV Score'], results_sorted['CV Std'])):
                ax3.text(bar.get_x() + bar.get_width()/2, score + std + 0.01, 
                        f'{score:.3f}', ha='center', fontweight='bold', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'CV scores not available', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Cross-Validation Performance')
        
        # 4. Training time comparison
        ax4 = axes[1, 1]
        if 'Training Time' in results_sorted.columns:
            bars = ax4.bar(range(len(results_sorted)), results_sorted['Training Time'],
                          color=plt.cm.coolwarm(np.linspace(0, 1, len(results_sorted))))
            
            ax4.set_xlabel('Models')
            ax4.set_ylabel('Training Time (seconds)')
            ax4.set_title('Training Time Comparison')
            ax4.set_xticks(range(len(results_sorted)))
            ax4.set_xticklabels(results_sorted['Model'], rotation=45, ha='right')
            ax4.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (bar, time) in enumerate(zip(bars, results_sorted['Training Time'])):
                ax4.text(bar.get_x() + bar.get_width()/2, time + max(results_sorted['Training Time'])*0.01, 
                        f'{time:.1f}s', ha='center', fontweight='bold', fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'Training times not available', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Training Time Comparison')
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            self.logger.info(f"Model comparison plot saved to {save_path}")
        
        return save_path or "model_comparison_plot"
    
    def plot_top_models_detailed(self, results_df: pd.DataFrame, top_n: int = 5, 
                               save_path: str = None) -> str:
        """Create detailed plot of top N models"""
        
        # Get top N models
        top_models = results_df.nlargest(top_n, 'Test Score')
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Top {top_n} Models - Detailed Analysis', fontsize=16, fontweight='bold')
        
        # 1. Performance comparison
        ax1 = axes[0, 0]
        x = np.arange(len(top_models))
        width = 0.35
        
        ax1.bar(x - width/2, top_models['Train Score'], width, label='Train Score', alpha=0.8)
        ax1.bar(x + width/2, top_models['Test Score'], width, label='Test Score', alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Train vs Test Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(top_models['Model'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Overfitting analysis
        ax2 = axes[0, 1]
        overfitting = top_models['Train Score'] - top_models['Test Score']
        colors = ['red' if x > 0.1 else 'orange' if x > 0.05 else 'green' for x in overfitting]
        
        bars = ax2.bar(range(len(top_models)), overfitting, color=colors, alpha=0.7)
        ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Warning (0.05)')
        ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='High (0.10)')
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Overfitting (Train - Test)')
        ax2.set_title('Overfitting Analysis')
        ax2.set_xticks(range(len(top_models)))
        ax2.set_xticklabels(top_models['Model'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Efficiency plot (Performance vs Time)
        ax3 = axes[0, 2]
        if 'Training Time' in top_models.columns:
            scatter = ax3.scatter(top_models['Training Time'], top_models['Test Score'], 
                                s=200, alpha=0.7, c=range(len(top_models)), cmap='viridis')
            
            for i, model in enumerate(top_models['Model']):
                ax3.annotate(model, (top_models['Training Time'].iloc[i], top_models['Test Score'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax3.set_xlabel('Training Time (seconds)')
            ax3.set_ylabel('Test Score (R²)')
            ax3.set_title('Efficiency: Performance vs Time')
            ax3.grid(alpha=0.3)
        
        # 4. Performance distribution
        ax4 = axes[1, 0]
        ax4.boxplot([top_models['Train Score'], top_models['Test Score']], 
                   labels=['Train Score', 'Test Score'])
        ax4.set_ylabel('R² Score')
        ax4.set_title('Score Distribution')
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Model ranking
        ax5 = axes[1, 1]
        positions = range(1, len(top_models) + 1)
        bars = ax5.barh(positions, top_models['Test Score'], 
                       color=plt.cm.RdYlGn(top_models['Test Score'] / top_models['Test Score'].max()))
        
        ax5.set_yticks(positions)
        ax5.set_yticklabels([f"{i}. {model}" for i, model in enumerate(top_models['Model'], 1)])
        ax5.set_xlabel('Test Score (R²)')
        ax5.set_title('Model Ranking')
        ax5.grid(axis='x', alpha=0.3)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, top_models['Test Score'])):
            ax5.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.4f}', va='center', fontweight='bold')
        
        # 6. Relative performance
        ax6 = axes[1, 2]
        best_score = top_models['Test Score'].max()
        relative_performance = (top_models['Test Score'] / best_score) * 100
        
        bars = ax6.bar(range(len(top_models)), relative_performance, 
                      color=plt.cm.RdYlGn(relative_performance / 100))
        
        ax6.set_xlabel('Models')
        ax6.set_ylabel('Relative Performance (%)')
        ax6.set_title('Performance Relative to Best Model')
        ax6.set_xticks(range(len(top_models)))
        ax6.set_xticklabels(top_models['Model'], rotation=45, ha='right')
        ax6.set_ylim(0, 105)
        ax6.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for bar, perf in zip(bars, relative_performance):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{perf:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            self.logger.info(f"Top models detailed plot saved to {save_path}")
        
        return save_path or "top_models_detailed_plot"
    
    def plot_feature_importance(self, feature_importance: Dict[str, np.ndarray], 
                              feature_names: List[str], save_path: str = None) -> str:
        """Plot feature importance for models that support it"""
        
        if not feature_importance:
            raise ValueError("No feature importance data provided")
        
        n_models = len(feature_importance)
        fig, axes = plt.subplots((n_models + 1) // 2, 2, figsize=(16, 6 * ((n_models + 1) // 2)))
        if n_models == 1:
            axes = [axes]
        elif n_models <= 2:
            axes = axes.reshape(-1)
        else:
            axes = axes.flatten()
        
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        for i, (model_name, importance) in enumerate(feature_importance.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Sort features by importance
            if len(importance) == len(feature_names):
                feature_imp_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=True).tail(15)  # Top 15 features
                
                bars = ax.barh(feature_imp_df['feature'], feature_imp_df['importance'],
                              color=plt.cm.viridis(np.linspace(0, 1, len(feature_imp_df))))
                
                ax.set_xlabel('Importance')
                ax.set_title(f'{model_name} - Feature Importance')
                ax.grid(axis='x', alpha=0.3)
                
                # Add value labels
                for bar, imp in zip(bars, feature_imp_df['importance']):
                    ax.text(imp + max(feature_imp_df['importance']) * 0.01, 
                           bar.get_y() + bar.get_height()/2,
                           f'{imp:.3f}', va='center', fontsize=8)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        return save_path or "feature_importance_plot"
    
    def plot_residuals_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              model_name: str, save_path: str = None) -> str:
        """Create residuals analysis plots"""
        
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Residuals Analysis - {model_name}', fontsize=16, fontweight='bold')
        
        # 1. Residuals vs Fitted
        ax1 = axes[0, 0]
        ax1.scatter(y_pred, residuals, alpha=0.6)
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted Values')
        ax1.grid(alpha=0.3)
        
        # 2. QQ Plot of residuals
        ax2 = axes[0, 1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Normal Q-Q Plot of Residuals')
        ax2.grid(alpha=0.3)
        
        # 3. Histogram of residuals
        ax3 = axes[1, 0]
        ax3.hist(residuals, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Overlay normal distribution
        mu, sigma = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax3.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
        
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution of Residuals')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Actual vs Predicted
        ax4 = axes[1, 1]
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        
        ax4.scatter(y_true, y_pred, alpha=0.6)
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax4.set_xlabel('Actual Values')
        ax4.set_ylabel('Predicted Values')
        ax4.set_title('Actual vs Predicted Values')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # Calculate and display R²
        r2 = 1 - (np.sum(residuals**2) / np.sum((y_true - np.mean(y_true))**2))
        ax4.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax4.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            self.logger.info(f"Residuals analysis plot saved to {save_path}")
        
        return save_path or "residuals_analysis_plot"
    
    def plot_learning_curves(self, train_scores: np.ndarray, val_scores: np.ndarray, 
                           train_sizes: np.ndarray, model_name: str, 
                           save_path: str = None) -> str:
        """Plot learning curves"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                       alpha=0.2, color='blue')
        
        ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                       alpha=0.2, color='red')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Score (R²)')
        ax.set_title(f'Learning Curves - {model_name}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            self.logger.info(f"Learning curves plot saved to {save_path}")
        
        return save_path or "learning_curves_plot"
    
    def plot_hyperparameter_importance(self, param_importance: Dict[str, float], 
                                     model_name: str, save_path: str = None) -> str:
        """Plot hyperparameter importance"""
        
        if not param_importance:
            raise ValueError("No hyperparameter importance data provided")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        params = list(param_importance.keys())
        importance = list(param_importance.values())
        
        # Sort by importance
        sorted_data = sorted(zip(params, importance), key=lambda x: x[1], reverse=True)
        params, importance = zip(*sorted_data)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(params)))
        bars = ax.barh(params, importance, color=colors)
        
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Hyperparameter Importance - {model_name}')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, imp in zip(bars, importance):
            ax.text(imp + max(importance) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{imp:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            self.logger.info(f"Hyperparameter importance plot saved to {save_path}")
        
        return save_path or "hyperparameter_importance_plot"
    
    def create_model_report(self, job_state: Dict[str, Any], save_dir: str) -> Dict[str, str]:
        """Create comprehensive visual report for a job"""
        
        os.makedirs(save_dir, exist_ok=True)
        report_files = {}
        
        try:
            # Extract data from job state
            model_results = job_state.get('model_results', {})
            all_models = model_results.get('all_models_performance', {})
            suggested_models = model_results.get('suggested_models_performance', {})
            best_model = model_results.get('best_model', {})
            
            if all_models:
                # Convert to DataFrame
                results_data = []
                for model_name, results in all_models.items():
                    results_data.append({
                        'Model': model_name,
                        'Train Score': results.get('train_score', 0),
                        'Test Score': results.get('test_score', 0),
                        'CV Score': results.get('cv_score', 0),
                        'CV Std': results.get('cv_std', 0),
                        'Training Time': results.get('training_time', 0)
                    })
                
                results_df = pd.DataFrame(results_data)
                
                # 1. Overall model comparison
                comparison_path = os.path.join(save_dir, 'model_comparison.png')
                self.plot_model_comparison(results_df, comparison_path, 
                                         f"Model Comparison - Job {job_state['job_info']['job_id']}")
                report_files['model_comparison'] = comparison_path
                
                # 2. Top models detailed analysis
                top_models_path = os.path.join(save_dir, 'top_models_detailed.png')
                self.plot_top_models_detailed(results_df, top_n=5, save_path=top_models_path)
                report_files['top_models_detailed'] = top_models_path
                
                # 3. Performance summary plot
                summary_path = os.path.join(save_dir, 'performance_summary.png')
                self.plot_performance_summary(results_df, best_model, summary_path)
                report_files['performance_summary'] = summary_path
            
            self.logger.info(f"Visual report created in {save_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to create visual report: {str(e)}")
        
        return report_files
    
    def plot_performance_summary(self, results_df: pd.DataFrame, best_model: Dict[str, Any], 
                               save_path: str) -> str:
        """Create a performance summary dashboard"""
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('AutoML Performance Summary', fontsize=20, fontweight='bold')
        
        # 1. Best model highlight (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        
        best_model_name = best_model.get('model_name', 'Unknown')
        best_score = best_model.get('test_score', 0)
        
        ax1.text(0.5, 0.7, f'🏆 Best Model', ha='center', va='center', 
                fontsize=24, fontweight='bold', transform=ax1.transAxes)
        ax1.text(0.5, 0.4, best_model_name, ha='center', va='center', 
                fontsize=18, color='darkblue', transform=ax1.transAxes)
        ax1.text(0.5, 0.1, f'R² Score: {best_score:.4f}', ha='center', va='center', 
                fontsize=16, color='darkgreen', fontweight='bold', transform=ax1.transAxes)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # Add colored background
        ax1.add_patch(plt.Rectangle((0.1, 0.05), 0.8, 0.9, facecolor='lightblue', alpha=0.3))
        
        # 2. Quick stats (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        
        stats_text = f"""
        📊 Training Summary
        
        Total Models: {len(results_df)}
        Mean Performance: {results_df['Test Score'].mean():.3f}
        Std Performance: {results_df['Test Score'].std():.3f}
        Best Performance: {results_df['Test Score'].max():.3f}
        Total Training Time: {results_df['Training Time'].sum():.1f}s
        """
        
        ax2.text(0.05, 0.95, stats_text, ha='left', va='top', fontsize=12, 
                transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        # 3. Top 5 models bar chart (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        top_5 = results_df.nlargest(5, 'Test Score')
        
        bars = ax3.barh(range(len(top_5)), top_5['Test Score'], 
                       color=plt.cm.RdYlGn(top_5['Test Score'] / top_5['Test Score'].max()))
        
        ax3.set_yticks(range(len(top_5)))
        ax3.set_yticklabels(top_5['Model'])
        ax3.set_xlabel('Test Score (R²)')
        ax3.set_title('Top 5 Models')
        ax3.grid(axis='x', alpha=0.3)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, top_5['Test Score'])):
            ax3.text(score + 0.005, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontweight='bold', fontsize=10)
        
        # 4. Performance distribution (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        
        ax4.hist(results_df['Test Score'], bins=15, density=True, alpha=0.7, 
                color='skyblue', edgecolor='black')
        
        # Add vertical line for best model
        ax4.axvline(best_score, color='red', linestyle='--', linewidth=2, 
                   label=f'Best: {best_score:.3f}')
        
        # Add mean line
        mean_score = results_df['Test Score'].mean()
        ax4.axvline(mean_score, color='orange', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_score:.3f}')
        
        ax4.set_xlabel('Test Score (R²)')
        ax4.set_ylabel('Density')
        ax4.set_title('Performance Distribution')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. Efficiency analysis (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        
        # Create efficiency score (performance / time)
        efficiency = results_df['Test Score'] / (results_df['Training Time'] + 1e-6)  # Avoid division by zero
        
        scatter = ax5.scatter(results_df['Training Time'], results_df['Test Score'], 
                            s=100, c=efficiency, cmap='viridis', alpha=0.7)
        
        # Highlight best model
        best_idx = results_df['Test Score'].idxmax()
        ax5.scatter(results_df.loc[best_idx, 'Training Time'], 
                   results_df.loc[best_idx, 'Test Score'],
                   s=200, c='red', marker='*', label='Best Model')
        
        ax5.set_xlabel('Training Time (seconds)')
        ax5.set_ylabel('Test Score (R²)')
        ax5.set_title('Model Efficiency: Performance vs Training Time')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # Add colorbar for efficiency
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Efficiency (Score/Time)')
        
        # Add model labels for interesting points
        for i, model in enumerate(results_df['Model']):
            if (results_df['Test Score'].iloc[i] > results_df['Test Score'].quantile(0.8) or 
                efficiency.iloc[i] > efficiency.quantile(0.9)):
                ax5.annotate(model, (results_df['Training Time'].iloc[i], results_df['Test Score'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            self.logger.info(f"Performance summary plot saved to {save_path}")
        
        return save_path or "performance_summary_plot"
    
    def save_results_table(self, results_df: pd.DataFrame, save_path: str) -> str:
        """Save results as a formatted table image"""
        
        fig, ax = plt.subplots(figsize=(14, len(results_df) * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Sort by test score
        results_sorted = results_df.sort_values('Test Score', ascending=False)
        
        # Create table
        table = ax.table(cellText=results_sorted.round(4).values,
                        colLabels=results_sorted.columns,
                        cellLoc='center',
                        loc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Color code the header
        for i in range(len(results_sorted.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code the best model row
        for i in range(len(results_sorted.columns)):
            table[(1, i)].set_facecolor('#E8F5E8')
            table[(1, i)].set_text_props(weight='bold')
        
        plt.title('Model Performance Results', fontsize=16, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            self.logger.info(f"Results table saved to {save_path}")
        
        return save_path or "results_table"
    
    def close_all_plots(self):
        """Close all open matplotlib figures"""
        plt.close('all')
        
    def set_style(self, style: str = 'default'):
        """Set matplotlib style"""
        try:
            plt.style.use(style)
            self.logger.info(f"Set plot style to: {style}")
        except Exception as e:
            self.logger.warning(f"Could not set style {style}: {str(e)}")
            plt.style.use('default')