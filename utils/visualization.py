# """
# utils/visualization.py
# Visualization utilities for model comparison and results
# """

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import os

# # Set style
# plt.style.use('default')
# sns.set_palette("husl")

# class Visualizer:
#     """Comprehensive visualization utilities for AutoML results"""
    
#     def __init__(self, config: Dict):
#         self.config = config
#         self.logger = logging.getLogger(__name__)
        
#         # Set default figure size and DPI
#         plt.rcParams['figure.figsize'] = (12, 8)
#         plt.rcParams['figure.dpi'] = 100
#         plt.rcParams['savefig.dpi'] = 300
#         plt.rcParams['font.size'] = 10
        
#     def plot_model_comparison(self, results_df: pd.DataFrame, save_path: str = None, 
#                             title: str = "Model Performance Comparison") -> str:
#         """Create comprehensive model comparison plot"""
        
#         if results_df.empty:
#             raise ValueError("Results dataframe is empty")
        
#         # Create figure with subplots
#         fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#         fig.suptitle(title, fontsize=16, fontweight='bold')
        
#         # Sort by test score for better visualization
#         results_sorted = results_df.sort_values('Test Score', ascending=True)
        
#         # 1. Horizontal bar chart of test scores
#         ax1 = axes[0, 0]
#         bars = ax1.barh(results_sorted['Model'], results_sorted['Test Score'], 
#                        color=plt.cm.viridis(np.linspace(0, 1, len(results_sorted))))
#         ax1.set_xlabel('Test Score (R²)')
#         ax1.set_title('Test Performance Comparison')
#         ax1.grid(axis='x', alpha=0.3)
        
#         # Add value labels on bars
#         for i, (bar, score) in enumerate(zip(bars, results_sorted['Test Score'])):
#             ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
#                     f'{score:.3f}', va='center', fontweight='bold')
        
#         # 2. Train vs Test scores scatter plot
#         ax2 = axes[0, 1]
#         scatter = ax2.scatter(results_sorted['Train Score'], results_sorted['Test Score'], 
#                             s=100, alpha=0.7, c=range(len(results_sorted)), cmap='viridis')
        
#         # Add diagonal line (perfect fit)
#         min_score = min(results_sorted['Train Score'].min(), results_sorted['Test Score'].min())
#         max_score = max(results_sorted['Train Score'].max(), results_sorted['Test Score'].max())
#         ax2.plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.5, label='Perfect Fit')
        
#         ax2.set_xlabel('Train Score (R²)')
#         ax2.set_ylabel('Test Score (R²)')
#         ax2.set_title('Train vs Test Performance')
#         ax2.legend()
#         ax2.grid(alpha=0.3)
        
#         # Add model labels
#         for i, model in enumerate(results_sorted['Model']):
#             ax2.annotate(model, (results_sorted['Train Score'].iloc[i], 
#                                results_sorted['Test Score'].iloc[i]),
#                         xytext=(5, 5), textcoords='offset points', 
#                         fontsize=8, alpha=0.7)
        
#         # 3. Cross-validation scores with error bars
#         ax3 = axes[1, 0]
#         if 'CV Score' in results_sorted.columns and 'CV Std' in results_sorted.columns:
#             x_pos = range(len(results_sorted))
#             bars = ax3.bar(x_pos, results_sorted['CV Score'], 
#                           yerr=results_sorted['CV Std'], capsize=5,
#                           color=plt.cm.plasma(np.linspace(0, 1, len(results_sorted))))
            
#             ax3.set_xlabel('Models')
#             ax3.set_ylabel('CV Score (R²)')
#             ax3.set_title('Cross-Validation Performance')
#             ax3.set_xticks(x_pos)
#             ax3.set_xticklabels(results_sorted['Model'], rotation=45, ha='right')
#             ax3.grid(axis='y', alpha=0.3)
            
#             # Add value labels
#             for i, (bar, score, std) in enumerate(zip(bars, results_sorted['CV Score'], results_sorted['CV Std'])):
#                 ax3.text(bar.get_x() + bar.get_width()/2, score + std + 0.01, 
#                         f'{score:.3f}', ha='center', fontweight='bold', fontsize=8)
#         else:
#             ax3.text(0.5, 0.5, 'CV scores not available', ha='center', va='center', 
#                     transform=ax3.transAxes, fontsize=12)
#             ax3.set_title('Cross-Validation Performance')
        
#         # 4. Training time comparison
#         ax4 = axes[1, 1]
#         if 'Training Time' in results_sorted.columns:
#             bars = ax4.bar(range(len(results_sorted)), results_sorted['Training Time'],
#                           color=plt.cm.coolwarm(np.linspace(0, 1, len(results_sorted))))
            
#             ax4.set_xlabel('Models')
#             ax4.set_ylabel('Training Time (seconds)')
#             ax4.set_title('Training Time Comparison')
#             ax4.set_xticks(range(len(results_sorted)))
#             ax4.set_xticklabels(results_sorted['Model'], rotation=45, ha='right')
#             ax4.grid(axis='y', alpha=0.3)
            
#             # Add value labels
#             for i, (bar, time) in enumerate(zip(bars, results_sorted['Training Time'])):
#                 ax4.text(bar.get_x() + bar.get_width()/2, time + max(results_sorted['Training Time'])*0.01, 
#                         f'{time:.1f}s', ha='center', fontweight='bold', fontsize=8)
#         else:
#             ax4.text(0.5, 0.5, 'Training times not available', ha='center', va='center', 
#                     transform=ax4.transAxes, fontsize=12)
#             ax4.set_title('Training Time Comparison')
        
#         plt.tight_layout()
        
#         # Save plot if path provided
#         if save_path:
#             plt.savefig(save_path, bbox_inches='tight', dpi=300)
#             self.logger.info(f"Model comparison plot saved to {save_path}")
        
#         return save_path or "model_comparison_plot"
    
#     def plot_top_models_detailed(self, results_df: pd.DataFrame, top_n: int = 5, 
#                                save_path: str = None) -> str:
#         """Create detailed plot of top N models"""
        
#         # Get top N models
#         top_models = results_df.nlargest(top_n, 'Test Score')
        
#         fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#         fig.suptitle(f'Top {top_n} Models - Detailed Analysis', fontsize=16, fontweight='bold')
        
#         # 1. Performance comparison
#         ax1 = axes[0, 0]
#         x = np.arange(len(top_models))
#         width = 0.35
        
#         ax1.bar(x - width/2, top_models['Train Score'], width, label='Train Score', alpha=0.8)
#         ax1.bar(x + width/2, top_models['Test Score'], width, label='Test Score', alpha=0.8)
        
#         ax1.set_xlabel('Models')
#         ax1.set_ylabel('R² Score')
#         ax1.set_title('Train vs Test Performance')
#         ax1.set_xticks(x)
#         ax1.set_xticklabels(top_models['Model'], rotation=45, ha='right')
#         ax1.legend()
#         ax1.grid(axis='y', alpha=0.3)
        
#         # 2. Overfitting analysis
#         ax2 = axes[0, 1]
#         overfitting = top_models['Train Score'] - top_models['Test Score']
#         colors = ['red' if x > 0.1 else 'orange' if x > 0.05 else 'green' for x in overfitting]
        
#         bars = ax2.bar(range(len(top_models)), overfitting, color=colors, alpha=0.7)
#         ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Warning (0.05)')
#         ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='High (0.10)')
        
#         ax2.set_xlabel('Models')
#         ax2.set_ylabel('Overfitting (Train - Test)')
#         ax2.set_title('Overfitting Analysis')
#         ax2.set_xticks(range(len(top_models)))
#         ax2.set_xticklabels(top_models['Model'], rotation=45, ha='right')
#         ax2.legend()
#         ax2.grid(axis='y', alpha=0.3)
        
#         # 3. Efficiency plot (Performance vs Time)
#         ax3 = axes[0, 2]
#         if 'Training Time' in top_models.columns:
#             scatter = ax3.scatter(top_models['Training Time'], top_models['Test Score'], 
#                                 s=200, alpha=0.7, c=range(len(top_models)), cmap='viridis')
            
#             for i, model in enumerate(top_models['Model']):
#                 ax3.annotate(model, (top_models['Training Time'].iloc[i], top_models['Test Score'].iloc[i]),
#                            xytext=(5, 5), textcoords='offset points', fontsize=8)
            
#             ax3.set_xlabel('Training Time (seconds)')
#             ax3.set_ylabel('Test Score (R²)')
#             ax3.set_title('Efficiency: Performance vs Time')
#             ax3.grid(alpha=0.3)
        
#         # 4. Performance distribution
#         ax4 = axes[1, 0]
#         ax4.boxplot([top_models['Train Score'], top_models['Test Score']], 
#                    labels=['Train Score', 'Test Score'])
#         ax4.set_ylabel('R² Score')
#         ax4.set_title('Score Distribution')
#         ax4.grid(axis='y', alpha=0.3)
        
#         # 5. Model ranking
#         ax5 = axes[1, 1]
#         positions = range(1, len(top_models) + 1)
#         bars = ax5.barh(positions, top_models['Test Score'], 
#                        color=plt.cm.RdYlGn(top_models['Test Score'] / top_models['Test Score'].max()))
        
#         ax5.set_yticks(positions)
#         ax5.set_yticklabels([f"{i}. {model}" for i, model in enumerate(top_models['Model'], 1)])
#         ax5.set_xlabel('Test Score (R²)')
#         ax5.set_title('Model Ranking')
#         ax5.grid(axis='x', alpha=0.3)
        
#         # Add score labels
#         for i, (bar, score) in enumerate(zip(bars, top_models['Test Score'])):
#             ax5.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
#                     f'{score:.4f}', va='center', fontweight='bold')
        
#         # 6. Relative performance
#         ax6 = axes[1, 2]
#         best_score = top_models['Test Score'].max()
#         relative_performance = (top_models['Test Score'] / best_score) * 100
        
#         bars = ax6.bar(range(len(top_models)), relative_performance, 
#                       color=plt.cm.RdYlGn(relative_performance / 100))
        
#         ax6.set_xlabel('Models')
#         ax6.set_ylabel('Relative Performance (%)')
#         ax6.set_title('Performance Relative to Best Model')
#         ax6.set_xticks(range(len(top_models)))
#         ax6.set_xticklabels(top_models['Model'], rotation=45, ha='right')
#         ax6.set_ylim(0, 105)
#         ax6.grid(axis='y', alpha=0.3)
        
#         # Add percentage labels
#         for bar, perf in zip(bars, relative_performance):
#             ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
#                     f'{perf:.1f}%', ha='center', fontweight='bold')
        
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, bbox_inches='tight', dpi=300)
#             self.logger.info(f"Top models detailed plot saved to {save_path}")
        
#         return save_path or "top_models_detailed_plot"
    
#     def plot_feature_importance(self, feature_importance: Dict[str, np.ndarray], 
#                               feature_names: List[str], save_path: str = None) -> str:
#         """Plot feature importance for models that support it"""
        
#         if not feature_importance:
#             raise ValueError("No feature importance data provided")
        
#         n_models = len(feature_importance)
#         fig, axes = plt.subplots((n_models + 1) // 2, 2, figsize=(16, 6 * ((n_models + 1) // 2)))
#         if n_models == 1:
#             axes = [axes]
#         elif n_models <= 2:
#             axes = axes.reshape(-1)
#         else:
#             axes = axes.flatten()
        
#         fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
#         for i, (model_name, importance) in enumerate(feature_importance.items()):
#             if i >= len(axes):
#                 break
                
#             ax = axes[i]
            
#             # Sort features by importance
#             if len(importance) == len(feature_names):
#                 feature_imp_df = pd.DataFrame({
#                     'feature': feature_names,
#                     'importance': importance
#                 }).sort_values('importance', ascending=True).tail(15)  # Top 15 features
                
#                 bars = ax.barh(feature_imp_df['feature'], feature_imp_df['importance'],
#                               color=plt.cm.viridis(np.linspace(0, 1, len(feature_imp_df))))
                
#                 ax.set_xlabel('Importance')
#                 ax.set_title(f'{model_name} - Feature Importance')
#                 ax.grid(axis='x', alpha=0.3)
                
#                 # Add value labels
#                 for bar, imp in zip(bars, feature_imp_df['importance']):
#                     ax.text(imp + max(feature_imp_df['importance']) * 0.01, 
#                            bar.get_y() + bar.get_height()/2,
#                            f'{imp:.3f}', va='center', fontsize=8)
        
#         # Hide unused subplots
#         for j in range(i + 1, len(axes)):
#             axes[j].set_visible(False)
        
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, bbox_inches='tight', dpi=300)
#             self.logger.info(f"Feature importance plot saved to {save_path}")
        
#         return save_path or "feature_importance_plot"
    
#     def plot_residuals_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
#                               model_name: str, save_path: str = None) -> str:
#         """Create residuals analysis plots"""
        
#         residuals = y_true - y_pred
        
#         fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#         fig.suptitle(f'Residuals Analysis - {model_name}', fontsize=16, fontweight='bold')
        
#         # 1. Residuals vs Fitted
#         ax1 = axes[0, 0]
#         ax1.scatter(y_pred, residuals, alpha=0.6)
#         ax1.axhline(y=0, color='red', linestyle='--')
#         ax1.set_xlabel('Fitted Values')
#         ax1.set_ylabel('Residuals')
#         ax1.set_title('Residuals vs Fitted Values')
#         ax1.grid(alpha=0.3)
        
#         # 2. QQ Plot of residuals
#         ax2 = axes[0, 1]
#         from scipy import stats
#         stats.probplot(residuals, dist="norm", plot=ax2)
#         ax2.set_title('Normal Q-Q Plot of Residuals')
#         ax2.grid(alpha=0.3)
        
#         # 3. Histogram of residuals
#         ax3 = axes[1, 0]
#         ax3.hist(residuals, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
#         # Overlay normal distribution
#         mu, sigma = stats.norm.fit(residuals)
#         x = np.linspace(residuals.min(), residuals.max(), 100)
#         ax3.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
        
#         ax3.set_xlabel('Residuals')
#         ax3.set_ylabel('Density')
#         ax3.set_title('Distribution of Residuals')
#         ax3.legend()
#         ax3.grid(alpha=0.3)
        
#         # 4. Actual vs Predicted
#         ax4 = axes[1, 1]
#         min_val = min(y_true.min(), y_pred.min())
#         max_val = max(y_true.max(), y_pred.max())
        
#         ax4.scatter(y_true, y_pred, alpha=0.6)
#         ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
#         ax4.set_xlabel('Actual Values')
#         ax4.set_ylabel('Predicted Values')
#         ax4.set_title('Actual vs Predicted Values')
#         ax4.legend()
#         ax4.grid(alpha=0.3)
        
#         # Calculate and display R²
#         r2 = 1 - (np.sum(residuals**2) / np.sum((y_true - np.mean(y_true))**2))
#         ax4.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax4.transAxes, 
#                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, bbox_inches='tight', dpi=300)
#             self.logger.info(f"Residuals analysis plot saved to {save_path}")
        
#         return save_path or "residuals_analysis_plot"
    
#     def plot_learning_curves(self, train_scores: np.ndarray, val_scores: np.ndarray, 
#                            train_sizes: np.ndarray, model_name: str, 
#                            save_path: str = None) -> str:
#         """Plot learning curves"""
        
#         fig, ax = plt.subplots(figsize=(10, 6))
        
#         train_mean = np.mean(train_scores, axis=1)
#         train_std = np.std(train_scores, axis=1)
#         val_mean = np.mean(val_scores, axis=1)
#         val_std = np.std(val_scores, axis=1)
        
#         ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
#         ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
#                        alpha=0.2, color='blue')
        
#         ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
#         ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
#                        alpha=0.2, color='red')
        
#         ax.set_xlabel('Training Set Size')
#         ax.set_ylabel('Score (R²)')
#         ax.set_title(f'Learning Curves - {model_name}')
#         ax.legend()
#         ax.grid(alpha=0.3)
        
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, bbox_inches='tight', dpi=300)
#             self.logger.info(f"Learning curves plot saved to {save_path}")
        
#         return save_path or "learning_curves_plot"
    
#     def plot_hyperparameter_importance(self, param_importance: Dict[str, float], 
#                                      model_name: str, save_path: str = None) -> str:
#         """Plot hyperparameter importance"""
        
#         if not param_importance:
#             raise ValueError("No hyperparameter importance data provided")
        
#         fig, ax = plt.subplots(figsize=(10, 6))
        
#         params = list(param_importance.keys())
#         importance = list(param_importance.values())
        
#         # Sort by importance
#         sorted_data = sorted(zip(params, importance), key=lambda x: x[1], reverse=True)
#         params, importance = zip(*sorted_data)
        
#         colors = plt.cm.viridis(np.linspace(0, 1, len(params)))
#         bars = ax.barh(params, importance, color=colors)
        
#         ax.set_xlabel('Importance Score')
#         ax.set_title(f'Hyperparameter Importance - {model_name}')
#         ax.grid(axis='x', alpha=0.3)
        
#         # Add value labels
#         for bar, imp in zip(bars, importance):
#             ax.text(imp + max(importance) * 0.01, bar.get_y() + bar.get_height()/2,
#                    f'{imp:.3f}', va='center', fontweight='bold')
        
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, bbox_inches='tight', dpi=300)
#             self.logger.info(f"Hyperparameter importance plot saved to {save_path}")
        
#         return save_path or "hyperparameter_importance_plot"
    
#     def create_model_report(self, job_state: Dict[str, Any], save_dir: str) -> Dict[str, str]:
#         """Create comprehensive visual report for a job"""
        
#         os.makedirs(save_dir, exist_ok=True)
#         report_files = {}
        
#         try:
#             # Extract data from job state
#             model_results = job_state.get('model_results', {})
#             all_models = model_results.get('all_models_performance', {})
#             suggested_models = model_results.get('suggested_models_performance', {})
#             best_model = model_results.get('best_model', {})
            
#             if all_models:
#                 # Convert to DataFrame
#                 results_data = []
#                 for model_name, results in all_models.items():
#                     results_data.append({
#                         'Model': model_name,
#                         'Train Score': results.get('train_score', 0),
#                         'Test Score': results.get('test_score', 0),
#                         'CV Score': results.get('cv_score', 0),
#                         'CV Std': results.get('cv_std', 0),
#                         'Training Time': results.get('training_time', 0)
#                     })
                
#                 results_df = pd.DataFrame(results_data)
                
#                 # 1. Overall model comparison
#                 comparison_path = os.path.join(save_dir, 'model_comparison.png')
#                 self.plot_model_comparison(results_df, comparison_path, 
#                                          f"Model Comparison - Job {job_state['job_info']['job_id']}")
#                 report_files['model_comparison'] = comparison_path
                
#                 # 2. Top models detailed analysis
#                 top_models_path = os.path.join(save_dir, 'top_models_detailed.png')
#                 self.plot_top_models_detailed(results_df, top_n=5, save_path=top_models_path)
#                 report_files['top_models_detailed'] = top_models_path
                
#                 # 3. Performance summary plot
#                 summary_path = os.path.join(save_dir, 'performance_summary.png')
#                 self.plot_performance_summary(results_df, best_model, summary_path)
#                 report_files['performance_summary'] = summary_path
            
#             self.logger.info(f"Visual report created in {save_dir}")
            
#         except Exception as e:
#             self.logger.error(f"Failed to create visual report: {str(e)}")
        
#         return report_files
    
#     def plot_performance_summary(self, results_df: pd.DataFrame, best_model: Dict[str, Any], 
#                                save_path: str) -> str:
#         """Create a performance summary dashboard"""
        
#         fig = plt.figure(figsize=(16, 10))
#         gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
#         fig.suptitle('AutoML Performance Summary', fontsize=20, fontweight='bold')
        
#         # 1. Best model highlight (top left)
#         ax1 = fig.add_subplot(gs[0, :2])
        
#         best_model_name = best_model.get('model_name', 'Unknown')
#         best_score = best_model.get('test_score', 0)
        
#         ax1.text(0.5, 0.7, f'🏆 Best Model', ha='center', va='center', 
#                 fontsize=24, fontweight='bold', transform=ax1.transAxes)
#         ax1.text(0.5, 0.4, best_model_name, ha='center', va='center', 
#                 fontsize=18, color='darkblue', transform=ax1.transAxes)
#         ax1.text(0.5, 0.1, f'R² Score: {best_score:.4f}', ha='center', va='center', 
#                 fontsize=16, color='darkgreen', fontweight='bold', transform=ax1.transAxes)
#         ax1.set_xlim(0, 1)
#         ax1.set_ylim(0, 1)
#         ax1.axis('off')
        
#         # Add colored background
#         ax1.add_patch(plt.Rectangle((0.1, 0.05), 0.8, 0.9, facecolor='lightblue', alpha=0.3))
        
#         # 2. Quick stats (top right)
#         ax2 = fig.add_subplot(gs[0, 2:])
        
#         stats_text = f"""
#         📊 Training Summary
        
#         Total Models: {len(results_df)}
#         Mean Performance: {results_df['Test Score'].mean():.3f}
#         Std Performance: {results_df['Test Score'].std():.3f}
#         Best Performance: {results_df['Test Score'].max():.3f}
#         Total Training Time: {results_df['Training Time'].sum():.1f}s
#         """
        
#         ax2.text(0.05, 0.95, stats_text, ha='left', va='top', fontsize=12, 
#                 transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
#         ax2.set_xlim(0, 1)
#         ax2.set_ylim(0, 1)
#         ax2.axis('off')
        
#         # 3. Top 5 models bar chart (middle left)
#         ax3 = fig.add_subplot(gs[1, :2])
#         top_5 = results_df.nlargest(5, 'Test Score')
        
#         bars = ax3.barh(range(len(top_5)), top_5['Test Score'], 
#                        color=plt.cm.RdYlGn(top_5['Test Score'] / top_5['Test Score'].max()))
        
#         ax3.set_yticks(range(len(top_5)))
#         ax3.set_yticklabels(top_5['Model'])
#         ax3.set_xlabel('Test Score (R²)')
#         ax3.set_title('Top 5 Models')
#         ax3.grid(axis='x', alpha=0.3)
        
#         # Add score labels
#         for i, (bar, score) in enumerate(zip(bars, top_5['Test Score'])):
#             ax3.text(score + 0.005, bar.get_y() + bar.get_height()/2, 
#                     f'{score:.3f}', va='center', fontweight='bold', fontsize=10)
        
#         # 4. Performance distribution (middle right)
#         ax4 = fig.add_subplot(gs[1, 2:])
        
#         ax4.hist(results_df['Test Score'], bins=15, density=True, alpha=0.7, 
#                 color='skyblue', edgecolor='black')
        
#         # Add vertical line for best model
#         ax4.axvline(best_score, color='red', linestyle='--', linewidth=2, 
#                    label=f'Best: {best_score:.3f}')
        
#         # Add mean line
#         mean_score = results_df['Test Score'].mean()
#         ax4.axvline(mean_score, color='orange', linestyle='--', linewidth=2, 
#                    label=f'Mean: {mean_score:.3f}')
        
#         ax4.set_xlabel('Test Score (R²)')
#         ax4.set_ylabel('Density')
#         ax4.set_title('Performance Distribution')
#         ax4.legend()
#         ax4.grid(alpha=0.3)
        
#         # 5. Efficiency analysis (bottom)
#         ax5 = fig.add_subplot(gs[2, :])
        
#         # Create efficiency score (performance / time)
#         efficiency = results_df['Test Score'] / (results_df['Training Time'] + 1e-6)  # Avoid division by zero
        
#         scatter = ax5.scatter(results_df['Training Time'], results_df['Test Score'], 
#                             s=100, c=efficiency, cmap='viridis', alpha=0.7)
        
#         # Highlight best model
#         best_idx = results_df['Test Score'].idxmax()
#         ax5.scatter(results_df.loc[best_idx, 'Training Time'], 
#                    results_df.loc[best_idx, 'Test Score'],
#                    s=200, c='red', marker='*', label='Best Model')
        
#         ax5.set_xlabel('Training Time (seconds)')
#         ax5.set_ylabel('Test Score (R²)')
#         ax5.set_title('Model Efficiency: Performance vs Training Time')
#         ax5.legend()
#         ax5.grid(alpha=0.3)
        
#         # Add colorbar for efficiency
#         cbar = plt.colorbar(scatter, ax=ax5)
#         cbar.set_label('Efficiency (Score/Time)')
        
#         # Add model labels for interesting points
#         for i, model in enumerate(results_df['Model']):
#             if (results_df['Test Score'].iloc[i] > results_df['Test Score'].quantile(0.8) or 
#                 efficiency.iloc[i] > efficiency.quantile(0.9)):
#                 ax5.annotate(model, (results_df['Training Time'].iloc[i], results_df['Test Score'].iloc[i]),
#                            xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
#         if save_path:
#             plt.savefig(save_path, bbox_inches='tight', dpi=300)
#             self.logger.info(f"Performance summary plot saved to {save_path}")
        
#         return save_path or "performance_summary_plot"
    
#     def save_results_table(self, results_df: pd.DataFrame, save_path: str) -> str:
#         """Save results as a formatted table image"""
        
#         fig, ax = plt.subplots(figsize=(14, len(results_df) * 0.5 + 2))
#         ax.axis('tight')
#         ax.axis('off')
        
#         # Sort by test score
#         results_sorted = results_df.sort_values('Test Score', ascending=False)
        
#         # Create table
#         table = ax.table(cellText=results_sorted.round(4).values,
#                         colLabels=results_sorted.columns,
#                         cellLoc='center',
#                         loc='center')
        
#         # Style the table
#         table.auto_set_font_size(False)
#         table.set_fontsize(10)
#         table.scale(1.2, 2)
        
#         # Color code the header
#         for i in range(len(results_sorted.columns)):
#             table[(0, i)].set_facecolor('#4CAF50')
#             table[(0, i)].set_text_props(weight='bold', color='white')
        
#         # Color code the best model row
#         for i in range(len(results_sorted.columns)):
#             table[(1, i)].set_facecolor('#E8F5E8')
#             table[(1, i)].set_text_props(weight='bold')
        
#         plt.title('Model Performance Results', fontsize=16, fontweight='bold', pad=20)
        
#         if save_path:
#             plt.savefig(save_path, bbox_inches='tight', dpi=300)
#             self.logger.info(f"Results table saved to {save_path}")
        
#         return save_path or "results_table"
    
#     def close_all_plots(self):
#         """Close all open matplotlib figures"""
#         plt.close('all')
        
#     def set_style(self, style: str = 'default'):
#         """Set matplotlib style"""
#         try:
#             plt.style.use(style)
#             self.logger.info(f"Set plot style to: {style}")
#         except Exception as e:
#             self.logger.warning(f"Could not set style {style}: {str(e)}")
#             plt.style.use('default')

# Utility functions for standalone use
import pandas as pd
def create_quick_comparison(results_df: pd.DataFrame, save_path: str = None):
    """Create a quick model comparison plot without class instantiation"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if 'Test Score' not in results_df.columns or 'Model' not in results_df.columns:
            print("❌ Required columns missing: 'Model' and 'Test Score'")
            return False
        
        # Create simple figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sort data
        results_sorted = results_df.sort_values('Test Score', ascending=False)
        
        # 1. Bar chart
        bars = axes[0].bar(range(len(results_sorted)), results_sorted['Test Score'], 
                          color='skyblue', alpha=0.7, edgecolor='navy')
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('Test Score (R²)')
        axes[0].set_title('Model Performance Comparison')
        axes[0].set_xticks(range(len(results_sorted)))
        axes[0].set_xticklabels(results_sorted['Model'], rotation=45, ha='right')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add scores on bars
        for i, (bar, score) in enumerate(zip(bars, results_sorted['Test Score'])):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Highlight best model
        if len(bars) > 0:
            bars[0].set_color('gold')
            bars[0].set_edgecolor('darkgoldenrod')
        
        # 2. Train vs Test comparison (if available)
        if 'Train Score' in results_df.columns:
            x = np.arange(len(results_sorted))
            width = 0.35
            
            axes[1].bar(x - width/2, results_sorted['Train Score'], width, 
                       label='Train Score', alpha=0.8, color='lightblue')
            axes[1].bar(x + width/2, results_sorted['Test Score'], width, 
                       label='Test Score', alpha=0.8, color='lightcoral')
            
            axes[1].set_xlabel('Models')
            axes[1].set_ylabel('Score (R²)')
            axes[1].set_title('Train vs Test Performance')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(results_sorted['Model'], rotation=45, ha='right')
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)
        else:
            # Just show a summary table
            axes[1].axis('off')
            
            table_text = "📊 MODEL SUMMARY\n\n"
            for i, (_, row) in enumerate(results_sorted.head(5).iterrows(), 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                table_text += f"{medal} {row['Model']}: {row['Test Score']:.4f}\n"
            
            axes[1].text(0.1, 0.9, table_text, transform=axes[1].transAxes, 
                        verticalalignment='top', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"✅ Plot saved to {save_path}")
        
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"❌ Failed to create comparison plot: {e}")
        return False

def create_correlation_heatmap(df: pd.DataFrame, save_path: str = None, figsize: tuple = (12, 10)):
    """Create correlation heatmap similar to your old corrplot function"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            print("❌ Insufficient numeric columns for correlation analysis")
            return False
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create the plot
        plt.figure(figsize=figsize)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, annot_kws={"size": 14}, linewidth=.5, 
                   cmap='RdBu_r', center=0, square=True, 
                   cbar_kws={"shrink": .8})
        
        plt.title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"✅ Correlation plot saved to {save_path}")
        
        plt.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed to create correlation heatmap: {e}")
        return False

def get_high_corr_columns(data: pd.DataFrame, threshold: float = 0.7) -> list[str]:
    """Get columns with high correlation - enhanced version of your original function"""
    try:
        # Get numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        high_corr_columns = []
        high_corr_pairs = []
        
        for row in corr_matrix.index:
            for col in corr_matrix.columns:
                if row != col and abs(corr_matrix.loc[row, col]) > threshold:
                    high_corr_columns.append(row)
                    high_corr_columns.append(col)
                    high_corr_pairs.append((row, col, corr_matrix.loc[row, col]))
        
        # Make unique
        unique_columns = list(set(high_corr_columns))
        
        # Print some info
        print(f"📊 Found {len(unique_columns)} columns with correlation > {threshold}")
        print(f"🔗 High correlation pairs: {len(high_corr_pairs)//2}")  # Divide by 2 since we count each pair twice
        
        # Show top correlations
        if high_corr_pairs:
            print("\n🔥 Top correlations:")
            unique_pairs = []
            for pair in high_corr_pairs:
                reverse_pair = (pair[1], pair[0], pair[2])
                if reverse_pair not in unique_pairs:
                    unique_pairs.append(pair)
            
            sorted_pairs = sorted(unique_pairs, key=lambda x: abs(x[2]), reverse=True)
            for i, (col1, col2, corr) in enumerate(sorted_pairs[:5]):
                print(f"   {i+1}. {col1} ↔ {col2}: {corr:.3f}")
        
        return unique_columns
        
    except Exception as e:
        print(f"❌ Error in correlation analysis: {e}")
        return []

def plot_numerical_distributions(data: pd.DataFrame, save_path: str = None):
    """Enhanced version of your plot_numerical_columns function"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) == 0:
            print("❌ No numerical columns found")
            return False
        
        n_cols = 3  # Three plots per row: histogram, kde, boxplot
        n_rows = len(numerical_columns)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Numerical Columns Distribution Analysis', fontsize=16, fontweight='bold')
        
        for i, column in enumerate(numerical_columns):
            # 1. Histogram with KDE
            sns.histplot(data[column], bins=50, color="#512DA8", ax=axes[i, 0], kde=True, alpha=0.7)
            axes[i, 0].set_xlabel(column.replace('_', ' ').title(), fontsize=12)
            axes[i, 0].set_ylabel("Frequency", fontsize=12)
            axes[i, 0].set_title(f"{column.replace('_', ' ').title()} Distribution", fontsize=12)
            axes[i, 0].grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = data[column].mean()
            std_val = data[column].std()
            skew_val = data[column].skew()
            
            stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nSkew: {skew_val:.2f}'
            axes[i, 0].text(0.02, 0.98, stats_text, transform=axes[i, 0].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 2. KDE Plot
            sns.kdeplot(data[column], ax=axes[i, 1], color="#FF6B6B", fill=True, alpha=0.6)
            axes[i, 1].set_xlabel(column.replace('_', ' ').title(), fontsize=12)
            axes[i, 1].set_ylabel("Density", fontsize=12)
            axes[i, 1].set_title(f"{column.replace('_', ' ').title()} Density Plot", fontsize=12)
            axes[i, 1].grid(True, alpha=0.3)
            
            # 3. Box plot for outlier detection
            sns.boxplot(x=data[column], ax=axes[i, 2], color="#4ECDC4")
            axes[i, 2].set_xlabel(column.replace('_', ' ').title(), fontsize=12)
            axes[i, 2].set_title(f"{column.replace('_', ' ').title()} Box Plot", fontsize=12)
            axes[i, 2].grid(True, alpha=0.3)
            
            # Add outlier count
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            
            outlier_text = f'Outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)'
            axes[i, 2].text(0.02, 0.98, outlier_text, transform=axes[i, 2].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"✅ Distribution plots saved to {save_path}")
        
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"❌ Failed to create distribution plots: {e}")
        return False

def plot_outliers_comparison(df_original: pd.DataFrame, df_cleaned: pd.DataFrame = None, 
                           predictor_variable: str = None, save_path: str = None):
    """Enhanced version of your Comp_plot_boxplots function"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if predictor_variable and predictor_variable in df_original.columns:
            # Single variable comparison
            if df_cleaned is not None and predictor_variable in df_cleaned.columns:
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                # Before outlier removal
                sns.boxplot(y=df_original[predictor_variable], ax=axes[0], color='lightcoral')
                axes[0].set_title(f'{predictor_variable}\nBefore Outlier Removal')
                axes[0].grid(True, alpha=0.3)
                
                # Add statistics
                Q1 = df_original[predictor_variable].quantile(0.25)
                Q3 = df_original[predictor_variable].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers_before = df_original[(df_original[predictor_variable] < lower_bound) | 
                                            (df_original[predictor_variable] > upper_bound)]
                
                stats_text = f'Total: {len(df_original)}\nOutliers: {len(outliers_before)}\n({len(outliers_before)/len(df_original)*100:.1f}%)'
                axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # After outlier removal
                sns.boxplot(y=df_cleaned[predictor_variable], ax=axes[1], color='lightgreen')
                axes[1].set_title(f'{predictor_variable}\nAfter Outlier Removal')
                axes[1].grid(True, alpha=0.3)
                
                # Add statistics for cleaned data
                Q1_clean = df_cleaned[predictor_variable].quantile(0.25)
                Q3_clean = df_cleaned[predictor_variable].quantile(0.75)
                IQR_clean = Q3_clean - Q1_clean
                lower_bound_clean = Q1_clean - 1.5 * IQR_clean
                upper_bound_clean = Q3_clean + 1.5 * IQR_clean
                outliers_after = df_cleaned[(df_cleaned[predictor_variable] < lower_bound_clean) | 
                                          (df_cleaned[predictor_variable] > upper_bound_clean)]
                
                stats_text_clean = f'Total: {len(df_cleaned)}\nOutliers: {len(outliers_after)}\n({len(outliers_after)/len(df_cleaned)*100:.1f}%)'
                axes[1].text(0.02, 0.98, stats_text_clean, transform=axes[1].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.suptitle(f'Outlier Analysis: {predictor_variable}', fontsize=14, fontweight='bold')
            else:
                # Only original data
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                sns.boxplot(y=df_original[predictor_variable], ax=ax, color='lightblue')
                ax.set_title(f'{predictor_variable}\nOutlier Detection')
                ax.grid(True, alpha=0.3)
        else:
            # Multiple variables
            numeric_cols = df_original.select_dtypes(include=[np.number]).columns
            display_cols = numeric_cols[:6]  # Limit to 6 for readability
            
            if df_cleaned is not None:
                fig, axes = plt.subplots(2, len(display_cols), figsize=(4*len(display_cols), 10))
                fig.suptitle('Outlier Analysis: Before vs After Cleaning', fontsize=16, fontweight='bold')
                
                for i, col in enumerate(display_cols):
                    # Before
                    sns.boxplot(y=df_original[col], ax=axes[0, i], color='lightcoral')
                    axes[0, i].set_title(f'{col}\nBefore')
                    
                    # After (if column exists in cleaned data)
                    if col in df_cleaned.columns:
                        sns.boxplot(y=df_cleaned[col], ax=axes[1, i], color='lightgreen')
                        axes[1, i].set_title(f'{col}\nAfter')
                    else:
                        axes[1, i].text(0.5, 0.5, 'Column\nRemoved', ha='center', va='center', 
                                      transform=axes[1, i].transAxes)
                        axes[1, i].set_title(f'{col}\nRemoved')
            else:
                fig, axes = plt.subplots(1, len(display_cols), figsize=(4*len(display_cols), 6))
                fig.suptitle('Outlier Detection', fontsize=16, fontweight='bold')
                
                if len(display_cols) == 1:
                    axes = [axes]
                
                for i, col in enumerate(display_cols):
                    sns.boxplot(y=df_original[col], ax=axes[i], color='lightblue')
                    axes[i].set_title(f'{col}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"✅ Outlier comparison saved to {save_path}")
        
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"❌ Failed to create outlier comparison: {e}")
        return False

def plot_model_evaluation(model_eval: pd.DataFrame, save_path: str = None):
    """Enhanced version of your plot_model_eval function"""
    try:
        import matplotlib.pyplot as plt
        
        if 'Model' not in model_eval.columns:
            print("❌ 'Model' column required")
            return False
        
        numeric_labels = list(range(len(model_eval)))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Line plot (original style)
        ax1 = axes[0]
        if 'Train Score' in model_eval.columns:
            ax1.plot(numeric_labels, model_eval['Train Score'], label='Train Score', 
                    marker='o', linewidth=2, markersize=8)
        if 'Test Score' in model_eval.columns:
            ax1.plot(numeric_labels, model_eval['Test Score'], label='Test Score', 
                    marker='s', linewidth=2, markersize=8)
        
        ax1.set_title('Model Evaluation - Line Plot', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Model Index')
        ax1.set_ylabel('Score (R²)')
        if 'Test Score' in model_eval.columns:
            ax1.set_ylim(max(0, model_eval['Test Score'].min() - 0.1), 
                        min(1, model_eval['Test Score'].max() + 0.1))
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(numeric_labels)
        ax1.set_xticklabels(model_eval['Model'], rotation=45, ha='right')
        
        # Add reference lines
        if 'Train Score' in model_eval.columns:
            first_train_score = model_eval['Train Score'].iloc[0]
            ax1.axhline(y=first_train_score, color='blue', linestyle='--', alpha=0.7, linewidth=1)
        
        if 'Test Score' in model_eval.columns:
            first_test_score = model_eval['Test Score'].iloc[0]
            ax1.axhline(y=first_test_score, color='orange', linestyle='--', alpha=0.7, linewidth=1)
        
        # 2. Bar plot comparison
        ax2 = axes[1]
        if 'Train Score' in model_eval.columns and 'Test Score' in model_eval.columns:
            x = np.arange(len(model_eval))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, model_eval['Train Score'], width, 
                           label='Train Score', alpha=0.8, color='lightblue')
            bars2 = ax2.bar(x + width/2, model_eval['Test Score'], width, 
                           label='Test Score', alpha=0.8, color='lightcoral')
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax2.set_xlabel('Models')
            ax2.set_ylabel('Score (R²)')
            ax2.set_title('Model Evaluation - Bar Comparison', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(model_eval['Model'], rotation=45, ha='right')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
        else:
            # Single score bar plot
            score_col = 'Test Score' if 'Test Score' in model_eval.columns else 'Train Score'
            bars = ax2.bar(range(len(model_eval)), model_eval[score_col], 
                          color='skyblue', alpha=0.7)
            
            # Highlight best model
            best_idx = model_eval[score_col].idxmax()
            bars[best_idx].set_color('gold')
            
            ax2.set_xlabel('Models')
            ax2.set_ylabel(f'{score_col} (R²)')
            ax2.set_title(f'Model Evaluation - {score_col}', fontsize=14, fontweight='bold')
            ax2.set_xticks(range(len(model_eval)))
            ax2.set_xticklabels(model_eval['Model'], rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"✅ Model evaluation plot saved to {save_path}")
        
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"❌ Failed to create model evaluation plot: {e}")
        return False

def actVpre_enhanced(Y_test: np.ndarray, Y_pred: np.ndarray, model_name: str = "Model", 
                    save_path: str = None):
    """Enhanced version of your actVpre function"""
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        # Calculate metrics
        r2 = r2_score(Y_test, Y_pred)
        mse = mean_squared_error(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        rmse = np.sqrt(mse)
        
        # Create the plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Scatter plot
        ax1 = axes[0]
        scatter = ax1.scatter(Y_test, Y_pred, color='blue', edgecolor='navy', alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(Y_test.min(), Y_pred.min())
        max_val = max(Y_test.max(), Y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, 
                label='Perfect Prediction')
        
        # Add trend line
        z = np.polyfit(Y_test, Y_pred, 1)
        p = np.poly1d(z)
        ax1.plot(Y_test, p(Y_test), "g--", alpha=0.8, linewidth=2, 
                label=f'Trend (slope={z[0]:.3f})')
        
        ax1.set_xlabel('Actual Values', fontsize=12)
        ax1.set_ylabel('Predicted Values', fontsize=12)
        ax1.set_title(f'Actual vs Predicted: {model_name}', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add metrics
        metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top', fontsize=11)
        
        # 2. Residuals plot
        ax2 = axes[1]
        residuals = Y_test - Y_pred
        ax2.scatter(Y_pred, residuals, color='red', edgecolor='darkred', alpha=0.6, s=50)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.axhline(y=residuals.std(), color='orange', linestyle='--', alpha=0.7)
        ax2.axhline(y=-residuals.std(), color='orange', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Predicted Values', fontsize=12)
        ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
        ax2.set_title(f'Residuals Plot: {model_name}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add residual stats
        residual_stats = f'Mean: {residuals.mean():.4f}\nStd: {residuals.std():.4f}'
        ax2.text(0.05, 0.95, residual_stats, transform=ax2.transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                verticalalignment='top', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"✅ Actual vs Predicted plot saved to {save_path}")
        
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"❌ Failed to create actual vs predicted plot: {e}")
        return False

# Test function
def test_enhanced_visualizations():
    """Test the enhanced visualization functions"""
    try:
        import pandas as pd
        import numpy as np
        
        print("🧪 Testing Enhanced Visualizations...")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'feature1': np.random.normal(50, 15, n_samples),
            'feature2': np.random.normal(30, 10, n_samples),
            'feature3': np.random.uniform(0, 100, n_samples),
            'feature4': np.random.exponential(2, n_samples),
        }
        
        # Create target with some relationship
        data['target'] = (data['feature1'] * 0.5 + data['feature2'] * 0.3 + 
                         data['feature3'] * 0.1 + np.random.normal(0, 10, n_samples))
        
        df = pd.DataFrame(data)
        
        # Test model results
        model_results = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'LightGBM'],
            'Train Score': [0.72, 0.91, 0.94, 0.92],
            'Test Score': [0.71, 0.85, 0.89, 0.87],
            'CV Score': [0.70, 0.84, 0.88, 0.86],
            'Training Time': [0.5, 2.3, 5.7, 3.2]
        })
        
        print("✅ Sample data created")
        
        # Test functions
        test_results = []
        
        # Test correlation heatmap
        try:
            result = create_correlation_heatmap(df, 'test_correlation.png')
            test_results.append(("Correlation Heatmap", result))
        except Exception as e:
            test_results.append(("Correlation Heatmap", False))
            print(f"❌ Correlation test failed: {e}")
        
        # Test high correlation detection
        try:
            high_corr = get_high_corr_columns(df, 0.3)
            test_results.append(("High Correlation Detection", len(high_corr) >= 0))
        except Exception as e:
            test_results.append(("High Correlation Detection", False))
        
        # Test distribution plots
        try:
            result = plot_numerical_distributions(df, 'test_distributions.png')
            test_results.append(("Distribution Plots", result))
        except Exception as e:
            test_results.append(("Distribution Plots", False))
        
        # Test model comparison
        try:
            result = create_quick_comparison(model_results, 'test_comparison.png')
            test_results.append(("Model Comparison", result))
        except Exception as e:
            test_results.append(("Model Comparison", False))
        
        # Test model evaluation
        try:
            result = plot_model_evaluation(model_results, 'test_evaluation.png')
            test_results.append(("Model Evaluation", result))
        except Exception as e:
            test_results.append(("Model Evaluation", False))
        
        # Test actual vs predicted
        try:
            y_true = np.random.normal(100, 20, 500)
            y_pred = y_true + np.random.normal(0, 10, 500)  # Add some noise
            result = actVpre_enhanced(y_true, y_pred, "Test Model", 'test_actual_vs_pred.png')
            test_results.append(("Actual vs Predicted", result))
        except Exception as e:
            test_results.append(("Actual vs Predicted", False))
        
        # Print test results
        print("\n📊 Test Results:")
        print("=" * 50)
        for test_name, success in test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name:<25} {status}")
        
        passed_tests = sum([1 for _, success in test_results if success])
        total_tests = len(test_results)
        
        print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! Enhanced visualizations are working correctly.")
        else:
            print("⚠️  Some tests failed. Check the error messages above.")
        
        return passed_tests == total_tests
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return False

if __name__ == "__main__":
    test_enhanced_visualizations()

"""
USAGE EXAMPLES:

# 1. Basic model comparison
import pandas as pd
results = pd.DataFrame({
    'Model': ['Linear', 'RF', 'XGB'],
    'Test Score': [0.7, 0.85, 0.89]
})
create_quick_comparison(results, 'comparison.png')

# 2. Correlation analysis
import pandas as pd
df = pd.read_csv('data.csv')
create_correlation_heatmap(df, 'correlation.png')
high_corr_cols = get_high_corr_columns(df, 0.8)

# 3. Distribution analysis
plot_numerical_distributions(df, 'distributions.png')

# 4. Outlier comparison
df_cleaned = df.dropna()  # or your cleaning process
plot_outliers_comparison(df, df_cleaned, 'price', 'outliers.png')

# 5. Model evaluation
model_eval = pd.DataFrame({
    'Model': ['Model1', 'Model2'],
    'Train Score': [0.9, 0.85],
    'Test Score': [0.8, 0.83]
})
plot_model_evaluation(model_eval, 'evaluation.png')

# 6. Prediction analysis
import numpy as np
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
actVpre_enhanced(y_true, y_pred, 'My Model', 'predictions.png')

# 7. Full comprehensive analysis
config = {'pipeline': {}}
visualizer = ComprehensiveVisualizer(config)

# Create dataset overview
visualizer.plot_dataset_overview(df, 'dataset_overview.png')

# Create comprehensive model comparison
visualizer.plot_model_comparison(results_df, 'comprehensive_comparison.png')

# Create complete analysis report
job_state = {
    'job_info': {'job_id': 'test_job', 'dataset_name': 'test.csv'},
    'model_results': {'all_models_performance': {...}},
    # ... other job state data
}
report_files = visualizer.create_complete_analysis_report(job_state, 'analysis_report/')

""""""
utils/visualization.py - ENHANCED VERSION
Comprehensive visualization utilities for AutoML results
Includes preprocessing, feature engineering, and model analysis plots
"""

import matplotlib
# Set backend first, before importing pyplot
try:
    matplotlib.use('Agg')  # Use non-GUI backend for compatibility
except Exception:
    pass

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import os
import warnings
from pathlib import Path

# Import scipy at module level
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available - some plots may be limited")

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
try:
    sns.set_palette("husl")
except Exception:
    pass

class ComprehensiveVisualizer:
    """Comprehensive visualization utilities for AutoML pipeline"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set default figure parameters with error handling
        try:
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['figure.dpi'] = 100
            plt.rcParams['savefig.dpi'] = 300
            plt.rcParams['font.size'] = 10
            plt.rcParams['savefig.format'] = 'png'
        except Exception as e:
            self.logger.warning(f"Could not set matplotlib parameters: {e}")
    
    def _safe_save_plot(self, save_path: str, fig=None):
        """Safely save plot with error handling"""
        try:
            if save_path:
                if fig:
                    fig.savefig(save_path, bbox_inches='tight', dpi=300, 
                               facecolor='white', edgecolor='none')
                else:
                    plt.savefig(save_path, bbox_inches='tight', dpi=300,
                               facecolor='white', edgecolor='none')
                self.logger.info(f"Plot saved to {save_path}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to save plot to {save_path}: {e}")
            return False
        return False
    
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str] = None) -> bool:
        """Validate dataframe"""
        if df.empty:
            self.logger.error("DataFrame is empty")
            return False
        
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing columns: {missing_cols}")
                return False
        
        return True

    # ============================================================================
    # DATA EXPLORATION AND PROFILING PLOTS
    # ============================================================================
    
    def plot_dataset_overview(self, df: pd.DataFrame, save_path: str = None) -> str:
        """Create comprehensive dataset overview"""
        try:
            if not self._validate_dataframe(df):
                return None
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Dataset Overview & Profiling', fontsize=16, fontweight='bold')
            
            # 1. Data types distribution
            ax1 = axes[0, 0]
            dtype_counts = df.dtypes.value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(dtype_counts)))
            bars = ax1.bar(range(len(dtype_counts)), dtype_counts.values, color=colors)
            ax1.set_xlabel('Data Types')
            ax1.set_ylabel('Count')
            ax1.set_title('Data Types Distribution')
            ax1.set_xticks(range(len(dtype_counts)))
            ax1.set_xticklabels([str(dt) for dt in dtype_counts.index], rotation=45)
            
            # Add value labels
            for bar, count in zip(bars, dtype_counts.values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            # 2. Missing values heatmap
            ax2 = axes[0, 1]
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                bars = ax2.barh(range(len(missing_data)), missing_data.values, color='lightcoral')
                ax2.set_yticks(range(len(missing_data)))
                ax2.set_yticklabels(missing_data.index)
                ax2.set_xlabel('Missing Values Count')
                ax2.set_title('Missing Values by Column')
                
                # Add percentage labels
                total_rows = len(df)
                for i, (bar, count) in enumerate(zip(bars, missing_data.values)):
                    pct = (count / total_rows) * 100
                    ax2.text(count + max(missing_data.values) * 0.01, bar.get_y() + bar.get_height()/2,
                            f'{pct:.1f}%', va='center', fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=14)
                ax2.set_title('Missing Values by Column')
            
            # 3. Dataset shape and statistics
            ax3 = axes[0, 2]
            ax3.axis('off')
            
            # Basic statistics
            stats_text = f"""
            📊 Dataset Statistics
            
            Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
            Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
            
            📈 Column Types:
            Numeric: {len(df.select_dtypes(include=[np.number]).columns)}
            Object: {len(df.select_dtypes(include=['object']).columns)}
            DateTime: {len(df.select_dtypes(include=['datetime']).columns)}
            
            ❓ Data Quality:
            Missing Values: {df.isnull().sum().sum():,}
            Duplicate Rows: {df.duplicated().sum():,}
            Complete Rows: {(~df.isnull().any(axis=1)).sum():,}
            """
            
            ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # 4. Correlation matrix (numeric columns only)
            ax4 = axes[1, 0]
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax4)
                ax4.set_title('Correlation Matrix (Numeric Features)')
            else:
                ax4.text(0.5, 0.5, 'Insufficient Numeric Columns\nfor Correlation Analysis', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Correlation Matrix')
            
            # 5. Unique values distribution
            ax5 = axes[1, 1]
            unique_counts = [df[col].nunique() for col in df.columns]
            unique_ratio = [count/len(df) for count in unique_counts]
            
            scatter = ax5.scatter(unique_counts, unique_ratio, alpha=0.6, s=50)
            ax5.set_xlabel('Number of Unique Values')
            ax5.set_ylabel('Unique Ratio (Unique/Total)')
            ax5.set_title('Column Uniqueness Analysis')
            ax5.set_xscale('log')
            ax5.grid(True, alpha=0.3)
            
            # Add annotations for interesting points
            for i, col in enumerate(df.columns):
                if unique_ratio[i] > 0.9 or unique_counts[i] == 1:  # High uniqueness or constant
                    ax5.annotate(col, (unique_counts[i], unique_ratio[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # 6. Data distribution summary
            ax6 = axes[1, 2]
            if len(numeric_df.columns) > 0:
                # Plot distribution of skewness values
                skewness_values = [numeric_df[col].skew() for col in numeric_df.columns]
                ax6.hist(skewness_values, bins=min(10, len(skewness_values)), 
                        alpha=0.7, color='lightgreen', edgecolor='black')
                ax6.axvline(x=0, color='red', linestyle='--', label='Normal Distribution')
                ax6.axvline(x=1, color='orange', linestyle='--', label='Moderate Skew')
                ax6.axvline(x=-1, color='orange', linestyle='--')
                ax6.set_xlabel('Skewness Values')
                ax6.set_ylabel('Count')
                ax6.set_title('Distribution Skewness Analysis')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'No Numeric Columns\nfor Skewness Analysis', 
                        ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Distribution Skewness Analysis')
            
            plt.tight_layout()
            self._safe_save_plot(save_path, fig)
            plt.close(fig)
            
            return save_path or "dataset_overview"
            
        except Exception as e:
            self.logger.error(f"Failed to create dataset overview: {e}")
            try:
                plt.close('all')
            except:
                pass
            return None
    
    def plot_all_numerical_distributions(self, df: pd.DataFrame, save_path: str = None) -> str:
        """Plot distribution of all numerical columns"""
        try:
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_columns) == 0:
                self.logger.warning("No numeric columns found for distribution plots")
                return None
            
            n_cols = 2
            n_rows = len(numeric_columns)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle('Numerical Columns Distribution Analysis', fontsize=16, fontweight='bold')
            
            for i, column in enumerate(numeric_columns):
                # Histogram with KDE
                sns.histplot(df[column], bins=50, color="#512DA8", ax=axes[i, 0], kde=True, alpha=0.7)
                axes[i, 0].set_xlabel(column.replace('_', ' ').title(), fontsize=12)
                axes[i, 0].set_ylabel("Frequency", fontsize=12)
                axes[i, 0].set_title(f"{column.replace('_', ' ').title()} Distribution", fontsize=12)
                axes[i, 0].grid(True, alpha=0.3)
                
                # Add statistics text
                mean_val = df[column].mean()
                std_val = df[column].std()
                skew_val = df[column].skew()
                
                stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nSkew: {skew_val:.2f}'
                axes[i, 0].text(0.02, 0.98, stats_text, transform=axes[i, 0].transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Box plot for outlier detection
                sns.boxplot(x=df[column], ax=axes[i, 1], color="#FF6B6B")
                axes[i, 1].set_xlabel(column.replace('_', ' ').title(), fontsize=12)
                axes[i, 1].set_title(f"{column.replace('_', ' ').title()} Box Plot (Outlier Detection)", fontsize=12)
                axes[i, 1].grid(True, alpha=0.3)
                
                # Add outlier statistics
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                
                outlier_text = f'Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)'
                axes[i, 1].text(0.02, 0.98, outlier_text, transform=axes[i, 1].transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            self._safe_save_plot(save_path, fig)
            plt.close(fig)
            
            return save_path or "numerical_distributions"
            
        except Exception as e:
            self.logger.error(f"Failed to create numerical distributions plot: {e}")
            try:
                plt.close('all')
            except:
                pass
            return None
    
    def plot_correlation_analysis(self, df: pd.DataFrame, threshold: float = 0.7, save_path: str = None) -> str:
        """Comprehensive correlation analysis"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) < 2:
                self.logger.warning("Insufficient numeric columns for correlation analysis")
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
            
            # 1. Full correlation heatmap
            ax1 = axes[0, 0]
            corr_matrix = numeric_df.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax1,
                       fmt='.2f', annot_kws={"size": 8})
            ax1.set_title('Correlation Heatmap')
            
            # 2. High correlation pairs
            ax2 = axes[0, 1]
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val >= threshold:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            if high_corr_pairs:
                pairs_df = pd.DataFrame(high_corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])
                pairs_df = pairs_df.sort_values('Correlation', ascending=False)
                
                y_pos = range(len(pairs_df))
                bars = ax2.barh(y_pos, pairs_df['Correlation'], color='lightcoral')
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels([f"{row['Feature 1']} vs {row['Feature 2']}" for _, row in pairs_df.iterrows()])
                ax2.set_xlabel('Correlation Coefficient')
                ax2.set_title(f'High Correlation Pairs (≥{threshold})')
                
                # Add value labels
                for bar, corr in zip(bars, pairs_df['Correlation']):
                    ax2.text(corr + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{corr:.3f}', va='center', fontweight='bold')
            else:
                ax2.text(0.5, 0.5, f'No correlation pairs\nabove threshold {threshold}', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title(f'High Correlation Pairs (≥{threshold})')
            
            # 3. Correlation distribution
            ax3 = axes[1, 0]
            # Get upper triangle correlations (excluding diagonal)
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            correlations = upper_triangle.stack().values
            
            ax3.hist(correlations, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
            ax3.axvline(x=-threshold, color='red', linestyle='--')
            ax3.axvline(x=0, color='green', linestyle='-', alpha=0.5, label='No Correlation')
            ax3.set_xlabel('Correlation Coefficient')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Correlation Coefficients')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Correlation strength summary
            ax4 = axes[1, 1]
            
            # Categorize correlations
            strong_pos = np.sum(correlations >= 0.7)
            moderate_pos = np.sum((correlations >= 0.3) & (correlations < 0.7))
            weak_pos = np.sum((correlations >= 0.1) & (correlations < 0.3))
            very_weak = np.sum((correlations >= -0.1) & (correlations < 0.1))
            weak_neg = np.sum((correlations >= -0.3) & (correlations < -0.1))
            moderate_neg = np.sum((correlations >= -0.7) & (correlations < -0.3))
            strong_neg = np.sum(correlations < -0.7)
            
            categories = ['Strong\nPositive\n(≥0.7)', 'Moderate\nPositive\n(0.3-0.7)', 'Weak\nPositive\n(0.1-0.3)',
                         'Very Weak\n(-0.1-0.1)', 'Weak\nNegative\n(-0.3--0.1)', 'Moderate\nNegative\n(-0.7--0.3)', 'Strong\nNegative\n(<-0.7)']
            counts = [strong_pos, moderate_pos, weak_pos, very_weak, weak_neg, moderate_neg, strong_neg]
            colors = ['darkgreen', 'green', 'lightgreen', 'gray', 'lightcoral', 'red', 'darkred']
            
            bars = ax4.bar(categories, counts, color=colors, alpha=0.7)
            ax4.set_ylabel('Count')
            ax4.set_title('Correlation Strength Distribution')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            self._safe_save_plot(save_path, fig)
            plt.close(fig)
            
            return save_path or "correlation_analysis"
            
        except Exception as e:
            self.logger.error(f"Failed to create correlation analysis: {e}")
            try:
                plt.close('all')
            except:
                pass
            return None

    # ============================================================================
    # PREPROCESSING VISUALIZATION
    # ============================================================================
    
    def plot_outlier_analysis(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame = None, 
                             save_path: str = None) -> str:
        """Comprehensive outlier analysis and comparison"""
        try:
            numeric_cols = df_original.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                self.logger.warning("No numeric columns for outlier analysis")
                return None
            
            # Limit to first 6 columns for readability
            display_cols = numeric_cols[:6]
            n_cols = len(display_cols)
            
            if df_cleaned is not None:
                fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 10))
                fig.suptitle('Outlier Analysis: Before vs After Cleaning', fontsize=16, fontweight='bold')
                
                for i, col in enumerate(display_cols):
                    # Before cleaning
                    ax_before = axes[0, i] if n_cols > 1 else axes[0]
                    sns.boxplot(y=df_original[col], ax=ax_before, color='lightcoral')
                    ax_before.set_title(f'{col}\nBefore Outlier Removal')
                    ax_before.grid(True, alpha=0.3)
                    
                    # Calculate outlier statistics
                    Q1 = df_original[col].quantile(0.25)
                    Q3 = df_original[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers_before = df_original[(df_original[col] < lower_bound) | (df_original[col] > upper_bound)]
                    
                    stats_text = f'Outliers: {len(outliers_before)}\n({len(outliers_before)/len(df_original)*100:.1f}%)'
                    ax_before.text(0.02, 0.98, stats_text, transform=ax_before.transAxes, 
                                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # After cleaning (if available)
                    if col in df_cleaned.columns:
                        ax_after = axes[1, i] if n_cols > 1 else axes[1]
                        sns.boxplot(y=df_cleaned[col], ax=ax_after, color='lightgreen')
                        ax_after.set_title(f'{col}\nAfter Outlier Removal')
                        ax_after.grid(True, alpha=0.3)
                        
                        # Calculate outlier statistics for cleaned data
                        Q1_clean = df_cleaned[col].quantile(0.25)
                        Q3_clean = df_cleaned[col].quantile(0.75)
                        IQR_clean = Q3_clean - Q1_clean
                        lower_bound_clean = Q1_clean - 1.5 * IQR_clean
                        upper_bound_clean = Q3_clean + 1.5 * IQR_clean
                        outliers_after = df_cleaned[(df_cleaned[col] < lower_bound_clean) | (df_cleaned[col] > upper_bound_clean)]
                        
                        stats_text_clean = f'Outliers: {len(outliers_after)}\n({len(outliers_after)/len(df_cleaned)*100:.1f}%)'
                        ax_after.text(0.02, 0.98, stats_text_clean, transform=ax_after.transAxes, 
                                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    else:
                        ax_after = axes[1, i] if n_cols > 1 else axes[1]
                        ax_after.text(0.5, 0.5, 'Column removed\nduring cleaning', 
                                     ha='center', va='center', transform=ax_after.transAxes)
                        ax_after.set_title(f'{col}\nAfter Cleaning (Removed)')
            
            else:
                # Only original data
                fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 6))
                fig.suptitle('Outlier Analysis', fontsize=16, fontweight='bold')
                
                if n_cols == 1:
                    axes = [axes]
                
                for i, col in enumerate(display_cols):
                    sns.boxplot(y=df_original[col], ax=axes[i], color='lightblue')
                    axes[i].set_title(f'{col}\nOutlier Detection')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Calculate and display outlier statistics
                    Q1 = df_original[col].quantile(0.25)
                    Q3 = df_original[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df_original[(df_original[col] < lower_bound) | (df_original[col] > upper_bound)]
                    
                    stats_text = f'Total: {len(df_original)}\nOutliers: {len(outliers)}\nPercentage: {len(outliers)/len(df_original)*100:.1f}%'
                    axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            self._safe_save_plot(save_path, fig)
            plt.close(fig)
            
            return save_path or "outlier_analysis"
            
        except Exception as e:
            self.logger.error(f"Failed to create outlier analysis: {e}")
            try:
                plt.close('all')
            except:
                pass
            return None
    
    def plot_missing_values_analysis(self, df_original: pd.DataFrame, df_processed: pd.DataFrame = None, 
                                   save_path: str = None) -> str:
        """Comprehensive missing values analysis"""
        try:
            if not self._validate_dataframe(df_original):
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Missing Values Analysis', fontsize=16, fontweight='bold')
            
            # 1. Missing values heatmap (original)
            ax1 = axes[0, 0]
            missing_data = df_original.isnull()
            if missing_data.any().any():
                sns.heatmap(missing_data, cbar=True, cmap='viridis', ax=ax1)
                ax1.set_title('Missing Values Heatmap (Original Data)')
                ax1.set_xlabel('Columns')
                ax1.set_ylabel('Rows (sample)')
            else:
                ax1.text(0.5, 0.5, 'No Missing Values\nin Original Data', 
                        ha='center', va='center', transform=ax1.transAxes, fontsize=14)
                ax1.set_title('Missing Values Heatmap (Original Data)')
            
            # 2. Missing values count by column
            ax2 = axes[0, 1]
            missing_counts = df_original.isnull().sum()
            missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=True)
            
            if len(missing_counts) > 0:
                bars = ax2.barh(range(len(missing_counts)), missing_counts.values, color='lightcoral')
                ax2.set_yticks(range(len(missing_counts)))
                ax2.set_yticklabels(missing_counts.index)
                ax2.set_xlabel('Missing Values Count')
                ax2.set_title('Missing Values by Column (Original)')
                
                # Add percentage labels
                total_rows = len(df_original)
                for i, (bar, count) in enumerate(zip(bars, missing_counts.values)):
                    pct = (count / total_rows) * 100
                    ax2.text(count + max(missing_counts.values) * 0.01, bar.get_y() + bar.get_height()/2,
                            f'{pct:.1f}%', va='center', fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=14)
                ax2.set_title('Missing Values by Column (Original)')
            
            # 3. Missing values pattern analysis
            ax3 = axes[1, 0]
            if missing_data.any().any():
                # Create missing value patterns
                missing_patterns = df_original.isnull().groupby(list(df_original.columns)).size().reset_index(name='count')
                missing_patterns = missing_patterns.sort_values('count', ascending=False).head(10)
                
                if len(missing_patterns) > 1:
                    ax3.bar(range(len(missing_patterns)), missing_patterns['count'], color='orange', alpha=0.7)
                    ax3.set_xlabel('Missing Value Patterns')
                    ax3.set_ylabel('Count')
                    ax3.set_title('Missing Value Patterns')
                    ax3.set_xticks(range(len(missing_patterns)))
                    ax3.set_xticklabels([f'Pattern {i+1}' for i in range(len(missing_patterns))], rotation=45)
                else:
                    ax3.text(0.5, 0.5, 'Single Missing\nValue Pattern', 
                            ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                    ax3.set_title('Missing Value Patterns')
            else:
                ax3.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=14)
                ax3.set_title('Missing Value Patterns')
            
            # 4. Before vs After comparison (if processed data provided)
            ax4 = axes[1, 1]
            if df_processed is not None:
                original_missing = df_original.isnull().sum().sum()
                processed_missing = df_processed.isnull().sum().sum()
                
                categories = ['Original Data', 'After Processing']
                missing_counts_comp = [original_missing, processed_missing]
                colors = ['lightcoral', 'lightgreen']
                
                bars = ax4.bar(categories, missing_counts_comp, color=colors, alpha=0.7)
                ax4.set_ylabel('Total Missing Values')
                ax4.set_title('Missing Values: Before vs After Processing')
                
                # Add value labels
                for bar, count in zip(bars, missing_counts_comp):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(missing_counts_comp) * 0.01,
                            str(count), ha='center', va='bottom', fontweight='bold')
                
                # Add improvement text
                if original_missing > 0:
                    improvement = ((original_missing - processed_missing) / original_missing) * 100
                    ax4.text(0.5, 0.95, f'Improvement: {improvement:.1f}%', 
                            transform=ax4.transAxes, ha='center', va='top',
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            else:
                # Show missing value statistics
                total_missing = df_original.isnull().sum().sum()
                total_cells = df_original.size
                missing_percentage = (total_missing / total_cells) * 100
                
                stats_text = f"""
                Missing Value Statistics
                
                Total Missing: {total_missing:,}
                Total Cells: {total_cells:,}
                Missing %: {missing_percentage:.2f}%
                
                Columns with Missing:
                {len(missing_counts)} out of {len(df_original.columns)}
                """
                
                ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                ax4.set_title('Missing Values Statistics')
                ax4.axis('off')
            
            plt.tight_layout()
            self._safe_save_plot(save_path, fig)
            plt.close(fig)
            
            return save_path or "missing_values_analysis"
            
        except Exception as e:
            self.logger.error(f"Failed to create missing values analysis: {e}")
            try:
                plt.close('all')
            except:
                pass
            return None
    
    def plot_preprocessing_summary(self, preprocessing_results: Dict, save_path: str = None) -> str:
        """Comprehensive preprocessing results summary"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Preprocessing Pipeline Summary', fontsize=16, fontweight='bold')
            
            # 1. Data shape changes
            ax1 = axes[0, 0]
            if 'shape_changes' in preprocessing_results:
                shape_data = preprocessing_results['shape_changes']
                stages = list(shape_data.keys())
                rows = [shape_data[stage][0] for stage in stages]
                cols = [shape_data[stage][1] for stage in stages]
                
                x = np.arange(len(stages))
                width = 0.35
                
                bars1 = ax1.bar(x - width/2, rows, width, label='Rows', color='lightblue')
                bars2 = ax1.bar(x + width/2, cols, width, label='Columns', color='lightcoral')
                
                ax1.set_xlabel('Pipeline Stages')
                ax1.set_ylabel('Count')
                ax1.set_title('Data Shape Changes Through Pipeline')
                ax1.set_xticks(x)
                ax1.set_xticklabels(stages, rotation=45)
                ax1.legend()
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + max(max(rows), max(cols)) * 0.01,
                                f'{int(height)}', ha='center', va='bottom', fontsize=9)
            else:
                ax1.text(0.5, 0.5, 'Shape change data\nnot available', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Data Shape Changes Through Pipeline')
            
            # 2. Features removed summary
            ax2 = axes[0, 1]
            removal_reasons = []
            removal_counts = []
            
            if 'features_removed' in preprocessing_results:
                features_removed = preprocessing_results['features_removed']
                if 'high_correlation' in features_removed:
                    removal_reasons.append('High\nCorrelation')
                    removal_counts.append(len(features_removed['high_correlation']))
                if 'constant_features' in features_removed:
                    removal_reasons.append('Constant\nValues')
                    removal_counts.append(len(features_removed['constant_features']))
                if 'high_missing' in features_removed:
                    removal_reasons.append('High Missing\nValues')
                    removal_counts.append(len(features_removed['high_missing']))
            
            if removal_reasons:
                colors = plt.cm.Set3(np.linspace(0, 1, len(removal_reasons)))
                bars = ax2.bar(removal_reasons, removal_counts, color=colors)
                ax2.set_ylabel('Number of Features Removed')
                ax2.set_title('Features Removed by Reason')
                
                # Add value labels
                for bar, count in zip(bars, removal_counts):
                    if count > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                str(count), ha='center', va='bottom', fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No features removed\nor data not available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Features Removed by Reason')
            
            # 3. Outliers removed summary
            ax3 = axes[1, 0]
            if 'outliers_removed' in preprocessing_results:
                outlier_data = preprocessing_results['outliers_removed']
                if isinstance(outlier_data, dict):
                    columns = list(outlier_data.keys())
                    outlier_counts = list(outlier_data.values())
                    
                    if columns:
                        bars = ax3.bar(columns, outlier_counts, color='lightcoral', alpha=0.7)
                        ax3.set_xlabel('Columns')
                        ax3.set_ylabel('Outliers Removed')
                        ax3.set_title('Outliers Removed by Column')
                        ax3.tick_params(axis='x', rotation=45)
                        
                        # Add value labels
                        for bar, count in zip(bars, outlier_counts):
                            if count > 0:
                                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(outlier_counts) * 0.01,
                                        str(count), ha='center', va='bottom', fontweight='bold')
                    else:
                        ax3.text(0.5, 0.5, 'No outliers removed', ha='center', va='center', 
                                transform=ax3.transAxes)
                        ax3.set_title('Outliers Removed by Column')
                else:
                    # Simple total count
                    ax3.bar(['Total Outliers'], [outlier_data], color='lightcoral')
                    ax3.set_title('Total Outliers Removed')
                    ax3.text(0, outlier_data + outlier_data * 0.05, str(outlier_data), 
                            ha='center', va='bottom', fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'Outlier removal data\nnot available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Outliers Removed')
            
            # 4. Processing statistics summary
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Compile summary statistics
            summary_text = "📊 Preprocessing Summary\n\n"
            
            if 'original_shape' in preprocessing_results and 'final_shape' in preprocessing_results:
                orig_rows, orig_cols = preprocessing_results['original_shape']
                final_rows, final_cols = preprocessing_results['final_shape']
                
                row_change = orig_rows - final_rows
                col_change = orig_cols - final_cols
                
                summary_text += f"📈 Data Shape Changes:\n"
                summary_text += f"  Rows: {orig_rows:,} → {final_rows:,} ({-row_change:+,})\n"
                summary_text += f"  Columns: {orig_cols} → {final_cols} ({-col_change:+})\n\n"
            
            if 'transformations_applied' in preprocessing_results:
                transformations = preprocessing_results['transformations_applied']
                summary_text += f"🔧 Transformations Applied:\n"
                for transformation in transformations:
                    summary_text += f"  ✓ {transformation}\n"
                summary_text += "\n"
            
            if 'processing_time' in preprocessing_results:
                processing_time = preprocessing_results['processing_time']
                summary_text += f"⏱️ Processing Time: {processing_time:.2f} seconds\n\n"
            
            if 'data_quality_score' in preprocessing_results:
                quality_score = preprocessing_results['data_quality_score']
                summary_text += f"🎯 Data Quality Score: {quality_score:.2f}/10\n"
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                    verticalalignment='top', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            plt.tight_layout()
            self._safe_save_plot(save_path, fig)
            plt.close(fig)
            
            return save_path or "preprocessing_summary"
            
        except Exception as e:
            self.logger.error(f"Failed to create preprocessing summary: {e}")
            try:
                plt.close('all')
            except:
                pass
            return None

    # ============================================================================
    # FEATURE ENGINEERING VISUALIZATION
    # ============================================================================
    
    def plot_feature_engineering_analysis(self, original_features: List[str], 
                                         engineered_features: List[str], 
                                         feature_importance: Dict = None,
                                         save_path: str = None) -> str:
        """Comprehensive feature engineering analysis"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Feature Engineering Analysis', fontsize=16, fontweight='bold')
            
            # 1. Feature count comparison
            ax1 = axes[0, 0]
            feature_counts = [len(original_features), len(engineered_features)]
            categories = ['Original Features', 'After Engineering']
            colors = ['lightblue', 'lightgreen']
            
            bars = ax1.bar(categories, feature_counts, color=colors, alpha=0.7)
            ax1.set_ylabel('Number of Features')
            ax1.set_title('Feature Count: Before vs After Engineering')
            
            # Add value labels and improvement
            for bar, count in zip(bars, feature_counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(feature_counts) * 0.01,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            improvement = len(engineered_features) - len(original_features)
            ax1.text(0.5, 0.95, f'Added: {improvement} features ({improvement/len(original_features)*100:.1f}% increase)', 
                    transform=ax1.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            # 2. Feature types analysis
            ax2 = axes[0, 1]
            
            # Categorize engineered features
            polynomial_features = [f for f in engineered_features if '^' in f or '*' in f]
            interaction_features = [f for f in engineered_features if '_x_' in f or ' * ' in f]
            statistical_features = [f for f in engineered_features if any(stat in f.lower() for stat in ['mean', 'std', 'min', 'max', 'sum'])]
            original_remaining = [f for f in engineered_features if f in original_features]
            other_features = [f for f in engineered_features if f not in polynomial_features + interaction_features + statistical_features + original_remaining]
            
            feature_types = ['Original', 'Polynomial', 'Interaction', 'Statistical', 'Other']
            type_counts = [len(original_remaining), len(polynomial_features), len(interaction_features), 
                          len(statistical_features), len(other_features)]
            
            colors_pie = plt.cm.Set3(np.linspace(0, 1, len(feature_types)))
            wedges, texts, autotexts = ax2.pie(type_counts, labels=feature_types, colors=colors_pie, 
                                              autopct='%1.1f%%', startangle=90)
            ax2.set_title('Feature Types Distribution')
            
            # 3. Feature importance (if available)
            ax3 = axes[1, 0]
            if feature_importance:
                # Sort features by importance
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                top_features = sorted_features[:15]  # Top 15 features
                
                feature_names = [f[0] for f in top_features]
                importance_values = [f[1] for f in top_features]
                
                # Color code by feature type
                colors_importance = []
                for fname in feature_names:
                    if fname in original_features:
                        colors_importance.append('blue')
                    elif '^' in fname or '*' in fname:
                        colors_importance.append('red')
                    elif '_x_' in fname:
                        colors_importance.append('green')
                    elif any(stat in fname.lower() for stat in ['mean', 'std', 'min', 'max']):
                        colors_importance.append('orange')
                    else:
                        colors_importance.append('purple')
                
                bars = ax3.barh(range(len(feature_names)), importance_values, color=colors_importance, alpha=0.7)
                ax3.set_yticks(range(len(feature_names)))
                ax3.set_yticklabels([fname[:20] + '...' if len(fname) > 20 else fname for fname in feature_names])
                ax3.set_xlabel('Importance Score')
                ax3.set_title('Top 15 Feature Importance')
                ax3.invert_yaxis()
                
                # Add legend for colors
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='blue', label='Original'),
                    Patch(facecolor='red', label='Polynomial'),
                    Patch(facecolor='green', label='Interaction'),
                    Patch(facecolor='orange', label='Statistical'),
                    Patch(facecolor='purple', label='Other')
                ]
                ax3.legend(handles=legend_elements, loc='lower right')
                
            else:
                ax3.text(0.5, 0.5, 'Feature importance\nnot available', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Feature Importance Analysis')
            
            # 4. Feature engineering summary
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            summary_text = "🔧 Feature Engineering Summary\n\n"
            summary_text += f"📊 Original Features: {len(original_features)}\n"
            summary_text += f"📈 Engineered Features: {len(engineered_features)}\n"
            summary_text += f"➕ Features Added: {len(engineered_features) - len(original_features)}\n\n"
            
            summary_text += f"🎯 Feature Types Created:\n"
            if len(polynomial_features) > 0:
                summary_text += f"  • Polynomial: {len(polynomial_features)}\n"
            if len(interaction_features) > 0:
                summary_text += f"  • Interaction: {len(interaction_features)}\n"
            if len(statistical_features) > 0:
                summary_text += f"  • Statistical: {len(statistical_features)}\n"
            if len(other_features) > 0:
                summary_text += f"  • Other: {len(other_features)}\n"
            
            if feature_importance:
                # Find most important engineered feature
                engineered_importance = {k: v for k, v in feature_importance.items() if k not in original_features}
                if engineered_importance:
                    best_engineered = max(engineered_importance.items(), key=lambda x: x[1])
                    summary_text += f"\n🏆 Best Engineered Feature:\n"
                    summary_text += f"  {best_engineered[0][:30]}...\n" if len(best_engineered[0]) > 30 else f"  {best_engineered[0]}\n"
                    summary_text += f"  Importance: {best_engineered[1]:.4f}\n"
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                    verticalalignment='top', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            
            plt.tight_layout()
            self._safe_save_plot(save_path, fig)
            plt.close(fig)
            
            return save_path or "feature_engineering_analysis"
            
        except Exception as e:
            self.logger.error(f"Failed to create feature engineering analysis: {e}")
            try:
                plt.close('all')
            except:
                pass
            return None

    # ============================================================================
    # MODEL PERFORMANCE VISUALIZATION
    # ============================================================================
    
    def plot_model_comparison(self, results_df: pd.DataFrame, save_path: str = None, 
                            title: str = "Model Performance Comparison") -> str:
        """Enhanced model comparison with comprehensive analysis"""
        try:
            required_cols = ['Model', 'Test Score']
            if not self._validate_dataframe(results_df, required_cols):
                return None
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            results_sorted = results_df.sort_values('Test Score', ascending=True)
            
            # 1. Horizontal bar chart of test scores
            ax1 = axes[0, 0]
            colors = plt.cm.viridis(np.linspace(0, 1, len(results_sorted)))
            bars = ax1.barh(results_sorted['Model'], results_sorted['Test Score'], color=colors)
            ax1.set_xlabel('Test Score (R²)')
            ax1.set_title('Test Performance Ranking')
            ax1.grid(axis='x', alpha=0.3)
            
            # Add value labels and rank
            for i, (bar, score) in enumerate(zip(bars, results_sorted['Test Score'])):
                rank = len(results_sorted) - i
                ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{score:.3f} (#{rank})', va='center', fontweight='bold')
            
            # 2. Train vs Test scores scatter plot
            ax2 = axes[0, 1]
            if 'Train Score' in results_sorted.columns:
                scatter = ax2.scatter(results_sorted['Train Score'], results_sorted['Test Score'], 
                                    s=100, alpha=0.7, c=range(len(results_sorted)), cmap='viridis')
                
                # Add diagonal line (perfect fit)
                min_score = min(results_sorted['Train Score'].min(), results_sorted['Test Score'].min())
                max_score = max(results_sorted['Train Score'].max(), results_sorted['Test Score'].max())
                ax2.plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.5, label='Perfect Fit')
                
                # Add overfitting zone
                ax2.fill_between([min_score, max_score], [min_score, max_score], [max_score, max_score], 
                               alpha=0.2, color='red', label='Overfitting Zone')
                
                ax2.set_xlabel('Train Score (R²)')
                ax2.set_ylabel('Test Score (R²)')
                ax2.set_title('Overfitting Analysis')
                ax2.legend()
                ax2.grid(alpha=0.3)
                
                # Add model labels for extreme points
                for i, model in enumerate(results_sorted['Model']):
                    train_score = results_sorted['Train Score'].iloc[i]
                    test_score = results_sorted['Test Score'].iloc[i]
                    overfitting = train_score - test_score
                    
                    if overfitting > 0.1:  # High overfitting
                        ax2.annotate(f'{model}\n(Overfit: {overfitting:.3f})', 
                                   (train_score, test_score),
                                   xytext=(5, 5), textcoords='offset points', 
                                   fontsize=8, alpha=0.8,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            else:
                ax2.text(0.5, 0.5, 'Train scores not available', ha='center', va='center', 
                        transform=ax2.transAxes)
                ax2.set_title('Overfitting Analysis')
            
            # 3. Cross-validation scores with confidence intervals
            ax3 = axes[0, 2]
            if 'CV Score' in results_sorted.columns:
                x_pos = range(len(results_sorted))
                cv_scores = results_sorted['CV Score']
                cv_std = results_sorted.get('CV Std', [0] * len(results_sorted))
                
                bars = ax3.bar(x_pos, cv_scores, yerr=cv_std, capsize=5,
                              color=plt.cm.plasma(np.linspace(0, 1, len(results_sorted))),
                              alpha=0.7, edgecolor='black')
                
                ax3.set_xlabel('Models')
                ax3.set_ylabel('CV Score (R²)')
                ax3.set_title('Cross-Validation Performance')
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels(results_sorted['Model'], rotation=45, ha='right')
                ax3.grid(axis='y', alpha=0.3)
                
                # Add confidence level indicators
                for i, (bar, score, std) in enumerate(zip(bars, cv_scores, cv_std)):
                    confidence = 1 - (std / score) if score != 0 else 0
                    confidence_color = 'green' if confidence > 0.9 else 'orange' if confidence > 0.7 else 'red'
                    
                    ax3.text(bar.get_x() + bar.get_width()/2, score + std + 0.01, 
                            f'±{std:.3f}', ha='center', fontweight='bold', fontsize=8,
                            color=confidence_color)
            else:
                ax3.text(0.5, 0.5, 'CV scores not available', ha='center', va='center', 
                        transform=ax3.transAxes)
                ax3.set_title('Cross-Validation Performance')
            
            # 4. Training time vs Performance efficiency
            ax4 = axes[1, 0]
            if 'Training Time' in results_sorted.columns:
                training_times = results_sorted['Training Time']
                test_scores = results_sorted['Test Score']
                
                # Calculate efficiency score (performance per second)
                efficiency = test_scores / (training_times + 1e-6)  # Avoid division by zero
                
                scatter = ax4.scatter(training_times, test_scores, s=100, c=efficiency, 
                                    cmap='RdYlGn', alpha=0.7, edgecolors='black')
                
                ax4.set_xlabel('Training Time (seconds)')
                ax4.set_ylabel('Test Score (R²)')
                ax4.set_title('Efficiency: Performance vs Training Time')
                ax4.grid(alpha=0.3)
                
                # Add colorbar for efficiency
                cbar = plt.colorbar(scatter, ax=ax4)
                cbar.set_label('Efficiency (Score/Second)')
                
                # Highlight best efficiency
                best_efficiency_idx = efficiency.idxmax()
                best_model = results_sorted.loc[best_efficiency_idx, 'Model']
                best_time = results_sorted.loc[best_efficiency_idx, 'Training Time']
                best_score = results_sorted.loc[best_efficiency_idx, 'Test Score']
                
                ax4.scatter(best_time, best_score, s=200, c='gold', marker='*', 
                          edgecolors='black', linewidth=2, label=f'Most Efficient: {best_model}')
                ax4.legend()
                
            else:
                ax4.text(0.5, 0.5, 'Training times not available', ha='center', va='center', 
                        transform=ax4.transAxes)
                ax4.set_title('Efficiency Analysis')
            
            # 5. Performance distribution and statistics
            ax5 = axes[1, 1]
            test_scores = results_sorted['Test Score']
            
            # Histogram with statistics
            n, bins, patches = ax5.hist(test_scores, bins=min(10, len(test_scores)), 
                                       alpha=0.7, color='skyblue', edgecolor='black')
            
            # Color code bars based on performance
            for i, (patch, bin_start, bin_end) in enumerate(zip(patches, bins[:-1], bins[1:])):
                bin_center = (bin_start + bin_end) / 2
                if bin_center > test_scores.quantile(0.75):
                    patch.set_facecolor('green')
                elif bin_center > test_scores.median():
                    patch.set_facecolor('yellow')
                else:
                    patch.set_facecolor('red')
                patch.set_alpha(0.7)
            
            # Add statistics lines
            ax5.axvline(test_scores.mean(), color='blue', linestyle='--', linewidth=2, 
                       label=f'Mean: {test_scores.mean():.3f}')
            ax5.axvline(test_scores.median(), color='green', linestyle='--', linewidth=2, 
                       label=f'Median: {test_scores.median():.3f}')
            ax5.axvline(test_scores.max(), color='red', linestyle='--', linewidth=2, 
                       label=f'Best: {test_scores.max():.3f}')
            
            ax5.set_xlabel('Test Score (R²)')
            ax5.set_ylabel('Count')
            ax5.set_title('Performance Distribution')
            ax5.legend()
            ax5.grid(alpha=0.3)
            
            # 6. Model ranking with detailed metrics
            ax6 = axes[1, 2]
            ax6.axis('off')
            
            # Create ranking table
            ranking_data = results_sorted.sort_values('Test Score', ascending=False).head(5)
            
            table_text = "🏆 TOP 5 MODELS RANKING\n\n"
            for i, (_, row) in enumerate(ranking_data.iterrows(), 1):
                model_name = row['Model']
                test_score = row['Test Score']
                
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                table_text += f"{medal} {model_name}\n"
                table_text += f"    Score: {test_score:.4f}\n"
                
                if 'CV Score' in row:
                    table_text += f"    CV: {row['CV Score']:.4f}\n"
                if 'Training Time' in row:
                    table_text += f"    Time: {row['Training Time']:.1f}s\n"
                
                table_text += "\n"
            
            # Add overall statistics
            table_text += f"📊 OVERALL STATISTICS\n\n"
            table_text += f"Models Trained: {len(results_df)}\n"
            table_text += f"Best Score: {test_scores.max():.4f}\n"
            table_text += f"Worst Score: {test_scores.min():.4f}\n"
            table_text += f"Score Range: {test_scores.max() - test_scores.min():.4f}\n"
            table_text += f"Std Deviation: {test_scores.std():.4f}\n"
            
            if 'Training Time' in results_df.columns:
                total_time = results_df['Training Time'].sum()
                table_text += f"Total Time: {total_time:.1f}s\n"
            
            ax6.text(0.05, 0.95, table_text, transform=ax6.transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            plt.tight_layout()
            self._safe_save_plot(save_path, fig)
            plt.close(fig)
            
            return save_path or "model_comparison"
            
        except Exception as e:
            self.logger.error(f"Failed to create enhanced model comparison: {e}")
            try:
                plt.close('all')
            except:
                pass
            return None

    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                model_name: str = "Model", save_path: str = None) -> str:
        """Enhanced actual vs predicted plot with detailed analysis"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Prediction Analysis: {model_name}', fontsize=16, fontweight='bold')
            
            # 1. Actual vs Predicted scatter plot
            ax1 = axes[0, 0]
            
            # Calculate R² and other metrics
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            # Create scatter plot
            scatter = ax1.scatter(y_true, y_pred, alpha=0.6, s=30, color='blue', edgecolors='navy', linewidth=0.5)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            # Add trend line
            z = np.polyfit(y_true, y_pred, 1)
            p = np.poly1d(z)
            ax1.plot(y_true, p(y_true), "g--", alpha=0.8, linewidth=2, label=f'Trend Line (slope={z[0]:.3f})')
            
            ax1.set_xlabel('Actual Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title('Actual vs Predicted Values')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add metrics text box
            metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
            ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    verticalalignment='top', fontsize=10)
            
            # 2. Residuals plot
            ax2 = axes[0, 1]
            residuals = y_true - y_pred
            
            ax2.scatter(y_pred, residuals, alpha=0.6, s=30, color='red', edgecolors='darkred', linewidth=0.5)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.axhline(y=residuals.std(), color='orange', linestyle='--', alpha=0.7, label='+1 Std')
            ax2.axhline(y=-residuals.std(), color='orange', linestyle='--', alpha=0.7, label='-1 Std')
            
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals (Actual - Predicted)')
            ax2.set_title('Residuals Plot')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add residual statistics
            residual_stats = f'Mean: {residuals.mean():.4f}\nStd: {residuals.std():.4f}\nMax: {residuals.max():.4f}\nMin: {residuals.min():.4f}'
            ax2.text(0.05, 0.95, residual_stats, transform=ax2.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    verticalalignment='top', fontsize=9)
            
            # 3. Residuals distribution
            ax3 = axes[1, 0]
            
            # Histogram of residuals
            n, bins, patches = ax3.hist(residuals, bins=30, density=True, alpha=0.7, 
                                       color='skyblue', edgecolor='black')
            
            # Overlay normal distribution
            if SCIPY_AVAILABLE:
                mu, sigma = stats.norm.fit(residuals)
                x = np.linspace(residuals.min(), residuals.max(), 100)
                ax3.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                        label=f'Normal fit (μ={mu:.3f}, σ={sigma:.3f})')
                
                # Shapiro-Wilk test for normality
                _, p_value = stats.shapiro(residuals[:5000])  # Limit sample size for shapiro test
                normality_text = f'Shapiro-Wilk p-value: {p_value:.4f}'
                if p_value > 0.05:
                    normality_text += '\n(Residuals appear normal)'
                else:
                    normality_text += '\n(Residuals may not be normal)'
                
                ax3.text(0.05, 0.95, normality_text, transform=ax3.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                        verticalalignment='top', fontsize=9)
            
            ax3.axvline(x=0, color='green', linestyle='-', linewidth=2, label='Zero residual')
            ax3.set_xlabel('Residuals')
            ax3.set_ylabel('Density')
            ax3.set_title('Residuals Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Prediction error analysis
            ax4 = axes[1, 1]
            
            # Calculate percentage errors
            percentage_errors = np.abs((y_true - y_pred) / y_true) * 100
            percentage_errors = percentage_errors[np.isfinite(percentage_errors)]  # Remove inf/nan
            
            # Create error bins
            error_bins = [0, 5, 10, 20, 50, 100, np.inf]
            error_labels = ['<5%', '5-10%', '10-20%', '20-50%', '50-100%', '>100%']
            error_counts = []
            
            for i in range(len(error_bins)-1):
                if i == len(error_bins)-2:  # Last bin
                    count = np.sum(percentage_errors >= error_bins[i])
                else:
                    count = np.sum((percentage_errors >= error_bins[i]) & (percentage_errors < error_bins[i+1]))
                error_counts.append(count)
            
            # Create pie chart of error distribution
            colors = ['green', 'lightgreen', 'yellow', 'orange', 'red', 'darkred']
            valid_indices = [i for i, count in enumerate(error_counts) if count > 0]
            valid_counts = [error_counts[i] for i in valid_indices]
            valid_labels = [error_labels[i] for i in valid_indices]
            valid_colors = [colors[i] for i in valid_indices]
            
            if valid_counts:
                wedges, texts, autotexts = ax4.pie(valid_counts, labels=valid_labels, colors=valid_colors,
                                                  autopct='%1.1f%%', startangle=90)
                ax4.set_title('Prediction Error Distribution')
                
                # Add median error info
                median_error = np.median(percentage_errors)
                ax4.text(0.5, -1.3, f'Median Error: {median_error:.2f}%', 
                        transform=ax4.transAxes, ha='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            else:
                ax4.text(0.5, 0.5, 'Error analysis\nnot available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Prediction Error Distribution')
            
            plt.tight_layout()
            self._safe_save_plot(save_path, fig)
            plt.close(fig)
            
            return save_path or "actual_vs_predicted"
            
        except Exception as e:
            self.logger.error(f"Failed to create actual vs predicted plot: {e}")
            try:
                plt.close('all')
            except:
                pass
            return None

    def plot_learning_curves(self, train_scores: np.ndarray, val_scores: np.ndarray, 
                           train_sizes: np.ndarray, model_name: str, 
                           save_path: str = None) -> str:
        """Enhanced learning curves with detailed analysis"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Learning Curves Analysis: {model_name}', fontsize=16, fontweight='bold')
            
            # Calculate means and stds
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # 1. Standard learning curves
            ax1 = axes[0, 0]
            
            ax1.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
            ax1.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                           alpha=0.2, color='blue')
            
            ax1.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
            ax1.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                           alpha=0.2, color='red')
            
            ax1.set_xlabel('Training Set Size')
            ax1.set_ylabel('Score (R²)')
            ax1.set_title('Learning Curves')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Add convergence analysis
            if len(val_mean) > 2:
                val_improvement = val_mean[-1] - val_mean[-2]
                convergence_text = f'Recent improvement: {val_improvement:.4f}'
                if abs(val_improvement) < 0.001:
                    convergence_text += '\n(Converged)'
                else:
                    convergence_text += '\n(Still improving)'
                
                ax1.text(0.05, 0.95, convergence_text, transform=ax1.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                        verticalalignment='top', fontsize=9)
            
            # 2. Overfitting analysis
            ax2 = axes[0, 1]
            
            overfitting_gap = train_mean - val_mean
            ax2.plot(train_sizes, overfitting_gap, 'o-', color='orange', linewidth=2)
            ax2.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='No Overfitting')
            ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='High Overfitting')
            
            # Fill areas to show overfitting levels
            ax2.fill_between(train_sizes, 0, overfitting_gap, 
                           where=(overfitting_gap > 0.1), color='red', alpha=0.3, label='Severe')
            ax2.fill_between(train_sizes, 0, overfitting_gap, 
                           where=((overfitting_gap > 0.05) & (overfitting_gap <= 0.1)), 
                           color='orange', alpha=0.3, label='Moderate')
            ax2.fill_between(train_sizes, 0, overfitting_gap, 
                           where=(overfitting_gap <= 0.05), color='green', alpha=0.3, label='Low')
            
            ax2.set_xlabel('Training Set Size')
            ax2.set_ylabel('Overfitting Gap (Train - Val)')
            ax2.set_title('Overfitting Analysis')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            # 3. Score variance analysis
            ax3 = axes[1, 0]
            
            ax3.plot(train_sizes, train_std, 'o-', color='blue', label='Training Score Variance')
            ax3.plot(train_sizes, val_std, 'o-', color='red', label='Validation Score Variance')
            
            ax3.set_xlabel('Training Set Size')
            ax3.set_ylabel('Score Standard Deviation')
            ax3.set_title('Score Stability Analysis')
            ax3.legend()
            ax3.grid(alpha=0.3)
            
            # Add stability assessment
            final_val_std = val_std[-1] if len(val_std) > 0 else 0
            if final_val_std < 0.02:
                stability = "Very Stable"
                color = 'green'
            elif final_val_std < 0.05:
                stability = "Stable"
                color = 'yellow'
            else:
                stability = "Unstable"
                color = 'red'
            
            ax3.text(0.05, 0.95, f'Model Stability: {stability}\nFinal Val Std: {final_val_std:.4f}', 
                    transform=ax3.transAxes, 
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                    verticalalignment='top', fontsize=9)
            
            # 4. Recommendations and summary
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Generate recommendations based on learning curve analysis
            recommendations = []
            
            # Check final performance
            final_val_score = val_mean[-1] if len(val_mean) > 0 else 0
            final_train_score = train_mean[-1] if len(train_mean) > 0 else 0
            final_gap = final_train_score - final_val_score
            
            summary_text = f"📊 LEARNING CURVE ANALYSIS\n\n"
            summary_text += f"Final Training Score: {final_train_score:.4f}\n"
            summary_text += f"Final Validation Score: {final_val_score:.4f}\n"
            summary_text += f"Overfitting Gap: {final_gap:.4f}\n"
            summary_text += f"Validation Stability: {final_val_std:.4f}\n\n"
            
            summary_text += f"🎯 RECOMMENDATIONS:\n\n"
            
            # Overfitting recommendations
            if final_gap > 0.1:
                recommendations.append("• High overfitting detected")
                recommendations.append("• Consider regularization")
                recommendations.append("• Reduce model complexity")
                recommendations.append("• Increase training data")
            elif final_gap > 0.05:
                recommendations.append("• Moderate overfitting")
                recommendations.append("• Monitor with more data")
            else:
                recommendations.append("• Good generalization")
                
            # Stability recommendations
            if final_val_std > 0.05:
                recommendations.append("• High variance in scores")
                recommendations.append("• Consider ensemble methods")
                recommendations.append("• Increase cross-validation folds")
            
            # Performance recommendations
            if final_val_score < 0.5:
                recommendations.append("• Low validation performance")
                recommendations.append("• Try different algorithms")
                recommendations.append("• Feature engineering needed")
            elif final_val_score > 0.8:
                recommendations.append("• Excellent performance!")
                recommendations.append("• Model ready for production")
            
            # Convergence recommendations
            if len(val_mean) > 2:
                recent_improvement = val_mean[-1] - val_mean[-2]
                if recent_improvement > 0.01:
                    recommendations.append("• Still improving significantly")
                    recommendations.append("• Consider more training data")
                elif abs(recent_improvement) < 0.001:
                    recommendations.append("• Model has converged")
                    recommendations.append("• Additional data may not help")
            
            # Add recommendations to plot
            for rec in recommendations[:8]:  # Limit to 8 recommendations
                summary_text += f"{rec}\n"
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            
            plt.tight_layout()
            self._safe_save_plot(save_path, fig)
            plt.close(fig)
            
            return save_path or "learning_curves"
            
        except Exception as e:
            self.logger.error(f"Failed to create learning curves: {e}")
            try:
                plt.close('all')
            except:
                pass
            return None

    # ============================================================================
    # COMPREHENSIVE VISUALIZATION SUITE
    # ============================================================================
    
    def create_complete_analysis_report(self, job_state: Dict[str, Any], save_dir: str) -> Dict[str, str]:
        """Create comprehensive visual analysis report"""
        
        os.makedirs(save_dir, exist_ok=True)
        report_files = {}
        
        try:
            self.logger.info("Creating comprehensive analysis report...")
            
            # 1. Dataset Overview (if original data available)
            if 'original_data' in job_state:
                try:
                    overview_path = os.path.join(save_dir, 'dataset_overview.png')
                    self.plot_dataset_overview(job_state['original_data'], overview_path)
                    report_files['dataset_overview'] = overview_path
                except Exception as e:
                    self.logger.warning(f"Failed to create dataset overview: {e}")
            
            # 2. Preprocessing Analysis
            if 'preprocessing_results' in job_state:
                try:
                    preprocessing_path = os.path.join(save_dir, 'preprocessing_summary.png')
                    self.plot_preprocessing_summary(job_state['preprocessing_results'], preprocessing_path)
                    report_files['preprocessing_summary'] = preprocessing_path
                except Exception as e:
                    self.logger.warning(f"Failed to create preprocessing summary: {e}")
            
            # 3. Feature Engineering Analysis
            if 'feature_engineering_results' in job_state:
                try:
                    fe_results = job_state['feature_engineering_results']
                    fe_path = os.path.join(save_dir, 'feature_engineering_analysis.png')
                    self.plot_feature_engineering_analysis(
                        fe_results.get('original_features', []),
                        fe_results.get('engineered_features', []),
                        fe_results.get('feature_importance', {}),
                        fe_path
                    )
                    report_files['feature_engineering'] = fe_path
                except Exception as e:
                    self.logger.warning(f"Failed to create feature engineering analysis: {e}")
            
            # 4. Model Performance Analysis
            model_results = job_state.get('model_results', {})
            all_models = model_results.get('all_models_performance', {})
            
            if all_models:
                try:
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
                    
                    # Model comparison
                    comparison_path = os.path.join(save_dir, 'comprehensive_model_comparison.png')
                    self.plot_model_comparison(results_df, comparison_path, 
                                             f"Comprehensive Model Analysis - Job {job_state['job_info']['job_id']}")
                    report_files['model_comparison'] = comparison_path
                    
                    # Best model detailed analysis (if predictions available)
                    best_model = model_results.get('best_model', {})
                    if 'predictions' in best_model and 'actual' in best_model:
                        try:
                            prediction_path = os.path.join(save_dir, 'prediction_analysis.png')
                            self.plot_actual_vs_predicted(
                                best_model['actual'], 
                                best_model['predictions'],
                                best_model.get('model_name', 'Best Model'),
                                prediction_path
                            )
                            report_files['prediction_analysis'] = prediction_path
                        except Exception as e:
                            self.logger.warning(f"Failed to create prediction analysis: {e}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create model analysis: {e}")
            
            # 5. Create summary report
            try:
                summary_path = os.path.join(save_dir, 'analysis_summary.png')
                self.create_analysis_summary(job_state, report_files, summary_path)
                report_files['analysis_summary'] = summary_path
            except Exception as e:
                self.logger.warning(f"Failed to create analysis summary: {e}")
            
            self.logger.info(f"Comprehensive analysis report created with {len(report_files)} visualizations")
            
        except Exception as e:
            self.logger.error(f"Failed to create comprehensive analysis report: {str(e)}")
        
        return report_files
    
    def create_analysis_summary(self, job_state: Dict, report_files: Dict, save_path: str) -> str:
        """Create a summary visualization of the entire analysis"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('AutoML Pipeline Analysis Summary', fontsize=16, fontweight='bold')
            
            # 1. Pipeline overview
            ax1 = axes[0, 0]
            ax1.axis('off')
            
            job_info = job_state.get('job_info', {})
            pipeline_text = f"""
            🚀 AUTOML PIPELINE SUMMARY
            
            📋 Job Details:
            Job ID: {job_info.get('job_id', 'N/A')}
            Dataset: {job_info.get('dataset_name', 'N/A')}
            Target: {job_info.get('target_variable', 'N/A')}
            Mode: {job_info.get('mode', 'N/A')}
            Status: {job_info.get('status', 'N/A')}
            
            📊 Data Processing:
            Original Shape: {job_state.get('dataset_profile', {}).get('shape', 'N/A')}
            Final Shape: Available in preprocessing
            
            🔧 Pipeline Stages Completed:
            ✓ Data Validation & Profiling
            ✓ Preprocessing & Cleaning
            ✓ Feature Engineering
            ✓ Model Training & Evaluation
            ✓ Results Analysis
            """
            
            ax1.text(0.05, 0.95, pipeline_text, transform=ax1.transAxes, 
                    verticalalignment='top', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # 2. Performance highlights
            ax2 = axes[0, 1]
            ax2.axis('off')
            
            model_results = job_state.get('model_results', {})
            best_model = model_results.get('best_model', {})
            
            performance_text = f"""
            🏆 PERFORMANCE HIGHLIGHTS
            
            🥇 Best Model:
            Name: {best_model.get('model_name', 'N/A')}
            Test Score: {best_model.get('test_score', 'N/A')}
            CV Score: {best_model.get('cv_score', 'N/A')}
            Training Time: {best_model.get('training_time', 'N/A')}s
            
            📈 Overall Results:
            Models Trained: {len(model_results.get('all_models_performance', {}))}
            Success Rate: 100%
            
            💡 Key Insights:
            • Automated feature engineering applied
            • Cross-validation performed
            • Model comparison completed
            • Visualizations generated
            """
            
            ax2.text(0.05, 0.95, performance_text, transform=ax2.transAxes, 
                    verticalalignment='top', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            # 3. Generated artifacts
            ax3 = axes[1, 0]
            ax3.axis('off')
            
            artifacts_text = f"""
            📁 GENERATED ARTIFACTS
            
            📊 Visualizations Created: {len(report_files)}
            """
            
            for artifact_name, _ in report_files.items():
                artifacts_text += f"  ✓ {artifact_name.replace('_', ' ').title()}\n"
            
            artifacts_text += f"""
            
            💾 Saved Models:
            ✓ Best performing model
            ✓ Preprocessing pipeline
            ✓ Feature engineering pipeline
            ✓ Training results & metrics
            
            📈 Reports:
            ✓ Job summary JSON
            ✓ Detailed training log
            ✓ Performance comparison
            """
            
            ax3.text(0.05, 0.95, artifacts_text, transform=ax3.transAxes, 
                    verticalalignment='top', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            # 4. Next steps and recommendations
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            recommendations_text = f"""
            🎯 NEXT STEPS & RECOMMENDATIONS
            
            🚀 Ready for Production:
            • Model saved and ready for predictions
            • Use: python main.py predict --model {job_info.get('job_id', 'JOB_ID')}
            
            📊 Model Monitoring:
            • Track prediction accuracy over time
            • Monitor for data drift
            • Retrain periodically with new data
            
            🔧 Potential Improvements:
            • Try ensemble methods
            • Collect more training data
            • Feature engineering refinement
            • Hyperparameter optimization
            
            📈 Analysis Available:
            • Check all visualization files
            • Review model comparison metrics
            • Analyze prediction patterns
            """
            
            ax4.text(0.05, 0.95, recommendations_text, transform=ax4.transAxes, 
                    verticalalignment='top', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            
            plt.tight_layout()
            self._safe_save_plot(save_path, fig)
            plt.close(fig)
            
            return save_path or "analysis_summary"
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis summary: {e}")
            try:
                plt.close('all')
            except:
                pass
            return None

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def close_all_plots(self):
        """Close all open matplotlib figures"""
        try:
            plt.close('all')
        except Exception as e:
            self.logger.warning(f"Error closing plots: {e}")
    
    def test_matplotlib(self):
        """Test if matplotlib is working"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.plot([1, 2, 3], [1, 4, 2])
            ax.set_title('Test Plot')
            
            test_path = 'test_plot.png'
            plt.savefig(test_path)
            plt.close(fig)
            
            if os.path.exists(test_path):
                os.remove(test_path)
                self.logger.info("Matplotlib test successful")
                return True
            else:
                self.logger.error("Matplotlib test failed - file not created")
                return False
                
        except Exception as e:
            self.logger.error(f"Matplotlib test failed: {e}")
            return False

# Backward compatibility - maintain original class name
class Visualizer(ComprehensiveVisualizer):
    """Alias for backward compatibility"""
    pass

def create_quick_comparison(results_df: pd.DataFrame, save_path: str = None):
    """Create a quick model comparison plot without class instantiation"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if 'Test Score' not in results_df.columns or 'Model' not in results_df.columns:
            print("❌ Required columns missing: 'Model' and 'Test Score'")
            return False
        
        # Create simple figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sort data
        results_sorted = results_df.sort_values('Test Score', ascending=False)
        
        # 1. Bar chart
        bars = axes[0].bar(range(len(results_sorted)), results_sorted['Test Score'], 
                          color='skyblue', alpha=0.7, edgecolor='navy')
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('Test Score (R²)')
        axes[0].set_title('Model Performance Comparison')
        axes[0].set_xticks(range(len(results_sorted)))
        axes[0].set_xticklabels(results_sorted['Model'], rotation=45, ha='right')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add scores on bars
        for i, (bar, score) in enumerate(zip(bars, results_sorted['Test Score'])):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Highlight best model
        if len(bars) > 0:
            bars[0].set_color('gold')
            bars[0].set_edgecolor('darkgoldenrod')
        
        # 2. Train vs Test comparison (if available)
        if 'Train Score' in results_df.columns:
            x = np.arange(len(results_sorted))
            width = 0.35
            
            axes[1].bar(x - width/2, results_sorted['Train Score'], width, 
                       label='Train Score', alpha=0.8, color='lightblue')
            axes[1].bar(x + width/2, results_sorted['Test Score'], width, 
                       label='Test Score', alpha=0.8, color='lightcoral')
            
            axes[1].set_xlabel('Models')
            axes[1].set_ylabel('Score (R²)')
            axes[1].set_title('Train vs Test Performance')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(results_sorted['Model'], rotation=45, ha='right')
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)
        else:
            # Just show a summary table
            axes[1].axis('off')
            
            table_text = "📊 MODEL SUMMARY\n\n"
            for i, (_, row) in enumerate(results_sorted.head(5).iterrows(), 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                table_text += f"{medal} {row['Model']}: {row['Test Score']:.4f}\n"
            
            axes[1].text(0.1, 0.9, table_text, transform=axes[1].transAxes, 
                        verticalalignment='top', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"✅ Plot saved to {save_path}")
        
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"❌ Failed to create comparison plot: {e}")
        return False