#!/usr/bin/env python3
"""
generate_report.py
Standalone AutoML Report Generator with Ollama AI explanations
Usage: python3 generate_report.py job_20250819_005202
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import requests
import subprocess
from datetime import datetime
import numpy as np
import sys

class AutoMLReportGenerator:
    def __init__(self, job_id, storage_path="storage/models/jobs"):
        self.job_id = job_id
        self.job_path = Path(storage_path) / job_id
        self.report_data = {}
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Try different models in order of preference
        self.available_models = ["llama3.2:latest", "llama3:latest", "llama2:latest"]
        self.model_name = self.check_ollama_models()
        
        # Load all job data
        self.load_job_data()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def check_ollama_models(self):
        """Check which Ollama models are available"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                available = [model['name'] for model in response.json().get('models', [])]
                print(f"✅ Ollama running with models: {available}")
                
                # Find the first available model from our preference list
                for model in self.available_models:
                    if model in available:
                        print(f"🤖 Using model: {model}")
                        return model
                
                # If none of our preferred models, use the first available
                if available:
                    print(f"🤖 Using first available model: {available[0]}")
                    return available[0]
                else:
                    print("⚠️  No models found in Ollama")
                    return None
            else:
                print("⚠️  Ollama not responding correctly")
                return None
        except:
            print("❌ Ollama not detected. Install with: curl -fsSL https://ollama.ai/install.sh | sh")
            print("   Then run: ollama serve & ollama pull llama3.2:latest")
            return None
    
    def load_job_data(self):
        """Load all relevant data from job files"""
        files_to_load = {
            'state': 'state.json',
            'training_summary': 'training_summary.json', 
            'job_summary': 'job_summary.json',
            'metadata': 'metadata.json'
        }
        
        print(f"🔍 Loading job data from: {self.job_path}")
        
        for key, filename in files_to_load.items():
            filepath = self.job_path / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    self.report_data[key] = json.load(f)
                print(f"✅ Loaded: {filename}")
            else:
                print(f"⚠️  Missing: {filename}")
    
    def query_ollama(self, prompt, context=""):
        """Query Ollama for AI-generated explanations"""
        if not self.model_name:
            return "AI explanations unavailable - Ollama not configured"
            
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 400
            }
        }
        
        try:
            print(f"🤖 Generating AI insight...")
            response = requests.post(self.ollama_url, json=payload, timeout=45)
            if response.status_code == 200:
                return response.json().get('response', 'No response generated')
            else:
                return f"Error: Ollama returned status {response.status_code}"
        except requests.exceptions.RequestException as e:
            return f"Error: Cannot connect to Ollama - {str(e)}"
    
    def create_performance_plot(self):
        """Create model performance comparison plot"""
        all_models = self.report_data.get('training_summary', {}).get('all_models_performance', {})
        
        if not all_models:
            print("⚠️  No model performance data found")
            return None
            
        models = list(all_models.keys())
        test_scores = [all_models[model].get('test_score', 0) for model in models]
        train_scores = [all_models[model].get('train_score', 0) for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Test scores comparison
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        bars1 = ax1.bar(models, test_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('Model Performance Comparison (Test Scores)', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('R² Score', fontsize=13)
        ax1.set_xlabel('Models', fontsize=13)
        ax1.tick_params(axis='x', rotation=45, labelsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(min(0, min(test_scores) - 0.1), max(test_scores) + 0.1)
        
        # Add value labels on bars
        for bar, score in zip(bars1, test_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Train vs Test comparison for top 5 models
        top_models = sorted(all_models.items(), key=lambda x: x[1].get('test_score', 0), reverse=True)[:5]
        top_model_names = [model[0] for model in top_models]
        top_test_scores = [model[1].get('test_score', 0) for model in top_models]
        top_train_scores = [model[1].get('train_score', 0) for model in top_models]
        
        x = np.arange(len(top_model_names))
        width = 0.35
        
        bars2 = ax2.bar(x - width/2, top_test_scores, width, label='Test Score', 
                       color='lightcoral', alpha=0.8, edgecolor='black')
        bars3 = ax2.bar(x + width/2, top_train_scores, width, label='Train Score', 
                       color='lightgreen', alpha=0.8, edgecolor='black')
        
        ax2.set_title('Train vs Test Performance (Top 5 Models)', fontsize=16, fontweight='bold', pad=20)
        ax2.set_ylabel('R² Score', fontsize=13)
        ax2.set_xlabel('Models', fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels(top_model_names, rotation=45, fontsize=10)
        ax2.legend(fontsize=11)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Ensure visualizations directory exists
        viz_dir = self.job_path / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        plot_path = viz_dir / 'performance_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Performance plot saved: {plot_path}")
        return str(plot_path)
    
    def create_feature_analysis_plot(self):
        """Create feature analysis visualization"""
        dataset_profile = self.report_data.get('state', {}).get('dataset_profile', {})
        missing_data = dataset_profile.get('missing_data', {})
        job_summary = self.report_data.get('job_summary', {})
        
        if not missing_data and not job_summary:
            print("⚠️  No feature data found")
            return None
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Missing data analysis
        if missing_data:
            missing_df = pd.DataFrame(list(missing_data.items()), columns=['Feature', 'Missing_Percentage'])
            missing_df = missing_df.sort_values('Missing_Percentage', ascending=False).head(15)
            
            colors = ['red' if x > 50 else 'orange' if x > 20 else 'lightgreen' for x in missing_df['Missing_Percentage']]
            bars = ax1.barh(missing_df['Feature'], missing_df['Missing_Percentage'], 
                           color=colors, alpha=0.8, edgecolor='black')
            ax1.set_title('Missing Data Analysis (Top 15 Features)', fontsize=16, fontweight='bold', pad=20)
            ax1.set_xlabel('Missing Percentage (%)', fontsize=13)
            ax1.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Add percentage labels
            for bar, pct in zip(bars, missing_df['Missing_Percentage']):
                width = bar.get_width()
                ax1.text(width + 1, bar.get_y() + bar.get_height()/2.,
                        f'{pct:.1f}%', ha='left', va='center', fontweight='bold', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No missing data information available', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=14)
            ax1.set_title('Missing Data Analysis', fontsize=16, fontweight='bold')
        
        # Feature engineering summary
        original_features = job_summary.get('original_features', 0)
        final_features = job_summary.get('final_features', 0)
        features_removed = job_summary.get('features_removed', 0)
        
        if original_features > 0:
            categories = ['Original\nFeatures', 'Features\nRemoved', 'Final\nFeatures']
            values = [original_features, features_removed, final_features]
            colors = ['lightblue', 'lightcoral', 'lightgreen']
            
            bars2 = ax2.bar(categories, values, color=colors, alpha=0.8, 
                           edgecolor='black', linewidth=2)
            ax2.set_title('Feature Engineering Summary', fontsize=16, fontweight='bold', pad=20)
            ax2.set_ylabel('Number of Features', fontsize=13)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels
            for bar, value in zip(bars2, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                        str(value), ha='center', va='bottom', fontweight='bold', fontsize=14)
        else:
            ax2.text(0.5, 0.5, 'No feature engineering data available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Feature Engineering Summary', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        viz_dir = self.job_path / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        plot_path = viz_dir / 'feature_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Feature plot saved: {plot_path}")
        return str(plot_path)
    
    def generate_markdown_report(self):
        """Generate comprehensive markdown report"""
        job_info = self.report_data.get('state', {}).get('job_info', {})
        job_summary = self.report_data.get('job_summary', {})
        training_summary = self.report_data.get('training_summary', {})
        best_model = training_summary.get('best_model', {})
        dataset_profile = self.report_data.get('state', {}).get('dataset_profile', {})
        
        print("📊 Creating visualizations...")
        
        # Generate plots
        perf_plot = self.create_performance_plot()
        feature_plot = self.create_feature_analysis_plot()
        
        # Get AI explanations if available
        print("🤖 Generating AI explanations...")
        
        model_context = f"""
        AutoML Analysis Context:
        - Dataset: {job_info.get('dataset_name', 'N/A')}
        - Target: {job_info.get('target_variable', 'N/A')}
        - Best Model: {best_model.get('model_name', 'N/A')} with R² = {best_model.get('test_score', 0):.4f}
        - Models Trained: {job_summary.get('models_trained', 0)}
        - Original Features: {job_summary.get('original_features', 0)}
        - Final Features: {job_summary.get('final_features', 0)}
        - Outliers Removed: {job_summary.get('outliers_removed', 0)}
        """
        
        if self.model_name:
            executive_summary = self.query_ollama(
                "Write a professional executive summary (2-3 paragraphs) for this AutoML analysis. Focus on key findings, model performance, and business value. Keep it concise and actionable.",
                model_context
            )
            
            model_interpretation = self.query_ollama(
                f"Explain why {best_model.get('model_name', 'the best model')} performed well for this regression task with R² = {best_model.get('test_score', 0):.4f}. Discuss its strengths, why it was selected, and what this performance means.",
                model_context
            )
            
            results_analysis = self.query_ollama(
                f"Analyze the R² score of {best_model.get('test_score', 0):.4f} in practical terms. What does this mean for prediction accuracy and business applications? Is this good/fair/excellent performance?",
                model_context
            )
        else:
            executive_summary = f"""This AutoML analysis successfully trained {job_summary.get('models_trained', 0)} machine learning models on the {job_info.get('dataset_name', '')} dataset to predict {job_info.get('target_variable', 'the target variable')}. The best performing model was {best_model.get('model_name', 'unknown')} with an R² score of {best_model.get('test_score', 0):.4f}, indicating {'excellent' if best_model.get('test_score', 0) > 0.8 else 'good' if best_model.get('test_score', 0) > 0.6 else 'moderate'} predictive performance.

The automated pipeline processed {job_summary.get('original_features', 0)} original features, intelligently selecting {job_summary.get('final_features', 0)} optimized features and removing {job_summary.get('outliers_removed', 0)} outlier samples. This optimization improved model efficiency while maintaining predictive accuracy."""
            
            model_interpretation = f"""{best_model.get('model_name', 'The selected model')} was chosen as the optimal solution based on its superior test performance (R² = {best_model.get('test_score', 0):.4f}) and cross-validation stability (CV R² = {best_model.get('cv_score', 0):.4f}). The model demonstrates good generalization with minimal overfitting risk, making it suitable for production deployment."""
            
            results_analysis = f"""The R² score of {best_model.get('test_score', 0):.4f} indicates that the model explains {best_model.get('test_score', 0)*100:.1f}% of the variance in the target variable. This represents {'excellent' if best_model.get('test_score', 0) > 0.8 else 'good' if best_model.get('test_score', 0) > 0.6 else 'moderate' if best_model.get('test_score', 0) > 0.4 else 'limited'} predictive capability and is {'highly suitable' if best_model.get('test_score', 0) > 0.7 else 'suitable' if best_model.get('test_score', 0) > 0.5 else 'potentially suitable'} for practical applications."""
        
        # Generate comprehensive markdown
        markdown_content = f"""
# AutoML Analysis Report
## Comprehensive Machine Learning Pipeline Results

---

**Job ID:** {self.job_id}  
**Dataset:** {job_info.get('dataset_name', 'N/A')}  
**Target Variable:** {job_info.get('target_variable', 'N/A')}  
**Analysis Date:** {datetime.now().strftime('%B %d, %Y')}  
**Processing Mode:** {job_info.get('mode', 'N/A').title()}  

---

## Executive Summary

{executive_summary}

### Key Performance Metrics

| Metric | Value |
|--------|--------|
| **Best Model** | {best_model.get('model_name', 'N/A')} |
| **Test Score (R²)** | {best_model.get('test_score', 0):.4f} |
| **Cross-Validation Score** | {best_model.get('cv_score', 0):.4f} |
| **Models Evaluated** | {job_summary.get('models_trained', 0)} |
| **Total Training Time** | {job_summary.get('total_training_time', 0):.2f} seconds |
| **Features Used** | {job_summary.get('final_features', 0)} (from {job_summary.get('original_features', 0)} original) |
| **Outliers Removed** | {job_summary.get('outliers_removed', 0)} samples |

---

## 1. Introduction and Objectives

This report presents a comprehensive analysis conducted using an automated machine learning (AutoML) pipeline designed for regression tasks. The analysis was performed on the **{job_info.get('dataset_name', 'N/A')}** dataset with the primary objective of predicting **{job_info.get('target_variable', 'N/A')}**.

### Primary Objectives
- Develop an accurate predictive model for {job_info.get('target_variable', 'N/A')}
- Identify the most important features influencing the target variable
- Compare performance across multiple machine learning algorithms
- Provide actionable insights for decision-making

### AutoML Pipeline Overview
The automated pipeline consists of six main stages:
1. **Data Validation & Profiling** - Comprehensive data quality assessment
2. **Data Preprocessing** - Automated cleaning and preparation
3. **Feature Engineering** - Intelligent feature creation and selection
4. **Model Selection** - Algorithm recommendation based on data characteristics
5. **Model Training** - Automated hyperparameter optimization
6. **Evaluation & Reporting** - Performance assessment and result interpretation

---

## 2. Dataset Overview and Analysis

### 2.1 Dataset Characteristics

| Attribute | Value |
|-----------|--------|
| **Number of Samples** | {dataset_profile.get('num_rows', 'N/A'):,} |
| **Number of Features** | {dataset_profile.get('num_cols', 'N/A')} |
| **File Size** | {dataset_profile.get('file_size_mb', 'N/A')} MB |
| **Memory Usage** | {dataset_profile.get('memory_usage_mb', 'N/A')} MB |
| **Data Types** | All Numeric (Phase 1 Pipeline) |

### 2.2 Target Variable Analysis

**Target Variable:** {job_info.get('target_variable', 'N/A')}

{dataset_profile.get('numeric_summaries', {}).get(job_info.get('target_variable', ''), {}) and f"""
| Statistic | Value |
|-----------|--------|
| **Mean** | {dataset_profile.get('numeric_summaries', {}).get(job_info.get('target_variable', ''), {}).get('mean', 'N/A'):.2f} |
| **Standard Deviation** | {dataset_profile.get('numeric_summaries', {}).get(job_info.get('target_variable', ''), {}).get('std', 'N/A'):.2f} |
| **Minimum** | {dataset_profile.get('numeric_summaries', {}).get(job_info.get('target_variable', ''), {}).get('min', 'N/A'):.2f} |
| **Maximum** | {dataset_profile.get('numeric_summaries', {}).get(job_info.get('target_variable', ''), {}).get('max', 'N/A'):.2f} |
| **Skewness** | {dataset_profile.get('numeric_summaries', {}).get(job_info.get('target_variable', ''), {}).get('skewness', 'N/A'):.3f} |
| **Unique Values** | {dataset_profile.get('numeric_summaries', {}).get(job_info.get('target_variable', ''), {}).get('unique_count', 'N/A'):,} |
""" or "Target variable statistics not available"}

### 2.3 Data Quality Assessment

{len([k for k, v in dataset_profile.get('missing_data', {}).items() if v > 0]) if dataset_profile.get('missing_data') else 0} columns contain missing values, with {len([k for k, v in dataset_profile.get('missing_data', {}).items() if v > 50]) if dataset_profile.get('missing_data') else 0} columns having >50% missing data. The pipeline automatically handled data quality issues including:

- **Missing Value Treatment:** Intelligent imputation strategies
- **Outlier Detection:** Multiple methods (IQR, Standard Deviation, Percentile)  
- **Feature Correlation:** Removal of highly correlated features (>0.9)
- **Constant Features:** {len(dataset_profile.get('constant_columns', []))} constant columns removed

{feature_plot and f"![Feature Analysis](visualizations/feature_analysis.png)" or ""}

---

## 3. Methodology and Data Preprocessing

### 3.1 Preprocessing Pipeline Results

| Process | Result |
|---------|--------|
| **Outliers Removed** | {job_summary.get('outliers_removed', 0)} samples |
| **Features Removed** | {job_summary.get('features_removed', 0)} (high correlation/low variance) |
| **Feature Engineering** | {job_summary.get('original_features', 0)} → {job_summary.get('final_features', 0)} features |
| **Data Transformations** | Scaling, normalization, feature selection applied |

### 3.2 Feature Engineering Process

The automated feature engineering pipeline:
- **Polynomial Features:** Created higher-order terms and interactions
- **Statistical Features:** Generated ratios and differences between features  
- **Feature Selection:** Intelligently selected top {job_summary.get('final_features', 0)} features using automated techniques
- **Dimensionality Reduction:** Reduced feature space from {job_summary.get('original_features', 0)} to {job_summary.get('final_features', 0)} features ({((job_summary.get('original_features', 1) - job_summary.get('final_features', 0)) / job_summary.get('original_features', 1) * 100):.1f}% reduction)

---

## 4. Model Selection and Training Results

### 4.1 Algorithm Evaluation

The AutoML system evaluated {job_summary.get('models_trained', 0)} different algorithms:

{chr(10).join([f"- **{model}:** R² = {metrics.get('test_score', 0):.4f} (Training Time: {metrics.get('training_time', 0):.3f}s)" for model, metrics in training_summary.get('all_models_performance', {}).items()])}

{perf_plot and f"![Model Performance Comparison](visualizations/performance_comparison.png)" or ""}

### 4.2 Detailed Performance Comparison

| Model | Test R² | Train R² | CV R² | Training Time (s) | Overfitting Risk |
|-------|---------|----------|--------|-------------------|------------------|
{chr(10).join([f"| **{model}** | {metrics.get('test_score', 0):.4f} | {metrics.get('train_score', 0):.4f} | {metrics.get('cv_score', 0):.4f} | {metrics.get('training_time', 0):.3f} | {'**High**' if metrics.get('train_score', 0) - metrics.get('test_score', 0) > 0.2 else '**Medium**' if metrics.get('train_score', 0) - metrics.get('test_score', 0) > 0.1 else 'Low'} |" for model, metrics in training_summary.get('all_models_performance', {}).items()])}

### 4.3 Best Model Selection

{model_interpretation}

**Key Selection Criteria:**
- **Highest Test Performance:** R² = {best_model.get('test_score', 0):.4f}
- **Cross-Validation Stability:** CV R² = {best_model.get('cv_score', 0):.4f}
- **Training Efficiency:** {best_model.get('training_time', 0):.3f} seconds
- **Generalization:** {'Excellent' if abs(best_model.get('train_score', 0) - best_model.get('test_score', 0)) < 0.05 else 'Good' if abs(best_model.get('train_score', 0) - best_model.get('test_score', 0)) < 0.1 else 'Moderate'} (overfitting risk: {'Low' if abs(best_model.get('train_score', 0) - best_model.get('test_score', 0)) < 0.1 else 'Medium' if abs(best_model.get('train_score', 0) - best_model.get('test_score', 0)) < 0.2 else 'High'})

---

## 5. Results Analysis and Interpretation

### 5.1 Performance Interpretation

{results_analysis}

### 5.2 Model Performance Metrics Explained

- **Test R² ({best_model.get('test_score', 0):.4f}):** The most critical metric - performance on completely unseen data
- **Cross-Validation R² ({best_model.get('cv_score', 0):.4f}):** Indicates model stability across different data subsets
- **Training Time ({best_model.get('training_time', 0):.3f}s):** Demonstrates computational efficiency
- **Overfitting Assessment:** {abs(best_model.get('train_score', 0) - best_model.get('test_score', 0)):.3f} difference between train/test scores

### 5.3 Business Impact Assessment

With an R² score of **{best_model.get('test_score', 0):.4f}**, this model:

- Explains **{best_model.get('test_score', 0)*100:.1f}%** of variance in {job_info.get('target_variable', 'the target')}
- Provides **{'High' if best_model.get('test_score', 0) > 0.8 else 'Good' if best_model.get('test_score', 0) > 0.6 else 'Moderate' if best_model.get('test_score', 0) > 0.4 else 'Limited'}** predictive capability
- Is **{'Highly Recommended' if best_model.get('test_score', 0) > 0.7 else 'Recommended' if best_model.get('test_score', 0) > 0.5 else 'Conditionally Recommended'}** for production deployment

---

## 6. Recommendations and Next Steps

### 6.1 Immediate Actions

1. **✅ Model Deployment:** The {best_model.get('model_name', 'selected model')} is ready for production use
2. **📊 Monitoring Setup:** Implement performance tracking on new data
3. **🔄 Retraining Schedule:** Plan quarterly model updates or when performance degrades >5%
4. **📋 Documentation:** Maintain feature definitions and preprocessing requirements

### 6.2 Model Optimization Opportunities

1. **Feature Enhancement:** 
   - Investigate domain-specific feature engineering
   - Consider external data sources for additional predictive power
   
2. **Advanced Techniques:**
   - Explore ensemble methods combining multiple algorithms
   - Implement SHAP or LIME for enhanced interpretability
   
3. **Data Quality Improvements:**
   - Collect additional training samples ({dataset_profile.get('num_rows', 0)} current samples)
   - Address high missing data in key features

---

## 7. Limitations and Considerations

### 7.1 Model Limitations

- **Sample Size:** Model trained on {dataset_profile.get('num_rows', 'N/A')} samples - performance may vary on different populations
- **Feature Dependencies:** Requires all {job_summary.get('final_features', 0)} selected features for optimal performance  
- **Temporal Stability:** Assumes feature relationships remain stable over time
- **Domain Coverage:** Performance guaranteed only within the range of training data

### 7.2 Technical Considerations

- **Preprocessing Pipeline:** New data must follow identical preprocessing steps
- **Feature Engineering:** Same feature creation logic must be applied consistently
- **Model Complexity:** {best_model.get('model_name', 'Selected model')} requires {'minimal' if best_model.get('model_name', '') in ['LinearRegression', 'Ridge', 'Lasso'] else 'moderate' if best_model.get('model_name', '') in ['RandomForest', 'GradientBoosting'] else 'standard'} computational resources

---

## 8. Conclusion and Future Work

### 8.1 Executive Summary

This AutoML analysis successfully identified **{best_model.get('model_name', 'a high-performing model')}** as the optimal solution for predicting {job_info.get('target_variable', 'the target variable')}. The model achieves an R² score of **{best_model.get('test_score', 0):.4f}**, indicating {'excellent' if best_model.get('test_score', 0) > 0.8 else 'good' if best_model.get('test_score', 0) > 0.6 else 'moderate' if best_model.get('test_score', 0) > 0.4 else 'limited'} predictive capability suitable for {'immediate production deployment' if best_model.get('test_score', 0) > 0.7 else 'production use with monitoring' if best_model.get('test_score', 0) > 0.5 else 'pilot testing and refinement'}.

The automated pipeline efficiently processed {job_summary.get('original_features', 0)} features down to {job_summary.get('final_features', 0)} optimized features, demonstrating effective dimensionality reduction while maintaining predictive power. Training completed in just {job_summary.get('total_training_time', 0):.1f} seconds, showcasing the efficiency of the automated approach.

### 8.2 Future Enhancement Opportunities

1. **Advanced Modeling:**
   - Implement ensemble methods (stacking, voting classifiers)
   - Explore deep learning approaches for complex feature interactions
   - Consider time-series modeling if temporal patterns exist

2. **Feature Engineering:**
   - Domain-specific feature creation based on business knowledge
   - External data integration (economic indicators, seasonal factors)
   - Advanced feature selection techniques (recursive feature elimination)

3. **Model Interpretability:**
   - SHAP (SHapley Additive exPlanations) analysis for feature importance
   - LIME (Local Interpretable Model-agnostic Explanations) for individual predictions
   - Partial dependence plots for feature relationship understanding

4. **Production Enhancements:**
   - A/B testing framework for model comparison
   - Real-time monitoring and alerting systems
   - Automated retraining pipelines with data drift detection

---

## 9. Technical Appendix

### 9.1 Reproducibility Information

**Environment Configuration:**
- Pipeline Version: Phase 1 - All-Numeric Regression
- Cross-Validation: {self.report_data.get('metadata', {}).get('config_used', {}).get('modeling', {}).get('evaluation', {}).get('cv_folds', 5)}-fold
- Test Split: {(self.report_data.get('metadata', {}).get('config_used', {}).get('modeling', {}).get('evaluation', {}).get('test_size', 0.2)*100):.0f}% holdout
- Random State: {self.report_data.get('metadata', {}).get('config_used', {}).get('modeling', {}).get('evaluation', {}).get('random_state', 42)}
- Hyperparameter Tuning: {self.report_data.get('metadata', {}).get('config_used', {}).get('modeling', {}).get('hyperparameter_tuning', {}).get('method', 'RandomSearch').title()}

**Reproduction Command:**
```bash
python main.py run --file {job_info.get('dataset_name', 'dataset.csv')} --target {job_info.get('target_variable', 'target')} --mode {job_info.get('mode', 'auto')}
```

### 9.2 Model Artifacts Location

All trained models and preprocessing components are saved in:
```
{self.job_path}/
├── all_models/                 # All trained model files
├── preprocessors/
│   └── pipeline.pkl           # Data preprocessing pipeline
├── feature_engineering_pipeline.pkl  # Feature engineering pipeline
├── visualizations/            # Generated plots and charts
├── metadata.json             # Configuration and settings
├── training_summary.json     # Detailed model results
└── job_summary.json          # High-level summary
```

### 9.3 Model Deployment Checklist

**Pre-deployment Requirements:**
- [ ] Validate model performance on holdout test set
- [ ] Confirm preprocessing pipeline compatibility
- [ ] Test feature engineering pipeline on new data
- [ ] Verify computational resource requirements
- [ ] Establish monitoring and alerting thresholds

**Production Deployment:**
- [ ] Implement model serving infrastructure
- [ ] Set up prediction logging and monitoring
- [ ] Configure automated retraining triggers
- [ ] Document model API and usage guidelines
- [ ] Establish rollback procedures

### 9.4 Performance Benchmarks

**Baseline Comparisons:**
- **Naive Baseline:** Mean prediction R² ≈ 0.000
- **Simple Linear Model:** R² ≈ {training_summary.get('all_models_performance', {}).get('LinearRegression', {}).get('test_score', 'N/A')}
- **Best Model ({best_model.get('model_name', 'N/A')}):** R² = {best_model.get('test_score', 0):.4f}
- **Improvement over baseline:** {(best_model.get('test_score', 0) * 100):.1f}% variance explained

---

## 10. Glossary of Terms

**R² Score (Coefficient of Determination):** Proportion of variance in the target variable explained by the model. Ranges from 0 to 1, with 1 being perfect prediction.

**Cross-Validation (CV):** Technique to assess model performance by training on multiple data subsets and averaging results.

**Overfitting:** When a model performs well on training data but poorly on new, unseen data.

**Feature Engineering:** Process of creating new features or transforming existing ones to improve model performance.

**Hyperparameter Tuning:** Systematic optimization of model configuration parameters to achieve best performance.

**Training/Test Split:** Division of data into separate sets for model training and unbiased performance evaluation.

---

*Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} using AutoML Phase 1 Pipeline*  
*Analysis completed in {job_summary.get('total_training_time', 0):.1f} seconds with {job_summary.get('models_trained', 0)} models evaluated*  
*Best model: {best_model.get('model_name', 'N/A')} (R² = {best_model.get('test_score', 0):.4f})*
"""
        
        return markdown_content
    
    def convert_markdown_to_pdf(self, markdown_content, output_path):
        """Convert markdown to PDF using pandoc"""
        temp_md_path = self.job_path / "temp_report.md"
        
        # Write markdown to temporary file
        with open(temp_md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        try:
            # Use pandoc to convert to PDF with professional formatting
            cmd = [
                'pandoc',
                str(temp_md_path),
                '-o', str(output_path),
                '--pdf-engine=xelatex',
                '-V', 'geometry:margin=0.8in',
                '-V', 'fontsize=11pt',
                '-V', 'documentclass=article',
                '-V', 'colorlinks=true',
                '-V', 'linkcolor=blue',
                '-V', 'urlcolor=blue',
                '--toc',
                '--toc-depth=3',
                '--number-sections',
                '--highlight-style=github',
                '--standalone'
            ]
            
            print("📋 Converting to PDF...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ PDF generated successfully!")
                # Clean up temporary file
                temp_md_path.unlink()
                return True
            else:
                print(f"❌ Pandoc error: {result.stderr}")
                print(f"📄 Markdown file saved as fallback: {temp_md_path}")
                return False
                
        except FileNotFoundError:
            print("❌ Pandoc not found. Install instructions:")
            print("   Ubuntu/Debian: sudo apt-get install pandoc texlive-xetex")
            print("   macOS: brew install pandoc basictex")
            print("   Windows: Download from https://pandoc.org/installing.html")
            print(f"📄 Markdown file saved: {temp_md_path}")
            return False
    
    def generate_report(self):
        """Generate the complete report"""
        print("🚀 AutoML Report Generator")
        print("=" * 50)
        print(f"📊 Generating report for job: {self.job_id}")
        
        if not any(self.report_data.values()):
            print("❌ No job data found. Please check:")
            print(f"   - Job ID: {self.job_id}")
            print(f"   - Path: {self.job_path}")
            print(f"   - Required files: state.json, training_summary.json, job_summary.json")
            return None
        
        print("📝 Generating comprehensive markdown content...")
        markdown_content = self.generate_markdown_report()
        
        # Save markdown for review
        md_output_path = self.job_path / f"AutoML_Report_{self.job_id}.md"
        with open(md_output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"📄 Markdown saved: {md_output_path}")
        
        # Convert to PDF
        pdf_output_path = self.job_path / f"AutoML_Report_{self.job_id}.pdf"
        
        if self.convert_markdown_to_pdf(markdown_content, pdf_output_path):
            print(f"\n🎉 SUCCESS! Complete report generated:")
            print(f"📁 PDF Report: {pdf_output_path}")
            print(f"📁 Markdown: {md_output_path}")
            return pdf_output_path
        else:
            print(f"\n⚠️  PDF conversion failed, but markdown is available:")
            print(f"📁 Markdown Report: {md_output_path}")
            return md_output_path

def main():
    """Main function for standalone execution"""
    if len(sys.argv) != 2:
        print("🚀 AutoML Report Generator")
        print("=" * 40)
        print("Usage: python3 generate_report.py <job_id>")
        print("Example: python3 generate_report.py job_20250819_005202")
        print("\nThis will generate a comprehensive PDF report with:")
        print("  • Executive summary with AI insights")
        print("  • Detailed model performance analysis") 
        print("  • Data quality assessment")
        print("  • Feature engineering results")
        print("  • Business recommendations")
        print("  • Technical appendix")
        sys.exit(1)
    
    job_id = sys.argv[1]
    
    # Check if job directory exists
    job_path = Path("storage/models/jobs") / job_id
    if not job_path.exists():
        print(f"❌ Job directory not found: {job_path}")
        print("Available jobs:")
        jobs_dir = Path("storage/models/jobs")
        if jobs_dir.exists():
            for job_dir in jobs_dir.iterdir():
                if job_dir.is_dir():
                    print(f"   • {job_dir.name}")
        sys.exit(1)
    
    # Generate report
    generator = AutoMLReportGenerator(job_id)
    report_path = generator.generate_report()
    
    if report_path:
        print(f"\n✨ Report generation completed successfully!")
        print(f"📂 Open: {report_path}")
        
        # Show file size
        if Path(report_path).exists():
            size_mb = Path(report_path).stat().st_size / (1024 * 1024)
            print(f"📊 File size: {size_mb:.2f} MB")
    else:
        print("\n❌ Report generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()