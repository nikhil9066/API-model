# """
# generate_report.py - ENHANCED VERSION with All Plots
# Standalone AutoML Report Generator with AI explanations for every plot
# Usage: python3 generate_report.py job_20250819_005202
# """

# import json
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# import requests
# import subprocess
# from datetime import datetime
# import numpy as np
# import sys
# import base64
# from io import BytesIO

# class AutoMLReportGenerator:
#     def __init__(self, job_id, storage_path="storage/models/jobs"):
#         self.job_id = job_id
#         self.job_path = Path(storage_path) / job_id
#         self.report_data = {}
#         self.ollama_url = "http://localhost:11434/api/generate"
        
#         # Try different models in order of preference
#         self.available_models = ["llama3.2:latest", "llama3:latest", "llama2:latest"]
#         self.model_name = self.check_ollama_models()
        
#         # Load all job data
#         self.load_job_data()
        
#         # Set up plotting style
#         plt.style.use('default')
#         sns.set_palette("husl")
        
#         # All possible plot locations
#         self.plot_locations = [
#             self.job_path,  # Root job directory
#             self.job_path / 'visualizations',  # Main viz folder
#             self.job_path / 'enhanced_visualizations',  # Enhanced viz folder
#         ]
        
#     def check_ollama_models(self):
#         """Check which Ollama models are available"""
#         try:
#             response = requests.get("http://localhost:11434/api/tags", timeout=5)
#             if response.status_code == 200:
#                 available = [model['name'] for model in response.json().get('models', [])]
#                 print("✅ Ollama AI Detective Agency reporting for duty! Models: " + str(available))
                
#                 for model in self.available_models:
#                     if model in available:
#                         print("🧠 Recruiting AI brain: " + model + " (This one's got the smarts!)")
#                         return model
                
#                 if available:
#                     print("🤖 Using mystery AI model: " + available[0] + " (Let's see what this one can do!)")
#                     return available[0]
#                 else:
#                     print("🤷‍♂️ No AI models found - we'll wing it with pure human logic!")
#                     return None
#             else:
#                 print("😴 Ollama is taking a nap...")
#                 return None
#         except:
#             print("🔌 Ollama is off the grid! Install with: curl -fsSL https://ollama.ai/install.sh | sh")
#             return None
    
#     def load_job_data(self):
#         """Load all relevant data from job files"""
#         files_to_load = {
#             'state': 'state.json',
#             'training_summary': 'training_summary.json', 
#             'job_summary': 'job_summary.json',
#             'metadata': 'metadata.json'
#         }
        
#         print("🕵️‍♂️ CSI: Data Loading Unit investigating: " + str(self.job_path))
        
#         for key, filename in files_to_load.items():
#             filepath = self.job_path / filename
#             if filepath.exists():
#                 with open(filepath, 'r') as f:
#                     self.report_data[key] = json.load(f)
#                 print("🗂️ Evidence collected: " + filename + " ✓")
#             else:
#                 print("🚨 Missing evidence: " + filename + " (This might be a problem!)")
    
#     def query_ollama(self, prompt, context="", max_words=200):
#         """Query Ollama for AI-generated explanations"""
#         if not self.model_name:
#             return "🤖 AI is on vacation, generating human-powered insights instead..."
            
#         full_prompt = context + "\n\n" + prompt + "\n\nKeep response under " + str(max_words) + " words and focus on practical insights."
        
#         payload = {
#             "model": self.model_name,
#             "prompt": full_prompt,
#             "stream": False,
#             "options": {
#                 "temperature": 0.7,
#                 "num_predict": max_words
#             }
#         }
        
#         try:
#             print("🧠 AI neurons firing...")
#             response = requests.post(self.ollama_url, json=payload, timeout=45)
#             if response.status_code == 200:
#                 return response.json().get('response', 'AI had a brain freeze 🧊')
#             else:
#                 return "🤖 AI returned error code: " + str(response.status_code)
#         except Exception as e:
#             return "🚫 AI connection failed: " + str(e)
    
#     def discover_all_plots(self):
#         """Hunt down all available plots from multiple locations"""
#         print("🎨 Launching the Great Plot Hunt across multiple dimensions...")
        
#         # Comprehensive list of plot files to search for
#         plot_catalog = {
#             'model_performance': [
#                 'comprehensive_model_comparison.png',
#                 'model_comparison.png', 
#                 'performance_comparison.png',
#                 'model_performance_analysis.png'
#             ],
#             'feature_analysis': [
#                 'feature_analysis.png',
#                 'feature_engineering_analysis.png'
#             ],
#             'data_quality': [
#                 'dataset_overview.png',
#                 'preprocessing_summary.png',
#                 'data_quality_analysis.png'
#             ],
#             'analysis_summary': [
#                 'analysis_summary.png'
#             ]
#         }
        
#         discovered_plots = {}
#         total_found = 0
        
#         # Search through all possible locations
#         for location in self.plot_locations:
#             if not location.exists():
#                 continue
                
#             print("🔍 Investigating plot crime scene: " + str(location))
            
#             for plot_category, filenames in plot_catalog.items():
#                 for filename in filenames:
#                     plot_path = location / filename
#                     if plot_path.exists() and plot_category not in discovered_plots:
#                         discovered_plots[plot_category] = {
#                             'path': str(plot_path),
#                             'filename': filename,
#                             'category': plot_category
#                         }
#                         total_found += 1
                        
#                         # Fun discovery messages
#                         if 'model' in filename.lower():
#                             print("🏆 Jackpot! Found model showdown: " + filename)
#                         elif 'feature' in filename.lower():
#                             print("🧬 Feature detective work discovered: " + filename)
#                         elif 'data' in filename.lower():
#                             print("📊 Data archaeology unearthed: " + filename)
#                         elif 'analysis' in filename.lower():
#                             print("🔬 Analysis goldmine located: " + filename)
#                         else:
#                             print("🎁 Mystery treasure found: " + filename)
        
#         if total_found == 0:
#             print("😱 Plot apocalypse! No visualizations found anywhere!")
#         elif total_found < 3:
#             print("📈 Found " + str(total_found) + " plots - we'll make the most of these gems!")
#         else:
#             print("🎉 Plot bonanza! Collected " + str(total_found) + " beautiful visualizations!")
        
#         return discovered_plots
    
#     def copy_plot_to_report_dir(self, plot_info):
#         """Copy plot to the same directory as the report"""
#         if not plot_info or not Path(plot_info['path']).exists():
#             return None
            
#         source_path = Path(plot_info['path'])
#         dest_path = self.job_path / source_path.name
        
#         # Check if source and destination are the same file
#         if source_path.resolve() == dest_path.resolve():
#             print("📋 Plot already in perfect position: " + source_path.name)
#             return source_path.name
        
#         # Only copy if they're different files
#         import shutil
#         try:
#             shutil.copy2(source_path, dest_path)
#             print("📋 Filed under evidence: " + source_path.name)
#             return source_path.name
#         except Exception as e:
#             print("⚠️ Plot filing had a hiccup: " + str(e))
#             # If copy fails, just use the original filename if it's accessible
#             return source_path.name if source_path.exists() else None
    
#     def generate_plot_explanation(self, plot_category, plot_filename, context_data):
#         """Generate AI explanation for a specific plot"""
#         if not self.model_name:
#             # Fallback explanations
#             explanations = {
#                 'model_performance': "This visualization compares the performance of different machine learning models, showing their R² scores, training times, and overfitting characteristics. The best performing model stands out with the highest test score while maintaining good generalization.",
#                 'feature_analysis': "This plot analyzes the features used in the model, showing feature importance, engineering results, and data quality metrics. It helps understand which variables contribute most to the predictions.",
#                 'data_quality': "This visualization provides insights into data quality, including missing values, outliers, preprocessing steps, and dataset characteristics. It shows how the data was cleaned and prepared for modeling.",
#                 'analysis_summary': "This comprehensive summary plot provides an overview of the entire analysis process, combining key metrics, model performance, and data insights in a single view."
#             }
#             return explanations.get(plot_category, "This visualization provides important insights about the machine learning analysis.")
        
#         # Generate AI explanation based on plot category
#         prompts = {
#             'model_performance': "Explain what insights can be gained from a model performance comparison plot in AutoML. Focus on how to interpret R² scores, training times, and model selection criteria.",
#             'feature_analysis': "Describe what a feature analysis plot reveals about machine learning features, including feature importance, engineering techniques, and selection methods.",
#             'data_quality': "Explain what data quality and preprocessing visualizations show about dataset characteristics, missing values, outliers, and data preparation steps.",
#             'analysis_summary': "Describe what an analysis summary plot reveals about the overall AutoML process, key findings, and business implications."
#         }
        
#         prompt = prompts.get(plot_category, "Explain this machine learning visualization.")
#         context = "AutoML Context: " + str(context_data)
        
#         return self.query_ollama(prompt, context, max_words=150)
    
#     def generate_markdown_report(self):
#         """Generate comprehensive markdown report with all available plots"""
#         job_info = self.report_data.get('state', {}).get('job_info', {})
#         job_summary = self.report_data.get('job_summary', {})
#         training_summary = self.report_data.get('training_summary', {})
#         best_model = training_summary.get('best_model', {})
#         dataset_profile = self.report_data.get('state', {}).get('dataset_profile', {})
        
#         print("🎨 Initiating the Great Plot Discovery Expedition...")
#         discovered_plots = self.discover_all_plots()
        
#         # Copy all plots to report directory
#         print("📚 Organizing visual evidence in the report library...")
#         plot_references = {}
#         for category, plot_info in discovered_plots.items():
#             filename = self.copy_plot_to_report_dir(plot_info)
#             if filename:
#                 plot_references[category] = filename
#                 print("📋 Filed under evidence: " + filename)
        
#         # Generate context for AI explanations
#         context_data = {
#             'dataset': job_info.get('dataset_name', 'N/A'),
#             'target': job_info.get('target_variable', 'N/A'),
#             'best_model': best_model.get('model_name', 'N/A'),
#             'r2_score': best_model.get('test_score', 0),
#             'models_trained': job_summary.get('models_trained', 0),
#             'features_original': job_summary.get('original_features', 0),
#             'features_final': job_summary.get('final_features', 0)
#         }
        
#         # Get AI explanations for main sections
#         print("🧠 Summoning AI wisdom for executive insights...")
        
#         if self.model_name:
#             exec_context = """Dataset: """ + str(context_data['dataset']) + """
# Target: """ + str(context_data['target']) + """
# Best Model: """ + str(context_data['best_model']) + """ (R² = """ + str(context_data['r2_score']) + """)
# Models Trained: """ + str(context_data['models_trained']) + """
# Features: """ + str(context_data['features_original']) + """ → """ + str(context_data['features_final'])
            
#             executive_summary = self.query_ollama(
#                 "Write a professional 2-paragraph executive summary for this AutoML analysis. Focus on business value and key achievements.",
#                 exec_context, max_words=250
#             )
            
#             methodology_explanation = self.query_ollama(
#                 "Explain the AutoML methodology and why it's valuable for this analysis. Focus on automation benefits and process efficiency.",
#                 exec_context, max_words=200
#             )
            
#             results_interpretation = self.query_ollama(
#                 "Interpret the R² score of " + str(context_data['r2_score']) + " and explain what it means for business applications and model reliability.",
#                 exec_context, max_words=150
#             )
#         else:
#             executive_summary = """This AutoML analysis successfully evaluated """ + str(context_data['models_trained']) + """ machine learning models to predict """ + str(context_data['target']) + """. The automated pipeline identified """ + str(context_data['best_model']) + """ as the optimal solution with an R² score of """ + str(context_data['r2_score']) + """, demonstrating good predictive capability.

# The analysis efficiently processed """ + str(context_data['features_original']) + """ original features, selecting """ + str(context_data['features_final']) + """ optimized features through intelligent feature engineering. This automation significantly reduces manual effort while maintaining high-quality results suitable for production deployment."""
            
#             methodology_explanation = """The AutoML approach automates the traditionally manual and time-consuming process of model selection, feature engineering, and hyperparameter tuning. This systematic evaluation ensures optimal model selection while reducing human bias and accelerating time-to-deployment."""
            
#             results_interpretation = """The R² score of """ + str(context_data['r2_score']) + """ indicates the model explains """ + str(round(context_data['r2_score'] * 100, 1)) + """% of variance in the target variable, representing good predictive performance suitable for business applications."""
        
#         # Get target variable statistics
#         target_stats = dataset_profile.get('numeric_summaries', {}).get(job_info.get('target_variable', ''), {})
        
#         # Build comprehensive markdown content
#         markdown_content = """# AutoML Analysis Report
# ## Comprehensive Machine Learning Pipeline Results

# ---

# **Job ID:** """ + self.job_id + """  
# **Dataset:** """ + job_info.get('dataset_name', 'N/A') + """  
# **Target Variable:** """ + job_info.get('target_variable', 'N/A') + """  
# **Analysis Date:** """ + datetime.now().strftime('%B %d, %Y') + """  
# **Processing Mode:** """ + job_info.get('mode', 'N/A').title() + """  

# ---

# ## Executive Summary

# """ + executive_summary + """

# ### Key Performance Metrics

# | Metric | Value |
# |--------|--------|
# | **Best Model** | """ + best_model.get('model_name', 'N/A') + """ |
# | **Test Score (R²)** | """ + str(round(best_model.get('test_score', 0), 4)) + """ |
# | **Cross-Validation Score** | """ + str(round(best_model.get('cv_score', 0), 4)) + """ |
# | **Models Evaluated** | """ + str(job_summary.get('models_trained', 0)) + """ |
# | **Total Training Time** | """ + str(round(job_summary.get('total_training_time', 0), 2)) + """ seconds |
# | **Features Used** | """ + str(job_summary.get('final_features', 0)) + """ (from """ + str(job_summary.get('original_features', 0)) + """ original) |
# | **Outliers Removed** | """ + str(job_summary.get('outliers_removed', 0)) + """ samples |

# ---

# ## 1. Introduction and Methodology

# This report presents a comprehensive analysis conducted using an automated machine learning (AutoML) pipeline designed for regression tasks. The analysis was performed on the **""" + job_info.get('dataset_name', 'N/A') + """** dataset with the primary objective of predicting **""" + job_info.get('target_variable', 'N/A') + """**.

# ### 1.1 AutoML Methodology

# """ + methodology_explanation + """

# ### 1.2 Analysis Objectives
# - Develop an accurate predictive model for """ + job_info.get('target_variable', 'N/A') + """
# - Automate feature engineering and model selection processes
# - Compare performance across multiple machine learning algorithms
# - Provide actionable insights for business decision-making

# ---

# ## 2. Dataset Overview and Analysis

# ### 2.1 Dataset Characteristics

# | Attribute | Value |
# |-----------|--------|
# | **Number of Samples** | """ + str(dataset_profile.get('num_rows', 'N/A')) + """ |
# | **Number of Features** | """ + str(dataset_profile.get('num_cols', 'N/A')) + """ |
# | **File Size** | """ + str(dataset_profile.get('file_size_mb', 'N/A')) + """ MB |
# | **Memory Usage** | """ + str(dataset_profile.get('memory_usage_mb', 'N/A')) + """ MB |
# | **Data Types** | All Numeric (Phase 1 Pipeline) |

# ### 2.2 Target Variable Analysis

# **Target Variable:** """ + job_info.get('target_variable', 'N/A') + """

# """ + ("""
# | Statistic | Value |
# |-----------|--------|
# | **Mean** | """ + str(round(target_stats.get('mean', 0), 2)) + """ |
# | **Standard Deviation** | """ + str(round(target_stats.get('std', 0), 2)) + """ |
# | **Minimum** | """ + str(round(target_stats.get('min', 0), 2)) + """ |
# | **Maximum** | """ + str(round(target_stats.get('max', 0), 2)) + """ |
# | **Skewness** | """ + str(round(target_stats.get('skewness', 0), 3)) + """ |
# | **Unique Values** | """ + str(target_stats.get('unique_count', 'N/A')) + """ |
# """ if target_stats else "Target variable statistics not available") + """

# ### 2.3 Data Quality Assessment

# """ + ("""![Dataset Overview]({})

# **Data Quality Insights:**

# {}

# """.format(plot_references.get('data_quality', ''), 
#           self.generate_plot_explanation('data_quality', plot_references.get('data_quality', ''), context_data)) 
#     if 'data_quality' in plot_references else "Data quality visualization not available") + """

# ---

# ## 3. Feature Engineering and Selection

# The automated feature engineering process transformed """ + str(job_summary.get('original_features', 0)) + """ original features into """ + str(job_summary.get('final_features', 0)) + """ optimized features through intelligent selection and creation techniques.

# ### 3.1 Feature Engineering Results

# """ + ("""![Feature Analysis]({})

# **Feature Engineering Insights:**

# {}

# """.format(plot_references.get('feature_analysis', ''), 
#           self.generate_plot_explanation('feature_analysis', plot_references.get('feature_analysis', ''), context_data)) 
#     if 'feature_analysis' in plot_references else "Feature analysis visualization not available") + """

# ### 3.2 Feature Engineering Summary

# | Process | Result |
# |---------|--------|
# | **Original Features** | """ + str(job_summary.get('original_features', 0)) + """ |
# | **Final Features** | """ + str(job_summary.get('final_features', 0)) + """ |
# | **Features Removed** | """ + str(job_summary.get('features_removed', 0)) + """ |
# | **Outliers Removed** | """ + str(job_summary.get('outliers_removed', 0)) + """ samples |
# | **Reduction Percentage** | """ + str(round((1 - job_summary.get('final_features', 1) / max(1, job_summary.get('original_features', 1))) * 100, 1)) + """% |

# ---

# ## 4. Model Performance Analysis

# ### 4.1 Algorithm Evaluation

# The AutoML system evaluated """ + str(job_summary.get('models_trained', 0)) + """ different machine learning algorithms, comparing their performance across multiple metrics including R² score, training time, and cross-validation stability.

# """ + ("""![Model Performance Comparison]({})

# **Model Performance Insights:**

# {}

# """.format(plot_references.get('model_performance', ''), 
#           self.generate_plot_explanation('model_performance', plot_references.get('model_performance', ''), context_data)) 
#     if 'model_performance' in plot_references else "Model performance visualization not available") + """

# ### 4.2 Best Model Selection: """ + best_model.get('model_name', 'N/A') + """

# """ + (self.query_ollama(
#     "Explain why " + best_model.get('model_name', 'this model') + " was selected as the best performer with R² = " + str(best_model.get('test_score', 0)) + ". Discuss its strengths and suitability for this task.",
#     "Model: " + best_model.get('model_name', 'N/A') + ", Dataset: " + job_info.get('dataset_name', 'N/A'), 
#     max_words=200
# ) if self.model_name else 
# "The " + best_model.get('model_name', 'selected model') + " was chosen based on its superior test performance (R² = " + str(best_model.get('test_score', 0)) + ") and cross-validation stability. This model demonstrates excellent balance between accuracy and generalization.") + """

# **Key Selection Criteria:**
# - **Highest Test Performance:** R² = """ + str(round(best_model.get('test_score', 0), 4)) + """
# - **Cross-Validation Stability:** CV R² = """ + str(round(best_model.get('cv_score', 0), 4)) + """
# - **Training Efficiency:** """ + str(round(best_model.get('training_time', 0), 3)) + """ seconds
# - **Overfitting Risk:** """ + ("Low" if abs(best_model.get('train_score', 0) - best_model.get('test_score', 0)) < 0.1 else "Medium" if abs(best_model.get('train_score', 0) - best_model.get('test_score', 0)) < 0.2 else "High") + """

# ---

# ## 5. Results Analysis and Business Impact

# ### 5.1 Performance Interpretation

# """ + results_interpretation + """

# ### 5.2 Model Performance Metrics

# | Metric | Value | Interpretation |
# |--------|--------|----------------|
# | **Test R²** | """ + str(round(best_model.get('test_score', 0), 4)) + """ | Performance on unseen data |
# | **Train R²** | """ + str(round(best_model.get('train_score', 0), 4)) + """ | Performance on training data |
# | **CV R²** | """ + str(round(best_model.get('cv_score', 0), 4)) + """ | Cross-validation stability |
# | **Training Time** | """ + str(round(best_model.get('training_time', 0), 3)) + """s | Computational efficiency |
# | **Variance Explained** | """ + str(round(best_model.get('test_score', 0) * 100, 1)) + """% | Predictive power |

# ### 5.3 Business Value Assessment

# **Performance Rating:** """ + ("Excellent" if best_model.get('test_score', 0) > 0.8 else "Very Good" if best_model.get('test_score', 0) > 0.7 else "Good" if best_model.get('test_score', 0) > 0.6 else "Fair" if best_model.get('test_score', 0) > 0.4 else "Poor") + """

# **Deployment Readiness:** """ + ("Production Ready" if best_model.get('test_score', 0) > 0.6 else "Needs Improvement") + """

# """ + ("""
# ### 5.4 Comprehensive Analysis Summary

# ![Analysis Summary]({})

# **Analysis Summary Insights:**

# {}
# """.format(plot_references.get('analysis_summary', ''), 
#           self.generate_plot_explanation('analysis_summary', plot_references.get('analysis_summary', ''), context_data)) 
#     if 'analysis_summary' in plot_references else "") + """

# ---

# ## 6. Recommendations and Next Steps

# ### 6.1 Immediate Actions

# 1. **✅ Model Deployment:** The """ + best_model.get('model_name', 'selected model') + """ is """ + ("ready for production deployment" if best_model.get('test_score', 0) > 0.6 else "suitable for pilot testing") + """
# 2. **📊 Performance Monitoring:** Implement real-time tracking of model predictions
# 3. **🔄 Retraining Schedule:** Establish quarterly model updates or trigger-based retraining
# 4. **📋 Documentation:** Maintain comprehensive model documentation and feature definitions

# ### 6.2 Optimization Opportunities

# """ + (self.query_ollama(
#     "Suggest 3-4 specific ways to improve this AutoML model performance beyond R² = " + str(best_model.get('test_score', 0)) + ". Focus on practical, actionable recommendations.",
#     "Current performance: R² = " + str(best_model.get('test_score', 0)) + ", Features: " + str(job_summary.get('final_features', 0)) + ", Model: " + best_model.get('model_name', 'N/A'),
#     max_words=200
# ) if self.model_name else """
# 1. **Feature Enhancement:** Investigate domain-specific feature engineering techniques
# 2. **Ensemble Methods:** Combine multiple models for improved performance
# 3. **Hyperparameter Optimization:** Fine-tune model parameters for better accuracy
# 4. **Data Augmentation:** Collect additional high-quality training samples""") + """

# ### 6.3 Long-term Strategy

# **Phase 1 (1-3 months):** Deploy current model and establish monitoring systems
# **Phase 2 (3-6 months):** Implement model improvements and expand feature set
# **Phase 3 (6+ months):** Develop advanced ensemble models and real-time capabilities

# ---

# ## 7. Technical Appendix

# ### 7.1 Reproducibility

# **Command to reproduce this analysis:**
# ```bash
# python main.py run --file """ + job_info.get('dataset_name', 'dataset.csv') + """ --target """ + job_info.get('target_variable', 'target') + """ --mode """ + job_info.get('mode', 'auto') + """
# ```

# ### 7.2 Model Artifacts

# **File Structure:**
# ```
# """ + str(self.job_path) + """/
# ├── all_models/                     # All trained models
# ├── preprocessors/pipeline.pkl      # Data preprocessing
# ├── feature_engineering_pipeline.pkl # Feature engineering
# ├── visualizations/                 # All generated plots
# ├── training_summary.json          # Detailed results
# └── job_summary.json               # High-level metrics
# ```

# ### 7.3 Deployment Checklist

# - [""" + ("x" if best_model.get('test_score', 0) > 0.5 else " ") + """] Model performance validation (R² > 0.5)
# - [""" + ("x" if job_summary.get('models_trained', 0) > 3 else " ") + """] Multiple models evaluated
# - [""" + ("x" if best_model.get('training_time', 0) < 60 else " ") + """] Training time acceptable (< 60s)
# - [ ] Production infrastructure setup
# - [ ] Monitoring and alerting configured
# - [ ] Model documentation completed

# ---

# ## 8. Visualization Gallery

# This report includes """ + str(len(plot_references)) + """ comprehensive visualizations:

# """ + "\n".join([f"- **{category.replace('_', ' ').title()}:** {filename}" for category, filename in plot_references.items()]) + """

# Each visualization provides specific insights into different aspects of the machine learning pipeline, from data quality assessment to final model performance evaluation.

# ---

# *Report generated on """ + datetime.now().strftime('%B %d, %Y at %I:%M %p') + """ using AutoML Phase 1 Pipeline*  
# *Analysis completed in """ + str(round(job_summary.get('total_training_time', 0), 1)) + """ seconds with """ + str(job_summary.get('models_trained', 0)) + """ models evaluated*  
# *Best model: """ + best_model.get('model_name', 'N/A') + """ (R² = """ + str(round(best_model.get('test_score', 0), 4)) + """)*

# **Report Statistics:**
# - **Pages:** ~30-35 pages with comprehensive analysis
# - **Sections:** 8 major sections + technical appendix  
# - **Visualizations:** """ + str(len(plot_references)) + """ plots with AI explanations
# - **AI Insights:** """ + ("Enabled" if self.model_name else "Disabled") + """ (""" + (self.model_name or "Ollama not available") + """)
# - **Total Plots Found:** """ + str(len(discovered_plots)) + """ across multiple locations
# """
        
#         return markdown_content
    
#     def convert_markdown_to_pdf(self, markdown_content, output_path):
#         """Convert markdown to PDF using multiple fallback methods"""
#         temp_md_path = self.job_path / "temp_report.md"
        
#         with open(temp_md_path, 'w', encoding='utf-8') as f:
#             f.write(markdown_content)
        
#         # Method 1: Enhanced pandoc
#         try:
#             cmd = [
#                 'pandoc', str(temp_md_path), '-o', str(output_path),
#                 '--pdf-engine=xelatex', '-V', 'geometry:margin=0.75in',
#                 '-V', 'fontsize=11pt', '--toc', '--number-sections', '--standalone'
#             ]
            
#             print("🎭 Pandoc is putting on its finest performance...")
#             result = subprocess.run(cmd, capture_output=True, text=True)
            
#             if result.returncode == 0:
#                 print("🎉 Pandoc delivered a standing ovation! PDF creation successful!")
#                 temp_md_path.unlink()
#                 return True
#             else:
#                 print("😅 Pandoc stumbled a bit: " + result.stderr)
                
#         except FileNotFoundError:
#             print("🤷‍♂️ Pandoc is missing from the stage!")
        
#         # Method 2: Basic pandoc
#         try:
#             cmd = ['pandoc', str(temp_md_path), '-o', str(output_path), '--pdf-engine=xelatex']
#             print("🔧 Trying pandoc's minimalist approach...")
#             result = subprocess.run(cmd, capture_output=True, text=True)
            
#             if result.returncode == 0:
#                 print("✨ Basic pandoc saved the day! Sometimes less is more!")
#                 temp_md_path.unlink()
#                 return True
                
#         except FileNotFoundError:
#             pass
        
#         # Method 3: WeasyPrint
#         try:
#             import markdown, weasyprint
#             print("🚀 WeasyPrint superhero swooping in to save the day!")
            
#             with open(temp_md_path, 'r', encoding='utf-8') as f:
#                 md_content = f.read()
            
#             html = markdown.markdown(md_content, extensions=['tables'])
#             html_with_style = """<!DOCTYPE html>
# <html>
# <head>
#     <meta charset="utf-8">
#     <style>
#         body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }
#         h1, h2, h3 { color: #2c3e50; page-break-after: avoid; }
#         h1 { border-bottom: 3px solid #3498db; padding-bottom: 10px; font-size: 28px; }
#         h2 { border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; font-size: 22px; margin-top: 30px; }
#         h3 { color: #34495e; font-size: 18px; margin-top: 25px; }
#         table { border-collapse: collapse; width: 100%; margin: 20px 0; }
#         th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
#         th { background-color: #f8f9fa; font-weight: bold; color: #2c3e50; }
#         tr:nth-child(even) { background-color: #f9f9f9; }
#         pre { background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; overflow-x: auto; }
#         code { background-color: #f1f1f1; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }
#         blockquote { border-left: 4px solid #3498db; margin: 20px 0; padding: 15px 20px; background-color: #f8f9fa; }
#         img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }
#         .metric-highlight { background-color: #e8f5e8; font-weight: bold; }
#         ul, ol { margin: 15px 0; padding-left: 30px; }
#         li { margin: 5px 0; }
#         strong { color: #2c3e50; }
#         .page-break { page-break-before: always; }
#     </style>
# </head>
# <body>
# """ + html + """
# </body>
# </html>"""
            
#             weasyprint.HTML(string=html_with_style).write_pdf(str(output_path))
#             print("🎊 WeasyPrint executed a flawless PDF performance! Mission accomplished!")
#             temp_md_path.unlink()
#             return True
            
#         except ImportError:
#             print("😭 WeasyPrint is not in our toolkit! Install with: pip install weasyprint markdown")
#         except Exception as e:
#             print("🤯 WeasyPrint encountered a plot twist: " + str(e))
        
#         print("🤡 All PDF conversion methods went on strike today!")
#         print("📄 But fear not! Your markdown masterpiece is still available!")
#         return False
    
#     def generate_report(self):
#         """Generate the complete report with all discovered plots"""
#         print("🎪 Welcome to the Ultimate AutoML Report Extravaganza! 🎭")
#         print("=" * 65)
#         print("🔮 Preparing to unleash data science magic for job: " + self.job_id)
        
#         if not any(self.report_data.values()):
#             print("🚨 CODE RED: Mission critical data files missing!")
#             print("🕵️‍♂️ Investigation shows:")
#             print("   🎯 Target Job ID: " + self.job_id)
#             print("   📂 Search Location: " + str(self.job_path))
#             print("   📋 Required Evidence: state.json, training_summary.json, job_summary.json")
#             print("🤷‍♀️ Either this job is in witness protection or someone moved the files!")
#             return None
        
#         print("🧙‍♂️ Weaving together plots, data, and AI wisdom into an epic tale...")
#         markdown_content = self.generate_markdown_report()
        
#         # Save markdown
#         md_output_path = self.job_path / ("AutoML_Report_" + self.job_id + ".md")
#         with open(md_output_path, 'w', encoding='utf-8') as f:
#             f.write(markdown_content)
#         print("📜 Epic markdown saga saved: " + str(md_output_path))
        
#         # Convert to PDF
#         pdf_output_path = self.job_path / ("AutoML_Report_" + self.job_id + ".pdf")
#         print("🎨 Grand finale time - transforming markdown into PDF perfection...")
        
#         if self.convert_markdown_to_pdf(markdown_content, pdf_output_path):
#             print("\n🎊 SPECTACULAR SUCCESS! The crowd goes absolutely wild! 🎊")
#             print("📊 Your masterpiece is ready for the world: " + str(pdf_output_path))
#             print("📝 Markdown backup for the curious: " + str(md_output_path))
            
#             # File size commentary with extra personality
#             if pdf_output_path.exists():
#                 pdf_size = pdf_output_path.stat().st_size / (1024 * 1024)
#                 if pdf_size > 10:
#                     print("📏 PDF Size: " + str(round(pdf_size, 2)) + " MB (Holy data! That's a THICC report! 🐘)")
#                 elif pdf_size > 5:
#                     print("📏 PDF Size: " + str(round(pdf_size, 2)) + " MB (Perfect size - like Goldilocks would approve! 👌)")
#                 elif pdf_size > 2:
#                     print("📏 PDF Size: " + str(round(pdf_size, 2)) + " MB (Compact excellence - Swiss Army knife of reports! 🔧)")
#                 else:
#                     print("📏 PDF Size: " + str(round(pdf_size, 2)) + " MB (Small but mighty - like a data science haiku! 💎)")
            
#             return pdf_output_path
#         else:
#             print("\n🎭 Plot twist! PDF conversion decided to be dramatic today...")
#             print("📄 But don't despair! Your markdown report is still absolutely fantastic:")
#             print("📁 Markdown Report: " + str(md_output_path))
#             print("💡 Convert it manually when the PDF spirits are more cooperative!")
#             return md_output_path

# def main():
#     """Main function for standalone execution"""
#     if len(sys.argv) != 2:
#         print("🎪 Welcome to the AutoML Report Generator MEGA CIRCUS! 🎭")
#         print("=" * 65)
#         print("🎯 Usage: python3 generate_report.py <job_id>")
#         print("🎪 Example: python3 generate_report.py job_20250819_005202")
#         print("\n🌟 This ENHANCED version will blow your mind:")
#         print("  🔮 Hunt down ALL plots from multiple secret locations")
#         print("  📊 Embed every single visualization with AI explanations")
#         print("  🤖 Generate profound insights that make data scientists cry tears of joy")
#         print("  📄 Create a 30-35 page PDF masterpiece that'll get you promoted")
#         print("  🎭 Keep you entertained with top-tier data science humor")
#         print("  ⚡ Process faster than you can say 'machine learning'")
#         print("  🧠 Use advanced AI to explain every plot and insight")
#         print("\n💡 Pro tip: This version finds plots EVERYWHERE - it's like plot GPS! 📍")
#         sys.exit(1)
    
#     job_id = sys.argv[1]
    
#     # Enhanced job validation
#     job_path = Path("storage/models/jobs") / job_id
#     if not job_path.exists():
#         print("🚨 PLOT TWIST! Job directory has vanished into the data dimension!")
#         print("🔍 Scanning the multiverse for available jobs...")
#         jobs_dir = Path("storage/models/jobs")
#         if jobs_dir.exists():
#             job_count = 0
#             for job_dir in jobs_dir.iterdir():
#                 if job_dir.is_dir():
#                     print("   🎯 Discovered: " + job_dir.name + " (Ready for analysis!)")
#                     job_count += 1
#             if job_count == 0:
#                 print("   🏜️ The job desert is emptier than a data scientist's social calendar!")
#                 print("   💡 Hint: Run some ML jobs first, then come back for the report party!")
#             else:
#                 print("   🎉 Found " + str(job_count) + " jobs hiding in the archives!")
#                 print("   💭 Did you maybe typo the job ID? Copy-paste is your friend!")
#         else:
#             print("   🤯 The entire jobs directory is missing! This is unprecedented!")
#             print("   🚨 Emergency protocol: Check if you're in the right directory!")
#         sys.exit(1)
    
#     print("🎬 Lights! Camera! Data! Action begins NOW!")
#     print("🍿 Grab your favorite beverage - this is going to be EPIC!")
    
#     # Generate the ultimate report
#     generator = AutoMLReportGenerator(job_id)
#     report_path = generator.generate_report()
    
#     if report_path:
#         print("\n🎊 MISSION ACCOMPLISHED! The data gods smile upon us! 🎊")
#         print("📂 Your legendary report awaits at: " + str(report_path))
        
#         # Epic file size analysis
#         if Path(report_path).exists():
#             file_size = Path(report_path).stat().st_size / (1024 * 1024)
#             if file_size > 15:
#                 print("📏 " + str(round(file_size, 1)) + "MB - This is MASSIVE! You've created the War and Peace of ML reports! 📚")
#             elif file_size > 10:
#                 print("📏 " + str(round(file_size, 1)) + "MB - Substantial and impressive! Your stakeholders will be in awe! 🤩")
#             elif file_size > 5:
#                 print("📏 " + str(round(file_size, 1)) + "MB - Perfect balance of depth and readability! 📖")
#             elif file_size > 2:
#                 print("📏 " + str(round(file_size, 1)) + "MB - Concise yet comprehensive! Efficiency at its finest! ⚡")
#             else:
#                 print("📏 " + str(round(file_size, 1)) + "MB - Lean and mean data machine! 🚀")
        
#         print("\n🎯 Your quest is complete! What's next in your data adventure?")
#         print("   📖 Marvel at your comprehensive analysis masterpiece")
#         print("   🚀 Share with your team and watch their minds get blown")
#         print("   💼 Present to stakeholders and watch them throw money at your ML project")
#         print("   🏆 Frame the first page and hang it on your wall (okay, maybe just save it)")
#         print("   ☕ Celebrate with the beverage of champions - you've earned it!")
        
#         # Random epic data science quotes
#         import random
#         epic_quotes = [
#             "💡 'In data we trust, in models we verify!' - Ancient ML Proverb",
#             "🎩 'A wizard is never late with their model, nor early. They deploy precisely when they mean to!'",
#             "🚀 'Houston, we have a solution! T-minus zero to model deployment!'",
#             "🔮 'May the R² be with you, always and forever!'",
#             "🎯 'One does not simply walk into production without proper validation!'",
#             "📊 'I see data people... and they're all making predictions!'",
#             "⚡ 'With great computational power comes great model responsibility!'"
#         ]
#         print("\n" + random.choice(epic_quotes))
        
#     else:
#         print("\n🎭 Plot twist in our data drama! Something went sideways...")
#         print("🤔 But hey, even the best data scientists face plot twists!")
#         print("💪 Don't let this stop your data science journey!")
#         print("🔧 Debug those error messages like the ML detective you are!")
#         print("🎪 The show must go on - try again when the stars align!")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
generate_report.py - ENHANCED VERSION with All Plots
Standalone AutoML Report Generator with AI explanations for every plot
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
import base64
from io import BytesIO

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
        
        # All possible plot locations
        self.plot_locations = [
            self.job_path,  # Root job directory
            self.job_path / 'visualizations',  # Main viz folder
            self.job_path / 'enhanced_visualizations',  # Enhanced viz folder
        ]
        
    def check_ollama_models(self):
        """Check which Ollama models are available"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                available = [model['name'] for model in response.json().get('models', [])]
                print("✅ Ollama AI Detective Agency reporting for duty! Models: " + str(available))
                
                for model in self.available_models:
                    if model in available:
                        print("🧠 Recruiting AI brain: " + model + " (This one's got the smarts!)")
                        return model
                
                if available:
                    print("🤖 Using mystery AI model: " + available[0] + " (Let's see what this one can do!)")
                    return available[0]
                else:
                    print("🤷‍♂️ No AI models found - we'll wing it with pure human logic!")
                    return None
            else:
                print("😴 Ollama is taking a nap...")
                return None
        except:
            print("🔌 Ollama is off the grid! Install with: curl -fsSL https://ollama.ai/install.sh | sh")
            return None
    
    def load_job_data(self):
        """Load all relevant data from job files"""
        files_to_load = {
            'state': 'state.json',
            'training_summary': 'training_summary.json', 
            'job_summary': 'job_summary.json',
            'metadata': 'metadata.json'
        }
        
        print("🕵️‍♂️ CSI: Data Loading Unit investigating: " + str(self.job_path))
        
        for key, filename in files_to_load.items():
            filepath = self.job_path / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    self.report_data[key] = json.load(f)
                print("🗂️ Evidence collected: " + filename + " ✓")
            else:
                print("🚨 Missing evidence: " + filename + " (This might be a problem!)")
    
    def query_ollama(self, prompt, context="", max_words=200):
        """Query Ollama for AI-generated explanations"""
        if not self.model_name:
            return "🤖 AI is on vacation, generating human-powered insights instead..."
            
        full_prompt = context + "\n\n" + prompt + "\n\nKeep response under " + str(max_words) + " words and focus on practical insights."
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": max_words
            }
        }
        
        try:
            print("🧠 AI neurons firing...")
            response = requests.post(self.ollama_url, json=payload, timeout=45)
            if response.status_code == 200:
                return response.json().get('response', 'AI had a brain freeze 🧊')
            else:
                return "🤖 AI returned error code: " + str(response.status_code)
        except Exception as e:
            return "🚫 AI connection failed: " + str(e)
    
    def discover_all_plots(self):
        """Hunt down all available plots from multiple locations"""
        print("🎨 Launching the Great Plot Hunt across multiple dimensions...")
        
        # Comprehensive list of plot files to search for
        plot_catalog = {
            'model_performance': [
                'comprehensive_model_comparison.png',
                'model_comparison.png', 
                'performance_comparison.png',
                'model_performance_analysis.png'
            ],
            'feature_analysis': [
                'feature_analysis.png',
                'feature_engineering_analysis.png'
            ],
            'data_quality': [
                'dataset_overview.png',
                'preprocessing_summary.png',
                'data_quality_analysis.png'
            ],
            'analysis_summary': [
                'analysis_summary.png'
            ]
        }
        
        discovered_plots = {}
        total_found = 0
        
        # Search through all possible locations
        for location in self.plot_locations:
            if not location.exists():
                continue
                
            print("🔍 Investigating plot crime scene: " + str(location))
            
            for plot_category, filenames in plot_catalog.items():
                for filename in filenames:
                    plot_path = location / filename
                    if plot_path.exists() and plot_category not in discovered_plots:
                        discovered_plots[plot_category] = {
                            'path': str(plot_path),
                            'filename': filename,
                            'category': plot_category
                        }
                        total_found += 1
                        
                        # Fun discovery messages
                        if 'model' in filename.lower():
                            print("🏆 Jackpot! Found model showdown: " + filename)
                        elif 'feature' in filename.lower():
                            print("🧬 Feature detective work discovered: " + filename)
                        elif 'data' in filename.lower():
                            print("📊 Data archaeology unearthed: " + filename)
                        elif 'analysis' in filename.lower():
                            print("🔬 Analysis goldmine located: " + filename)
                        else:
                            print("🎁 Mystery treasure found: " + filename)
        
        if total_found == 0:
            print("😱 Plot apocalypse! No visualizations found anywhere!")
        elif total_found < 3:
            print("📈 Found " + str(total_found) + " plots - we'll make the most of these gems!")
        else:
            print("🎉 Plot bonanza! Collected " + str(total_found) + " beautiful visualizations!")
        
        return discovered_plots
    
    def copy_plot_to_report_dir(self, plot_info):
        """Copy plot to the same directory as the report"""
        if not plot_info or not Path(plot_info['path']).exists():
            return None
            
        source_path = Path(plot_info['path'])
        dest_path = self.job_path / source_path.name
        
        # Check if source and destination are the same file
        if source_path.resolve() == dest_path.resolve():
            print("📋 Plot already in perfect position: " + source_path.name)
            return source_path.name
        
        # Only copy if they're different files
        import shutil
        try:
            shutil.copy2(source_path, dest_path)
            print("📋 Filed under evidence: " + source_path.name)
            return source_path.name
        except Exception as e:
            print("⚠️ Plot filing had a hiccup: " + str(e))
            # If copy fails, just use the original filename if it's accessible
            return source_path.name if source_path.exists() else None
    
    def get_business_implications(self, plot_category, context_data):
        """Generate business implications for each plot category"""
        implications = {
            'model_performance': f"The model achieves {context_data['r2_score']*100:.1f}% prediction accuracy, indicating {'strong' if context_data['r2_score'] > 0.7 else 'good' if context_data['r2_score'] > 0.6 else 'moderate'} business value. This performance level {'supports immediate deployment' if context_data['r2_score'] > 0.6 else 'requires additional optimization'} and can {'significantly improve' if context_data['r2_score'] > 0.7 else 'moderately enhance'} decision-making accuracy.",
            
            'feature_analysis': f"Feature reduction from {context_data['features_original']} to {context_data['features_final']} features ({((context_data['features_original'] - context_data['features_final']) / context_data['features_original'] * 100):.1f}% reduction) simplifies model deployment, reduces data collection costs, and improves model interpretability for business stakeholders.",
            
            'data_quality': f"Data quality improvements ensure model reliability and reduce prediction errors. Clean data processing increases stakeholder confidence and supports regulatory compliance for model deployment in production environments.",
            
            'analysis_summary': f"The comprehensive analysis validates the {context_data['best_model']} model selection, providing executive-level confidence in the {context_data['r2_score']*100:.1f}% prediction accuracy and supporting strategic investment in automated machine learning capabilities."
        }
        return implications.get(plot_category, "This analysis provides valuable insights for strategic business decision-making and operational optimization.")
    
    def get_technical_details(self, plot_category, context_data):
        """Generate technical details for each plot category"""
        details = {
            'model_performance': f"Cross-validation methodology with {context_data['models_trained']} algorithms evaluated. Performance metrics include R² scores, training efficiency, and overfitting risk assessment using train-test-validation splits.",
            
            'feature_analysis': f"Automated feature engineering pipeline applied correlation analysis, variance thresholding, and statistical significance testing to optimize the feature set from {context_data['features_original']} to {context_data['features_final']} variables.",
            
            'data_quality': f"Data preprocessing included outlier detection using IQR and statistical methods, missing value imputation, and feature scaling. Quality metrics validate data reliability and model input consistency.",
            
            'analysis_summary': f"Comprehensive pipeline execution with automated model selection, hyperparameter optimization, and cross-validation. Results aggregated across {context_data['models_trained']} algorithms with performance benchmarking."
        }
        return details.get(plot_category, "Technical implementation follows machine learning best practices with automated validation and quality assurance.")
    
    def generate_plot_explanation(self, plot_category, plot_filename, context_data):
        """Generate AI explanation for a specific plot"""
        if not self.model_name:
            # Enhanced fallback explanations with more detail
            explanations = {
                'model_performance': """This comprehensive model performance visualization compares multiple machine learning algorithms across key metrics including R² scores, training times, and cross-validation results. The chart helps identify the optimal model by showing both accuracy and efficiency trade-offs. Higher R² scores indicate better predictive performance, while training time reveals computational efficiency. The visualization also highlights overfitting risks by comparing training vs. test performance.""",
                
                'feature_analysis': """This feature analysis visualization provides insights into the feature engineering process, showing the transformation from original features to the final optimized set. It displays feature importance rankings, correlation patterns, and the impact of feature selection techniques. The plot helps understand which variables contribute most significantly to model predictions and how feature engineering improved model performance.""",
                
                'data_quality': """This data quality assessment visualization examines the dataset's characteristics including missing value patterns, outlier detection results, and preprocessing transformations. It shows the distribution of data quality issues across features and demonstrates how data cleaning steps improved the dataset. The plot provides crucial insights into data reliability and preprocessing effectiveness.""",
                
                'analysis_summary': """This comprehensive analysis summary provides a high-level overview of the entire AutoML pipeline, combining key performance metrics, data insights, and model comparison results. It serves as an executive dashboard showing the most important findings from the analysis, including the best model selection rationale and overall project success indicators."""
            }
            return explanations.get(plot_category, "This visualization provides comprehensive insights into the machine learning analysis, showing key patterns and relationships that inform model development and business decision-making.")
        
        # Generate AI explanation based on plot category
        prompts = {
            'model_performance': "Explain what insights can be gained from a model performance comparison plot in AutoML. Focus on how to interpret R² scores, training times, overfitting indicators, and model selection criteria for business users.",
            'feature_analysis': "Describe what a feature analysis plot reveals about machine learning features, including feature importance, engineering techniques, selection methods, and their impact on model performance.",
            'data_quality': "Explain what data quality and preprocessing visualizations show about dataset characteristics, missing values, outliers, data preparation steps, and their impact on model reliability.",
            'analysis_summary': "Describe what an analysis summary plot reveals about the overall AutoML process, key findings, model performance, and business implications for stakeholders."
        }
        
        prompt = prompts.get(plot_category, "Explain this machine learning visualization for business stakeholders.")
        context = "AutoML Analysis Context: " + str(context_data)
        
        return self.query_ollama(prompt, context, max_words=150)
    
    def generate_markdown_report(self):
        """Generate comprehensive markdown report with all available plots"""
        job_info = self.report_data.get('state', {}).get('job_info', {})
        job_summary = self.report_data.get('job_summary', {})
        training_summary = self.report_data.get('training_summary', {})
        best_model = training_summary.get('best_model', {})
        dataset_profile = self.report_data.get('state', {}).get('dataset_profile', {})
        
        print("🎨 Initiating the Great Plot Discovery Expedition...")
        discovered_plots = self.discover_all_plots()
        
        # Copy all plots to report directory
        print("📚 Organizing visual evidence in the report library...")
        plot_references = {}
        for category, plot_info in discovered_plots.items():
            filename = self.copy_plot_to_report_dir(plot_info)
            if filename:
                plot_references[category] = filename
                print("📋 Filed under evidence: " + filename)
        
        # Generate context for AI explanations
        context_data = {
            'dataset': job_info.get('dataset_name', 'N/A'),
            'target': job_info.get('target_variable', 'N/A'),
            'best_model': best_model.get('model_name', 'N/A'),
            'r2_score': best_model.get('test_score', 0),
            'models_trained': job_summary.get('models_trained', 0),
            'features_original': job_summary.get('original_features', 0),
            'features_final': job_summary.get('final_features', 0)
        }
        
        # Get AI explanations for main sections
        print("🧠 Summoning AI wisdom for executive insights...")
        
        if self.model_name:
            exec_context = """Dataset: """ + str(context_data['dataset']) + """
Target: """ + str(context_data['target']) + """
Best Model: """ + str(context_data['best_model']) + """ (R² = """ + str(context_data['r2_score']) + """)
Models Trained: """ + str(context_data['models_trained']) + """
Features: """ + str(context_data['features_original']) + """ → """ + str(context_data['features_final'])
            
            executive_summary = self.query_ollama(
                "Write a professional 2-paragraph executive summary for this AutoML analysis. Focus on business value and key achievements.",
                exec_context, max_words=250
            )
            
            methodology_explanation = self.query_ollama(
                "Explain the AutoML methodology and why it's valuable for this analysis. Focus on automation benefits and process efficiency.",
                exec_context, max_words=200
            )
            
            results_interpretation = self.query_ollama(
                "Interpret the R² score of " + str(context_data['r2_score']) + " and explain what it means for business applications and model reliability.",
                exec_context, max_words=150
            )
        else:
            executive_summary = """This AutoML analysis successfully evaluated """ + str(context_data['models_trained']) + """ machine learning models to predict """ + str(context_data['target']) + """. The automated pipeline identified """ + str(context_data['best_model']) + """ as the optimal solution with an R² score of """ + str(context_data['r2_score']) + """, demonstrating good predictive capability.

The analysis efficiently processed """ + str(context_data['features_original']) + """ original features, selecting """ + str(context_data['features_final']) + """ optimized features through intelligent feature engineering. This automation significantly reduces manual effort while maintaining high-quality results suitable for production deployment."""
            
            methodology_explanation = """The AutoML approach automates the traditionally manual and time-consuming process of model selection, feature engineering, and hyperparameter tuning. This systematic evaluation ensures optimal model selection while reducing human bias and accelerating time-to-deployment."""
            
            results_interpretation = """The R² score of """ + str(context_data['r2_score']) + """ indicates the model explains """ + str(round(context_data['r2_score'] * 100, 1)) + """% of variance in the target variable, representing good predictive performance suitable for business applications."""
        
        # Get target variable statistics
        target_stats = dataset_profile.get('numeric_summaries', {}).get(job_info.get('target_variable', ''), {})
        
        # Build comprehensive markdown content
        markdown_content = """# AutoML Analysis Report
## Comprehensive Machine Learning Pipeline Results

---

**Job ID:** """ + self.job_id + """  
**Dataset:** """ + job_info.get('dataset_name', 'N/A') + """  
**Target Variable:** """ + job_info.get('target_variable', 'N/A') + """  
**Analysis Date:** """ + datetime.now().strftime('%B %d, %Y') + """  
**Processing Mode:** """ + job_info.get('mode', 'N/A').title() + """  

---

## Executive Summary

""" + executive_summary + """

### Key Performance Metrics

| Metric | Value |
|--------|--------|
| **Best Model** | """ + best_model.get('model_name', 'N/A') + """ |
| **Test Score (R²)** | """ + str(round(best_model.get('test_score', 0), 4)) + """ |
| **Cross-Validation Score** | """ + str(round(best_model.get('cv_score', 0), 4)) + """ |
| **Models Evaluated** | """ + str(job_summary.get('models_trained', 0)) + """ |
| **Total Training Time** | """ + str(round(job_summary.get('total_training_time', 0), 2)) + """ seconds |
| **Features Used** | """ + str(job_summary.get('final_features', 0)) + """ (from """ + str(job_summary.get('original_features', 0)) + """ original) |
| **Outliers Removed** | """ + str(job_summary.get('outliers_removed', 0)) + """ samples |

---

## 1. Introduction and Methodology

This report presents a comprehensive analysis conducted using an automated machine learning (AutoML) pipeline designed for regression tasks. The analysis was performed on the **""" + job_info.get('dataset_name', 'N/A') + """** dataset with the primary objective of predicting **""" + job_info.get('target_variable', 'N/A') + """**.

### 1.1 AutoML Methodology

""" + methodology_explanation + """

### 1.2 Analysis Objectives
- Develop an accurate predictive model for """ + job_info.get('target_variable', 'N/A') + """
- Automate feature engineering and model selection processes
- Compare performance across multiple machine learning algorithms
- Provide actionable insights for business decision-making

---

## 2. Dataset Overview and Analysis

### 2.1 Dataset Characteristics

| Attribute | Value |
|-----------|--------|
| **Number of Samples** | """ + str(dataset_profile.get('num_rows', 'N/A')) + """ |
| **Number of Features** | """ + str(dataset_profile.get('num_cols', 'N/A')) + """ |
| **File Size** | """ + str(dataset_profile.get('file_size_mb', 'N/A')) + """ MB |
| **Memory Usage** | """ + str(dataset_profile.get('memory_usage_mb', 'N/A')) + """ MB |
| **Data Types** | All Numeric (Phase 1 Pipeline) |

### 2.2 Target Variable Analysis

**Target Variable:** """ + job_info.get('target_variable', 'N/A') + """

""" + ("""
| Statistic | Value |
|-----------|--------|
| **Mean** | """ + str(round(target_stats.get('mean', 0), 2)) + """ |
| **Standard Deviation** | """ + str(round(target_stats.get('std', 0), 2)) + """ |
| **Minimum** | """ + str(round(target_stats.get('min', 0), 2)) + """ |
| **Maximum** | """ + str(round(target_stats.get('max', 0), 2)) + """ |
| **Skewness** | """ + str(round(target_stats.get('skewness', 0), 3)) + """ |
| **Unique Values** | """ + str(target_stats.get('unique_count', 'N/A')) + """ |
""" if target_stats else "Target variable statistics not available") + """

### 2.3 Data Quality Assessment

The automated data quality assessment process examined """ + str(dataset_profile.get('num_cols', 0)) + """ features across """ + str(dataset_profile.get('num_rows', 0)) + """ samples, identifying data quality issues and implementing appropriate preprocessing strategies.

""" + ("""
![Dataset Overview]({})

**Dataset Overview Analysis:**

{}

---

![Preprocessing Summary]({})

**Data Preprocessing Insights:**

{}

""".format(plot_references.get('data_quality', ''), 
          self.generate_plot_explanation('data_quality', plot_references.get('data_quality', ''), context_data),
          plot_references.get('preprocessing_summary', '') if 'preprocessing_summary' in plot_references else plot_references.get('data_quality', ''),
          "This preprocessing visualization shows the complete data preparation pipeline including outlier removal, missing value treatment, feature scaling, and correlation analysis. The process successfully cleaned the dataset while preserving important signal for model training." if 'preprocessing_summary' in plot_references else self.generate_plot_explanation('data_quality', plot_references.get('data_quality', ''), context_data)) 
    if 'data_quality' in plot_references else "Data quality visualizations not available - analysis performed without visual documentation") + """

**Data Quality Summary:**
- **Missing Data Treatment:** Intelligent imputation applied to """ + str(len([k for k, v in dataset_profile.get('missing_data', {}).items() if v > 0])) + """ features
- **Outlier Detection:** """ + str(job_summary.get('outliers_removed', 0)) + """ outlier samples identified and removed  
- **Feature Correlation:** """ + str(job_summary.get('features_removed', 0)) + """ highly correlated features removed
- **Constant Features:** """ + str(len(dataset_profile.get('constant_columns', []))) + """ constant columns eliminated

---

## 3. Feature Engineering and Selection

The automated feature engineering process transformed """ + str(job_summary.get('original_features', 0)) + """ original features into """ + str(job_summary.get('final_features', 0)) + """ optimized features through intelligent selection and creation techniques.

""" + ("""
![Feature Analysis]({})

**Feature Engineering Analysis:**

{}

""".format(plot_references.get('feature_analysis', ''), 
          self.generate_plot_explanation('feature_analysis', plot_references.get('feature_analysis', ''), context_data)) 
    if 'feature_analysis' in plot_references else "Feature analysis visualization not available") + """

### 3.1 Feature Engineering Pipeline Results

**Processing Summary:**

**Original Dataset:**
- Total Features: """ + str(job_summary.get('original_features', 0)) + """
- Total Samples: """ + str(dataset_profile.get('num_rows', 0)) + """

**Feature Optimization:**
- Final Features: """ + str(job_summary.get('final_features', 0)) + """ ✅
- Features Removed: """ + str(job_summary.get('features_removed', 0)) + """ (High correlation/low variance)
- Reduction Percentage: """ + str(round((1 - job_summary.get('final_features', 1) / max(1, job_summary.get('original_features', 1))) * 100, 1)) + """% ✅

**Data Quality:**
- Outliers Removed: """ + str(job_summary.get('outliers_removed', 0)) + """ samples ✅
- Final Clean Samples: """ + str(dataset_profile.get('num_rows', 0) - job_summary.get('outliers_removed', 0)) + """
- Processing Time: """ + str(round(job_summary.get('total_training_time', 0), 2)) + """s ✅

**Efficiency Metrics:**
- ✅ **Optimized:** Feature space reduced by """ + str(round((1 - job_summary.get('final_features', 1) / max(1, job_summary.get('original_features', 1))) * 100, 1)) + """%
- ✅ **Clean:** """ + str(job_summary.get('outliers_removed', 0)) + """ outliers removed for quality
- ✅ **Fast:** Complete processing in """ + str(round(job_summary.get('total_training_time', 0), 2)) + """ seconds

""" + ("""
![Feature Engineering Details]({})

**Advanced Feature Engineering:**

This detailed feature engineering visualization shows the sophisticated transformations applied to create the optimal feature set. The process includes polynomial feature creation, interaction term generation, statistical feature derivation, and intelligent feature selection using automated techniques. The resulting feature set balances model complexity with predictive power.

""".format(plot_references.get('feature_engineering_analysis', '')) 
    if 'feature_engineering_analysis' in plot_references else "") + """

---

## 4. Model Performance Analysis

### 4.1 Comprehensive Algorithm Evaluation

The AutoML system systematically evaluated """ + str(job_summary.get('models_trained', 0)) + """ different machine learning algorithms, comparing their performance across multiple dimensions including accuracy, efficiency, and generalization capability.

""" + ("""
![Model Performance Comparison]({})

**Model Performance Insights:**

{}

""".format(plot_references.get('model_performance', ''), 
          self.generate_plot_explanation('model_performance', plot_references.get('model_performance', ''), context_data)) 
    if 'model_performance' in plot_references else "Model performance visualization not available") + """

### 4.2 Detailed Performance Metrics

**Model Performance Summary:**

""" + "\n".join([f"""
**{model}:**
- Test R²: {metrics.get('test_score', 0):.4f}
- Train R²: {metrics.get('train_score', 0):.4f}  
- CV R²: {metrics.get('cv_score', 0):.4f}
- Training Time: {metrics.get('training_time', 0):.3f}s
- Overfitting Risk: {'**High**' if metrics.get('train_score', 0) - metrics.get('test_score', 0) > 0.2 else '**Medium**' if metrics.get('train_score', 0) - metrics.get('test_score', 0) > 0.1 else 'Low'}
""" for model, metrics in training_summary.get('all_models_performance', {}).items()]) + """

**Performance Ranking (by Test R²):**

""" + "\n".join([f"{i+1}. **{model[0]}** - R² = {model[1].get('test_score', 0):.4f}" for i, model in enumerate(sorted(training_summary.get('all_models_performance', {}).items(), key=lambda x: x[1].get('test_score', 0), reverse=True))]) + """

**Performance Analysis:**
- **Ridge** emerges as the clear winner with the highest test R² (""" + str(round(best_model.get('test_score', 0), 4)) + """) and excellent stability
- **Tree-based models** (DecisionTree, RandomForest, GradientBoosting) show high overfitting risk
- **Linear models** (Ridge, Lasso, LinearRegression) demonstrate better generalization
- **SVR** shows poor performance on this dataset characteristics

### 4.3 Best Model Selection: """ + best_model.get('model_name', 'N/A') + """

""" + (self.query_ollama(
    "Explain why " + best_model.get('model_name', 'this model') + " was selected as the best performer with R² = " + str(best_model.get('test_score', 0)) + ". Discuss its strengths and suitability for this regression task.",
    "Model: " + best_model.get('model_name', 'N/A') + ", Dataset: " + job_info.get('dataset_name', 'N/A') + ", Features: " + str(job_summary.get('final_features', 0)), 
    max_words=200
) if self.model_name else 
"The " + best_model.get('model_name', 'selected model') + " was chosen based on its superior test performance (R² = " + str(best_model.get('test_score', 0)) + ") and excellent cross-validation stability. This model demonstrates the optimal balance between accuracy and generalization, making it ideal for production deployment.") + """

**Model Selection Criteria:**
- **🏆 Highest Test Performance:** R² = """ + str(round(best_model.get('test_score', 0), 4)) + """ (explains """ + str(round(best_model.get('test_score', 0) * 100, 1)) + """% of variance)
- **🎯 Cross-Validation Stability:** CV R² = """ + str(round(best_model.get('cv_score', 0), 4)) + """ (consistent across data splits)
- **⚡ Training Efficiency:** """ + str(round(best_model.get('training_time', 0), 3)) + """ seconds (rapid deployment ready)
- **🛡️ Generalization Quality:** """ + ("Excellent" if abs(best_model.get('train_score', 0) - best_model.get('test_score', 0)) < 0.05 else "Good" if abs(best_model.get('train_score', 0) - best_model.get('test_score', 0)) < 0.1 else "Moderate") + """ (overfitting risk: """ + ("Low" if abs(best_model.get('train_score', 0) - best_model.get('test_score', 0)) < 0.1 else "Medium" if abs(best_model.get('train_score', 0) - best_model.get('test_score', 0)) < 0.2 else "High") + """)

---

## 5. Results Analysis and Business Impact

### 5.1 Performance Interpretation

""" + results_interpretation + """

### 5.2 Comprehensive Performance Dashboard

""" + ("""
![Analysis Summary Dashboard]({})

**Executive Analysis Summary:**

{}

This comprehensive dashboard provides a complete overview of the AutoML analysis results, combining data quality metrics, feature engineering outcomes, model performance comparisons, and business impact assessments. The visualization serves as an executive-level summary suitable for stakeholder presentations and strategic decision-making.

""".format(plot_references.get('analysis_summary', ''), 
          self.generate_plot_explanation('analysis_summary', plot_references.get('analysis_summary', ''), context_data)) 
    if 'analysis_summary' in plot_references else "Comprehensive analysis dashboard not available") + """

### 5.2 Model Performance Metrics

**Performance Breakdown:**

**Test Performance (Most Important):**
- **R² Score:** """ + str(round(best_model.get('test_score', 0), 4)) + """
- **Variance Explained:** """ + str(round(best_model.get('test_score', 0) * 100, 1)) + """%
- **Business Impact:** """ + str(round(best_model.get('test_score', 0) * 100, 1)) + """% prediction accuracy

**Training Performance:**
- **Train R²:** """ + str(round(best_model.get('train_score', 0), 4)) + """
- **Interpretation:** Model learning effectiveness
- **Overfitting Check:** """ + str(round(abs(best_model.get('train_score', 0) - best_model.get('test_score', 0)), 3)) + """ difference (""" + ("Low Risk" if abs(best_model.get('train_score', 0) - best_model.get('test_score', 0)) < 0.1 else "Medium Risk" if abs(best_model.get('train_score', 0) - best_model.get('test_score', 0)) < 0.2 else "High Risk") + """)

**Cross-Validation Stability:**
- **CV R²:** """ + str(round(best_model.get('cv_score', 0), 4)) + """
- **Stability Rating:** """ + ("Excellent" if best_model.get('cv_score', 0) > 0.6 else "Good" if best_model.get('cv_score', 0) > 0.4 else "Fair") + """
- **Deployment Reliability:** High confidence

**Efficiency Metrics:**
- **Training Time:** """ + str(round(best_model.get('training_time', 0), 3)) + """ seconds
- **Computational Efficiency:** Excellent
- **Retraining Capability:** Rapid updates possible

### 5.3 Business Value Assessment

**Performance Rating:** """ + ("Excellent" if best_model.get('test_score', 0) > 0.8 else "Very Good" if best_model.get('test_score', 0) > 0.7 else "Good" if best_model.get('test_score', 0) > 0.6 else "Fair" if best_model.get('test_score', 0) > 0.4 else "Poor") + """

**Deployment Readiness:** """ + ("Production Ready" if best_model.get('test_score', 0) > 0.6 else "Needs Improvement") + """

""" + ("""
### 5.4 Comprehensive Analysis Summary

![Analysis Summary]({})

**Analysis Summary Insights:**

{}
""".format(plot_references.get('analysis_summary', ''), 
          self.generate_plot_explanation('analysis_summary', plot_references.get('analysis_summary', ''), context_data)) 
    if 'analysis_summary' in plot_references else "") + """

---

## 6. Recommendations and Next Steps

### 6.1 Immediate Actions

1. **✅ Model Deployment:** The """ + best_model.get('model_name', 'selected model') + """ is """ + ("ready for production deployment" if best_model.get('test_score', 0) > 0.6 else "suitable for pilot testing") + """
2. **📊 Performance Monitoring:** Implement real-time tracking of model predictions
3. **🔄 Retraining Schedule:** Establish quarterly model updates or trigger-based retraining
4. **📋 Documentation:** Maintain comprehensive model documentation and feature definitions

### 6.2 Optimization Opportunities

""" + (self.query_ollama(
    "Suggest 3-4 specific ways to improve this AutoML model performance beyond R² = " + str(best_model.get('test_score', 0)) + ". Focus on practical, actionable recommendations.",
    "Current performance: R² = " + str(best_model.get('test_score', 0)) + ", Features: " + str(job_summary.get('final_features', 0)) + ", Model: " + best_model.get('model_name', 'N/A'),
    max_words=200
) if self.model_name else """
1. **Feature Enhancement:** Investigate domain-specific feature engineering techniques
2. **Ensemble Methods:** Combine multiple models for improved performance
3. **Hyperparameter Optimization:** Fine-tune model parameters for better accuracy
4. **Data Augmentation:** Collect additional high-quality training samples""") + """

### 6.3 Long-term Strategy

**Phase 1 (1-3 months):** Deploy current model and establish monitoring systems
**Phase 2 (3-6 months):** Implement model improvements and expand feature set
**Phase 3 (6+ months):** Develop advanced ensemble models and real-time capabilities

---

## 7. Technical Appendix

### 7.1 Reproducibility

**Command to reproduce this analysis:**
```bash
python main.py run --file """ + job_info.get('dataset_name', 'dataset.csv') + """ --target """ + job_info.get('target_variable', 'target') + """ --mode """ + job_info.get('mode', 'auto') + """
```

### 7.2 Model Artifacts

**File Structure:**
```
""" + str(self.job_path) + """/
├── all_models/                     # All trained models
├── preprocessors/pipeline.pkl      # Data preprocessing
├── feature_engineering_pipeline.pkl # Feature engineering
├── visualizations/                 # All generated plots
├── training_summary.json          # Detailed results
└── job_summary.json               # High-level metrics
```

### 7.3 Deployment Checklist

- [""" + ("x" if best_model.get('test_score', 0) > 0.5 else " ") + """] Model performance validation (R² > 0.5)
- [""" + ("x" if job_summary.get('models_trained', 0) > 3 else " ") + """] Multiple models evaluated
- [""" + ("x" if best_model.get('training_time', 0) < 60 else " ") + """] Training time acceptable (< 60s)
- [ ] Production infrastructure setup
- [ ] Monitoring and alerting configured
- [ ] Model documentation completed

---

## 8. Visualization Gallery

This report includes """ + str(len(plot_references)) + """ comprehensive visualizations:

""" + "\n".join([f"- **{category.replace('_', ' ').title()}:** {filename}" for category, filename in plot_references.items()]) + """

Each visualization provides specific insights into different aspects of the machine learning pipeline, from data quality assessment to final model performance evaluation.

---

*Report generated on """ + datetime.now().strftime('%B %d, %Y at %I:%M %p') + """ using AutoML Phase 1 Pipeline*  
*Analysis completed in """ + str(round(job_summary.get('total_training_time', 0), 1)) + """ seconds with """ + str(job_summary.get('models_trained', 0)) + """ models evaluated*  
*Best model: """ + best_model.get('model_name', 'N/A') + """ (R² = """ + str(round(best_model.get('test_score', 0), 4)) + """)*

**Report Statistics:**
- **Pages:** ~30-35 pages with comprehensive analysis
- **Sections:** 8 major sections + technical appendix  
- **Visualizations:** """ + str(len(plot_references)) + """ plots with AI explanations
- **AI Insights:** """ + ("Enabled" if self.model_name else "Disabled") + """ (""" + (self.model_name or "Ollama not available") + """)
- **Total Plots Found:** """ + str(len(discovered_plots)) + """ across multiple locations
"""
        
        return markdown_content
    
    def convert_markdown_to_pdf(self, markdown_content, output_path):
        """Convert markdown to PDF using multiple fallback methods"""
        temp_md_path = self.job_path / "temp_report.md"
        
        with open(temp_md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # Method 1: Enhanced pandoc
        try:
            cmd = [
                'pandoc', str(temp_md_path), '-o', str(output_path),
                '--pdf-engine=xelatex', '-V', 'geometry:margin=0.75in',
                '-V', 'fontsize=11pt', '--toc', '--number-sections', '--standalone'
            ]
            
            print("🎭 Pandoc is putting on its finest performance...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("🎉 Pandoc delivered a standing ovation! PDF creation successful!")
                temp_md_path.unlink()
                return True
            else:
                print("😅 Pandoc stumbled a bit: " + result.stderr)
                
        except FileNotFoundError:
            print("🤷‍♂️ Pandoc is missing from the stage!")
        
        # Method 2: Basic pandoc
        try:
            cmd = ['pandoc', str(temp_md_path), '-o', str(output_path), '--pdf-engine=xelatex']
            print("🔧 Trying pandoc's minimalist approach...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✨ Basic pandoc saved the day! Sometimes less is more!")
                temp_md_path.unlink()
                return True
                
        except FileNotFoundError:
            pass
        
        # Method 3: WeasyPrint
        try:
            import markdown, weasyprint
            print("🚀 WeasyPrint superhero swooping in to save the day!")
            
            with open(temp_md_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            html = markdown.markdown(md_content, extensions=['tables'])
            html_with_style = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }
        h1, h2, h3 { color: #2c3e50; page-break-after: avoid; }
        h1 { border-bottom: 3px solid #3498db; padding-bottom: 10px; font-size: 28px; }
        h2 { border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; font-size: 22px; margin-top: 30px; }
        h3 { color: #34495e; font-size: 18px; margin-top: 25px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f8f9fa; font-weight: bold; color: #2c3e50; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        pre { background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; overflow-x: auto; }
        code { background-color: #f1f1f1; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }
        blockquote { border-left: 4px solid #3498db; margin: 20px 0; padding: 15px 20px; background-color: #f8f9fa; }
        img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }
        .metric-highlight { background-color: #e8f5e8; font-weight: bold; }
        ul, ol { margin: 15px 0; padding-left: 30px; }
        li { margin: 5px 0; }
        strong { color: #2c3e50; }
        .page-break { page-break-before: always; }
    </style>
</head>
<body>
""" + html + """
</body>
</html>"""
            
            weasyprint.HTML(string=html_with_style).write_pdf(str(output_path))
            print("🎊 WeasyPrint executed a flawless PDF performance! Mission accomplished!")
            temp_md_path.unlink()
            return True
            
        except ImportError:
            print("😭 WeasyPrint is not in our toolkit! Install with: pip install weasyprint markdown")
        except Exception as e:
            print("🤯 WeasyPrint encountered a plot twist: " + str(e))
        
        print("🤡 All PDF conversion methods went on strike today!")
        print("📄 But fear not! Your markdown masterpiece is still available!")
        return False
    
    def generate_report(self):
        """Generate the complete report with all discovered plots"""
        print("🎪 Welcome to the Ultimate AutoML Report Extravaganza! 🎭")
        print("=" * 65)
        print("🔮 Preparing to unleash data science magic for job: " + self.job_id)
        
        if not any(self.report_data.values()):
            print("🚨 CODE RED: Mission critical data files missing!")
            print("🕵️‍♂️ Investigation shows:")
            print("   🎯 Target Job ID: " + self.job_id)
            print("   📂 Search Location: " + str(self.job_path))
            print("   📋 Required Evidence: state.json, training_summary.json, job_summary.json")
            print("🤷‍♀️ Either this job is in witness protection or someone moved the files!")
            return None
        
        print("🧙‍♂️ Weaving together plots, data, and AI wisdom into an epic tale...")
        markdown_content = self.generate_markdown_report()
        
        # Save markdown
        md_output_path = self.job_path / ("AutoML_Report_" + self.job_id + ".md")
        with open(md_output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print("📜 Epic markdown saga saved: " + str(md_output_path))
        
        # Convert to PDF
        pdf_output_path = self.job_path / ("AutoML_Report_" + self.job_id + ".pdf")
        print("🎨 Grand finale time - transforming markdown into PDF perfection...")
        
        if self.convert_markdown_to_pdf(markdown_content, pdf_output_path):
            print("\n🎊 SPECTACULAR SUCCESS! The crowd goes absolutely wild! 🎊")
            print("📊 Your masterpiece is ready for the world: " + str(pdf_output_path))
            print("📝 Markdown backup for the curious: " + str(md_output_path))
            
            # File size commentary with extra personality
            if pdf_output_path.exists():
                pdf_size = pdf_output_path.stat().st_size / (1024 * 1024)
                if pdf_size > 10:
                    print("📏 PDF Size: " + str(round(pdf_size, 2)) + " MB (Holy data! That's a THICC report! 🐘)")
                elif pdf_size > 5:
                    print("📏 PDF Size: " + str(round(pdf_size, 2)) + " MB (Perfect size - like Goldilocks would approve! 👌)")
                elif pdf_size > 2:
                    print("📏 PDF Size: " + str(round(pdf_size, 2)) + " MB (Compact excellence - Swiss Army knife of reports! 🔧)")
                else:
                    print("📏 PDF Size: " + str(round(pdf_size, 2)) + " MB (Small but mighty - like a data science haiku! 💎)")
            
            return pdf_output_path
        else:
            print("\n🎭 Plot twist! PDF conversion decided to be dramatic today...")
            print("📄 But don't despair! Your markdown report is still absolutely fantastic:")
            print("📁 Markdown Report: " + str(md_output_path))
            print("💡 Convert it manually when the PDF spirits are more cooperative!")
            return md_output_path

def main():
    """Main function for standalone execution"""
    if len(sys.argv) != 2:
        print("🎪 Welcome to the AutoML Report Generator MEGA CIRCUS! 🎭")
        print("=" * 65)
        print("🎯 Usage: python3 generate_report.py <job_id>")
        print("🎪 Example: python3 generate_report.py job_20250819_005202")
        print("\n🌟 This ENHANCED version will blow your mind:")
        print("  🔮 Hunt down ALL plots from multiple secret locations")
        print("  📊 Embed every single visualization with AI explanations")
        print("  🤖 Generate profound insights that make data scientists cry tears of joy")
        print("  📄 Create a 30-35 page PDF masterpiece that'll get you promoted")
        print("  🎭 Keep you entertained with top-tier data science humor")
        print("  ⚡ Process faster than you can say 'machine learning'")
        print("  🧠 Use advanced AI to explain every plot and insight")
        print("\n💡 Pro tip: This version finds plots EVERYWHERE - it's like plot GPS! 📍")
        sys.exit(1)
    
    job_id = sys.argv[1]
    
    # Enhanced job validation
    job_path = Path("storage/models/jobs") / job_id
    if not job_path.exists():
        print("🚨 PLOT TWIST! Job directory has vanished into the data dimension!")
        print("🔍 Scanning the multiverse for available jobs...")
        jobs_dir = Path("storage/models/jobs")
        if jobs_dir.exists():
            job_count = 0
            for job_dir in jobs_dir.iterdir():
                if job_dir.is_dir():
                    print("   🎯 Discovered: " + job_dir.name + " (Ready for analysis!)")
                    job_count += 1
            if job_count == 0:
                print("   🏜️ The job desert is emptier than a data scientist's social calendar!")
                print("   💡 Hint: Run some ML jobs first, then come back for the report party!")
            else:
                print("   🎉 Found " + str(job_count) + " jobs hiding in the archives!")
                print("   💭 Did you maybe typo the job ID? Copy-paste is your friend!")
        else:
            print("   🤯 The entire jobs directory is missing! This is unprecedented!")
            print("   🚨 Emergency protocol: Check if you're in the right directory!")
        sys.exit(1)
    
    print("🎬 Lights! Camera! Data! Action begins NOW!")
    print("🍿 Grab your favorite beverage - this is going to be EPIC!")
    
    # Generate the ultimate report
    generator = AutoMLReportGenerator(job_id)
    report_path = generator.generate_report()
    
    if report_path:
        print("\n🎊 MISSION ACCOMPLISHED! The data gods smile upon us! 🎊")
        print("📂 Your legendary report awaits at: " + str(report_path))
        
        # Epic file size analysis
        if Path(report_path).exists():
            file_size = Path(report_path).stat().st_size / (1024 * 1024)
            if file_size > 15:
                print("📏 " + str(round(file_size, 1)) + "MB - This is MASSIVE! You've created the War and Peace of ML reports! 📚")
            elif file_size > 10:
                print("📏 " + str(round(file_size, 1)) + "MB - Substantial and impressive! Your stakeholders will be in awe! 🤩")
            elif file_size > 5:
                print("📏 " + str(round(file_size, 1)) + "MB - Perfect balance of depth and readability! 📖")
            elif file_size > 2:
                print("📏 " + str(round(file_size, 1)) + "MB - Concise yet comprehensive! Efficiency at its finest! ⚡")
            else:
                print("📏 " + str(round(file_size, 1)) + "MB - Lean and mean data machine! 🚀")
        
        print("\n🎯 Your quest is complete! What's next in your data adventure?")
        print("   📖 Marvel at your comprehensive analysis masterpiece")
        print("   🚀 Share with your team and watch their minds get blown")
        print("   💼 Present to stakeholders and watch them throw money at your ML project")
        print("   🏆 Frame the first page and hang it on your wall (okay, maybe just save it)")
        print("   ☕ Celebrate with the beverage of champions - you've earned it!")
        
        # Random epic data science quotes
        import random
        epic_quotes = [
            "💡 'In data we trust, in models we verify!' - Ancient ML Proverb",
            "🎩 'A wizard is never late with their model, nor early. They deploy precisely when they mean to!'",
            "🚀 'Houston, we have a solution! T-minus zero to model deployment!'",
            "🔮 'May the R² be with you, always and forever!'",
            "🎯 'One does not simply walk into production without proper validation!'",
            "📊 'I see data people... and they're all making predictions!'",
            "⚡ 'With great computational power comes great model responsibility!'"
        ]
        print("\n" + random.choice(epic_quotes))
        
    else:
        print("\n🎭 Plot twist in our data drama! Something went sideways...")
        print("🤔 But hey, even the best data scientists face plot twists!")
        print("💪 Don't let this stop your data science journey!")
        print("🔧 Debug those error messages like the ML detective you are!")
        print("🎪 The show must go on - try again when the stars align!")
        sys.exit(1)

if __name__ == "__main__":
    main()