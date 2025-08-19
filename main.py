"""
main.py - Updated with complete integration
Main CLI entry point for Phase 1 AutoML Pipeline
"""

import argparse
import sys
import os
import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our components
from core.data_validator import DataValidator
from core.preprocessing import PreprocessingPipeline
from core.feature_engineering import FeatureEngineer
from core.model_selector import ModelSelector
from core.model_trainer import ModelTrainer
from core.inference import InferenceEngine
from utils.progress_tracker import ProgressTracker
from utils.state_manager import StateManager
from utils.file_handler import FileHandler
from utils.visualization import Visualizer

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # Create logs directory
    log_dir = log_config.get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'automl.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = os.path.join('config', 'default_config.yaml')
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🤖 AutoML Phase 1                         ║
║              All-Numeric Regression Pipeline                 ║
║                                                              ║
║  🎯 Smart Model Selection                                     ║
║  ⚙️  Automated Feature Engineering                            ║
║  📊 Comprehensive Preprocessing                               ║
║  🚀 Real-time Progress Tracking                               ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def validate_file_path(file_path: str) -> str:
    """Validate and return absolute file path"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    return os.path.abspath(file_path)

def run_pipeline(args):
    """Run the main AutoML pipeline"""
    print_banner()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Starting AutoML Pipeline - Phase 1")
    
    # Validate file path
    file_path = validate_file_path(args.file)
    dataset_name = os.path.basename(file_path)
    
    try:
        # Initialize components
        data_validator = DataValidator(config['pipeline'])
        state_manager = StateManager(config['pipeline'])
        file_handler = FileHandler(config['pipeline'])
        visualizer = Visualizer(config['pipeline'])
        
        # Interactive target selection if not provided
        if not args.target and not getattr(args, 'batch', False):
            target_variable = interactive_target_selection(file_path, data_validator, args.mode)
        elif not args.target and getattr(args, 'batch', False):
            print("❌ Target variable required in batch mode. Use --target flag.")
            sys.exit(1)
        else:
            target_variable = validate_target_variable(file_path, args.target, data_validator)
        
        # Create new job
        job_id = state_manager.create_job(dataset_name, target_variable, args.mode)
        
        # Initialize progress tracker
        progress = ProgressTracker(
            job_id=job_id,
            config=config['pipeline'],
            console_output=True,
            log_to_file=True
        )
        
        print(f"\n🆔 Job ID: {job_id}")
        print(f"📂 Dataset: {dataset_name}")
        print(f"🎯 Target: {target_variable}")
        print(f"⚙️  Mode: {args.mode}")
        print()
        
        # Start pipeline execution
        state_manager.update_job_status(job_id, "running")
        progress.start_step("data_validation", "Validating and profiling dataset")
        
        # Step 1: Data Validation & Profiling
        progress.start_sub_step("data_validation", "file_validation", "Validating file format and size")
        validation_result = data_validator.validate_file(file_path)
        progress.complete_sub_step("data_validation", "file_validation")
        
        if not validation_result.is_valid:
            print("❌ Data validation failed:")
            for error in validation_result.errors:
                print(f"   • {error}")
            state_manager.update_job_status(job_id, "failed")
            return
        
        # Display warnings and suggestions
        if validation_result.warnings:
            print("⚠️  Warnings:")
            for warning in validation_result.warnings:
                print(f"   • {warning}")
        
        if validation_result.suggestions:
            print("💡 Suggestions:")
            for suggestion in validation_result.suggestions:
                print(f"   • {suggestion}")
        print()
        
        # Load and validate dataset
        progress.start_sub_step("data_validation", "data_loading", "Loading dataset")
        df = file_handler.load_data(file_path)
        progress.complete_sub_step("data_validation", "data_loading")
        
        # Update state with dataset profile
        progress.start_sub_step("data_validation", "profiling", "Creating dataset profile")
        state_manager.update_dataset_profile(job_id, validation_result.dataset_profile)
        progress.complete_sub_step("data_validation", "profiling")
        
        progress.complete_step("data_validation")
        
        # Step 2: Preprocessing Pipeline
        progress.start_step("preprocessing", "Applying data preprocessing")
        
        preprocessor = PreprocessingPipeline(config['pipeline'], job_id, progress, state_manager)
        preprocessing_result = preprocessor.run_preprocessing(df, target_variable)
        
        # Save preprocessing pipeline
        preprocessor_path = state_manager.get_preprocessor_path(job_id)
        preprocessor.save_preprocessing_pipeline(preprocessor_path, preprocessing_result)
        
        progress.complete_step("preprocessing")
        
        # Step 3: Feature Engineering
        progress.start_step("feature_engineering", "Automated feature engineering")
        
        feature_engineer = FeatureEngineer(config['pipeline'], job_id, progress, state_manager)
        fe_result = feature_engineer.run_feature_engineering(preprocessing_result.processed_data, target_variable)
        
        # Save feature engineering pipeline
        fe_path = os.path.join(state_manager.get_job_path(job_id), 'feature_engineering_pipeline.pkl')
        feature_engineer.save_feature_engineering_pipeline(fe_path, fe_result)
        
        progress.complete_step("feature_engineering")
        
        # Step 4: Model Selection & Training
        progress.start_step("model_selection", "Smart model selection and training")
        
        # Model selection
        progress.start_sub_step("model_selection", "dataset_analysis", "Analyzing dataset characteristics")
        model_selector = ModelSelector(config['pipeline'], job_id, progress, state_manager)
        characteristics, suggestions = model_selector.analyze_dataset_and_suggest_models(
            fe_result.engineered_data, target_variable
        )
        progress.complete_sub_step("model_selection", "dataset_analysis")
        
        progress.start_sub_step("model_selection", "model_suggestions", "Generating model suggestions")
        suggested_model_names = model_selector.get_suggested_models(suggestions)
        
        # Display suggestions
        print("\n🎯 MODEL RECOMMENDATIONS:")
        print("=" * 60)
        for i, suggestion in enumerate(suggestions, 1):
            confidence_icon = "🔥" if suggestion.confidence_score > 0.8 else "⭐" if suggestion.confidence_score > 0.6 else "💡"
            print(f"{confidence_icon} {i}. {suggestion.model_name}")
            print(f"   Confidence: {suggestion.confidence_score:.1%}")
            print(f"   Reasoning: {suggestion.reasoning}")
            print(f"   Expected Performance: {suggestion.expected_performance}")
            print(f"   Training Time: {suggestion.training_time}")
            print()
        
        progress.complete_sub_step("model_selection", "model_suggestions")
        
        # Model training
        progress.start_sub_step("model_selection", "model_training", "Training selected models")
        model_trainer = ModelTrainer(config['pipeline'], job_id, progress, state_manager)
        
        # Determine training strategy based on mode
        if args.mode == 'auto':
            # Train only suggested models
            training_summary = model_trainer.train_all_models(
                fe_result.engineered_data, target_variable, 
                suggested_model_names, train_all=False
            )
            print(f"\n🚀 Trained {len(suggested_model_names)} suggested models")
            
        elif args.mode == 'comprehensive':
            # Train all available models
            training_summary = model_trainer.train_all_models(
                fe_result.engineered_data, target_variable, 
                suggested_model_names, train_all=True
            )
            print(f"\n🚀 Trained all available models")
            
        elif args.mode == 'quick':
            # Train only top 3 suggested models
            top_3_models = suggested_model_names[:3] if len(suggested_model_names) >= 3 else suggested_model_names
            training_summary = model_trainer.train_all_models(
                fe_result.engineered_data, target_variable, 
                top_3_models, train_all=False
            )
            print(f"\n🚀 Trained top {len(top_3_models)} models in quick mode")
        
        progress.complete_sub_step("model_selection", "model_training")
        progress.complete_step("model_selection")
        
        # Step 5: Evaluation & Results
        progress.start_step("evaluation", "Model evaluation and results")
        
        progress.start_sub_step("evaluation", "performance_evaluation", "Evaluating model performance")
        
        # Get model comparison data
        comparison_df = model_trainer.get_model_comparison_data()
        
        # Display results
        print("\n📊 MODEL PERFORMANCE RESULTS:")
        print("=" * 80)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Highlight best model
        best_model = training_summary.get('best_model', {})
        if best_model:
            print(f"\n🏆 BEST MODEL: {best_model['name']}")
            print(f"   Test Score (R²): {best_model['test_score']:.4f}")
            print(f"   Train Score (R²): {best_model['train_score']:.4f}")
            print(f"   CV Score (R²): {best_model['cv_score']:.4f}")
            print(f"   Training Time: {best_model['training_time']:.1f}s")
        
        progress.complete_sub_step("evaluation", "performance_evaluation")
        
        # Generate visualizations
        progress.start_sub_step("evaluation", "model_comparison", "Creating comparison visualizations")
        
        # Create visualization directory
        viz_dir = os.path.join(state_manager.get_job_path(job_id), 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate plots
        try:
            # Model comparison plot
            comp_plot_path = os.path.join(viz_dir, 'model_comparison.png')
            visualizer.plot_model_comparison(comparison_df, comp_plot_path, 
                                           f"Model Comparison - {job_id}")
            
            # Top models detailed plot
            detailed_plot_path = os.path.join(viz_dir, 'top_models_detailed.png')
            visualizer.plot_top_models_detailed(comparison_df, top_n=5, save_path=detailed_plot_path)
            
            # Performance summary
            summary_plot_path = os.path.join(viz_dir, 'performance_summary.png')
            visualizer.plot_performance_summary(comparison_df, best_model, summary_plot_path)
            
            # Results table
            table_path = os.path.join(viz_dir, 'results_table.png')
            visualizer.save_results_table(comparison_df, table_path)
            
            print(f"\n📈 Visualizations saved to: {viz_dir}")
            
        except Exception as e:
            logger.warning(f"Visualization generation failed: {str(e)}")
        
        progress.complete_sub_step("evaluation", "model_comparison")
        
        # Save final results
        progress.start_sub_step("evaluation", "result_saving", "Saving models and results")
        
        # Save training summary
        state_manager.save_training_summary(job_id, training_summary)
        
        # Create job summary
        job_summary = {
            'job_id': job_id,
            'dataset_name': dataset_name,
            'target_variable': target_variable,
            'mode': args.mode,
            'original_features': len(df.columns) - 1,
            'final_features': len(fe_result.feature_names),
            'outliers_removed': preprocessing_result.outliers_removed,
            'features_removed': len(preprocessing_result.features_removed),
            'models_trained': len(comparison_df),
            'best_model': best_model['name'] if best_model else None,
            'best_score': best_model['test_score'] if best_model else None,
            'total_training_time': comparison_df['Training Time'].sum(),
            'visualizations_created': True
        }
        
        # Save summary
        summary_path = os.path.join(state_manager.get_job_path(job_id), 'job_summary.json')
        file_handler.save_json(job_summary, summary_path)
        
        progress.complete_sub_step("evaluation", "result_saving")
        progress.complete_step("evaluation")
        
        # Complete pipeline
        progress.complete_pipeline()
        state_manager.update_job_status(job_id, "completed")
        
        # Final summary
        print("\n" + "=" * 80)
        print("✨ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📁 Results saved in: {state_manager.get_job_path(job_id)}")
        print(f"🎯 Best Model: {best_model['name']} (R² = {best_model['test_score']:.4f})")
        print(f"⏱️  Total Time: {comparison_df['Training Time'].sum():.1f}s")
        print(f"📊 Models Trained: {len(comparison_df)}")
        print(f"🔍 Features: {len(df.columns)-1} → {len(fe_result.feature_names)}")
        print()
        print("💡 Next Steps:")
        print(f"   • Use 'python main.py predict --model {job_id} --file new_data.csv' to make predictions")
        print(f"   • Check visualizations in: {viz_dir}")
        print(f"   • View detailed results: python main.py status --job {job_id}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.debug(traceback.format_exc())
        print(f"❌ Pipeline failed: {str(e)}")
        if 'job_id' in locals():
            state_manager.update_job_status(job_id, "failed")
        sys.exit(1)

def interactive_target_selection(file_path: str, validator: DataValidator, mode: str) -> str:
    """Interactive target variable selection with smart recommendations"""
    print("🔍 Analyzing dataset for target selection...")
    
    try:
        # Load dataframe for analysis
        df = validator._load_dataframe(file_path)
        
        print(f"\n📊 Dataset Overview: {os.path.basename(file_path)}")
        print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Analyze columns and categorize
        column_analysis = analyze_columns_for_target_selection(df)
        
        # Display columns with analysis
        print("\n📋 Available Columns:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        for i, (col_name, analysis) in enumerate(column_analysis.items(), 1):
            recommendation = analysis['recommendation']
            dtype = analysis['dtype']
            stats = analysis['stats_summary']
            icon = analysis['icon']
            
            print(f"[{i:2d}] {col_name:<20} ({dtype:<10}) {icon} {stats}")
            if analysis['issues']:
                print(f"     ⚠️  {analysis['issues']}")
        
        # Show recommendations
        print("\n🎯 RECOMMENDATIONS:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        highly_recommended = [i for i, (_, analysis) in enumerate(column_analysis.items(), 1) 
                             if analysis['recommendation'] == 'highly_recommended']
        possible = [i for i, (_, analysis) in enumerate(column_analysis.items(), 1) 
                   if analysis['recommendation'] == 'possible']
        not_recommended = [i for i, (_, analysis) in enumerate(column_analysis.items(), 1) 
                          if analysis['recommendation'] == 'not_recommended']
        
        if highly_recommended:
            print(f"⭐ HIGHLY RECOMMENDED: {', '.join(map(str, highly_recommended))}")
        if possible:
            print(f"⚠️  POSSIBLE BUT CHALLENGING: {', '.join(map(str, possible))}")
        if not_recommended:
            print(f"❌ NOT RECOMMENDED: {', '.join(map(str, not_recommended))}")
        
        # Get user selection
        while True:
            try:
                print(f"\n🎯 Select target variable [1-{len(column_analysis)}] (or 'q' to quit): ", end="")
                user_input = input().strip()
                
                if user_input.lower() == 'q':
                    print("❌ Exiting...")
                    sys.exit(0)
                
                selection = int(user_input)
                if 1 <= selection <= len(column_analysis):
                    selected_column = list(column_analysis.keys())[selection - 1]
                    selected_analysis = column_analysis[selected_column]
                    
                    # Show confirmation with warnings
                    print(f"\n✅ Selected: {selected_column}")
                    print(f"   Type: {selected_analysis['dtype']}")
                    print(f"   {selected_analysis['stats_summary']}")
                    
                    if selected_analysis['recommendation'] != 'highly_recommended':
                        print(f"\n⚠️  WARNING: This column is {selected_analysis['recommendation']}")
                        if selected_analysis['issues']:
                            print(f"   Issues: {selected_analysis['issues']}")
                        
                        confirm = input("\nContinue anyway? [y/N]: ").strip().lower()
                        if confirm != 'y':
                            continue
                    
                    return selected_column
                else:
                    print(f"❌ Please enter a number between 1 and {len(column_analysis)}")
                    
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n❌ Interrupted by user")
                sys.exit(0)
    
    except Exception as e:
        print(f"❌ Error analyzing dataset: {str(e)}")
        print("Please provide target variable manually using --target flag")
        sys.exit(1)

def analyze_columns_for_target_selection(df: pd.DataFrame) -> Dict[str, Dict]:
    """Analyze columns and provide recommendations for target selection"""
    analysis = {}
    
    for col in df.columns:
        col_analysis = {
            'dtype': str(df[col].dtype),
            'recommendation': 'not_recommended',
            'icon': '❌',
            'stats_summary': '',
            'issues': ''
        }
        
        # Get basic stats
        nunique = df[col].nunique()
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        
        # Analyze based on data type
        if pd.api.types.is_numeric_dtype(df[col]):
            # Numeric columns
            if nunique == 1:
                col_analysis['recommendation'] = 'not_recommended'
                col_analysis['icon'] = '❌'
                col_analysis['stats_summary'] = 'Constant value'
                col_analysis['issues'] = 'No variance - cannot be predicted'
                
            elif nunique == len(df) or nunique > len(df) * 0.95:
                col_analysis['recommendation'] = 'not_recommended'
                col_analysis['icon'] = '❌'
                col_analysis['stats_summary'] = f'{nunique:,} unique values (likely ID)'
                col_analysis['issues'] = 'Too many unique values - likely an identifier'
                
            elif missing_pct > 50:
                col_analysis['recommendation'] = 'not_recommended'
                col_analysis['icon'] = '❌'
                col_analysis['stats_summary'] = f'{missing_pct:.1f}% missing'
                col_analysis['issues'] = 'Too many missing values'
                
            elif df[col].dtype in ['int64', 'float64']:
                # Good numeric target
                min_val, max_val = df[col].min(), df[col].max()
                
                if missing_pct > 20:
                    col_analysis['recommendation'] = 'possible'
                    col_analysis['icon'] = '⚠️'
                    col_analysis['issues'] = f'{missing_pct:.1f}% missing values'
                else:
                    col_analysis['recommendation'] = 'highly_recommended'
                    col_analysis['icon'] = '⭐'
                
                if abs(max_val - min_val) < 1e-10:
                    col_analysis['stats_summary'] = f'Range: {min_val:.2f} (constant)'
                    col_analysis['recommendation'] = 'not_recommended'
                    col_analysis['icon'] = '❌'
                else:
                    col_analysis['stats_summary'] = f'Range: {min_val:.2f} to {max_val:.2f}'
            
            else:
                # Other numeric types
                col_analysis['recommendation'] = 'possible'
                col_analysis['icon'] = '⚠️'
                col_analysis['stats_summary'] = f'{nunique:,} unique values'
        
        else:
            # Non-numeric columns
            col_analysis['recommendation'] = 'not_recommended'
            col_analysis['icon'] = '❌'
            
            if df[col].dtype == 'object':
                col_analysis['stats_summary'] = f'{nunique} categories'
                col_analysis['issues'] = 'Text/categorical - use Phase 2 for mixed data'
            elif df[col].dtype == 'datetime64[ns]':
                col_analysis['stats_summary'] = 'Date/time column'
                col_analysis['issues'] = 'DateTime - needs feature engineering'
            else:
                col_analysis['stats_summary'] = f'{nunique} unique values'
                col_analysis['issues'] = 'Non-numeric type'
        
        analysis[col] = col_analysis
    
    return analysis

def validate_target_variable(file_path: str, target: str, validator: DataValidator) -> str:
    """Validate manually provided target variable"""
    try:
        df = validator._load_dataframe(file_path)
        
        if target not in df.columns:
            print(f"❌ Target column '{target}' not found in dataset")
            print(f"Available columns: {', '.join(df.columns.tolist())}")
            sys.exit(1)
        
        # Quick validation
        if not pd.api.types.is_numeric_dtype(df[target]):
            print(f"⚠️  Warning: '{target}' is not numeric ({df[target].dtype})")
            print("Phase 1 is optimized for numeric targets")
            
            confirm = input("Continue anyway? [y/N]: ").strip().lower()
            if confirm != 'y':
                sys.exit(0)
        
        missing_pct = (df[target].isnull().sum() / len(df)) * 100
        if missing_pct > 50:
            print(f"⚠️  Warning: '{target}' has {missing_pct:.1f}% missing values")
            confirm = input("Continue anyway? [y/N]: ").strip().lower()
            if confirm != 'y':
                sys.exit(0)
        
        print(f"✅ Target variable '{target}' validated")
        return target
        
    except Exception as e:
        print(f"❌ Error validating target: {str(e)}")
        sys.exit(1)

def predict_with_model(args):
    """Make predictions using a trained model"""
    print("🔮 Making Predictions...")
    
    config = load_config(args.config)
    state_manager = StateManager(config['pipeline'])
    
    # Validate inputs
    if not args.model:
        print("❌ Model ID is required for prediction")
        sys.exit(1)
    
    file_path = validate_file_path(args.file)
    
    try:
        # Check if job exists
        job_state = state_manager.get_job_state(args.model)
        if not job_state:
            print(f"❌ Model not found: {args.model}")
            sys.exit(1)
        
        best_model = job_state.get('model_results', {}).get('best_model', {})
        if not best_model:
            print(f"❌ No trained model found for job {args.model}")
            sys.exit(1)
        
        print(f"📂 Loading model: {args.model}")
        print(f"🎯 Model: {best_model.get('model_name', 'Unknown')}")
        print(f"📊 Dataset: {os.path.basename(file_path)}")
        
        # Initialize inference engine
        inference_engine = InferenceEngine(config['pipeline'], state_manager)
        
        # Make predictions
        output_file = args.output or f"predictions_{args.model}.csv"
        result = inference_engine.predict(args.model, file_path, output_file)
        
        print(f"✅ Predictions completed!")
        print(f"📊 Predictions saved to: {output_file}")
        print(f"🔢 Generated {len(result.predictions)} predictions")
        print(f"🤖 Model used: {result.model_used}")
        print(f"⚙️  Preprocessing applied: {result.preprocessing_applied}")
        print(f"🔧 Feature engineering applied: {result.feature_engineering_applied}")
        
        # Show prediction summary
        predictions = result.predictions
        print(f"\n📈 Prediction Summary:")
        print(f"   Mean: {predictions.mean():.4f}")
        print(f"   Std:  {predictions.std():.4f}")
        print(f"   Min:  {predictions.min():.4f}")
        print(f"   Max:  {predictions.max():.4f}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        sys.exit(1)

def show_job_status(args):
    """Show job status and progress"""
    config = load_config(args.config)
    state_manager = StateManager(config['pipeline'])
    
    if args.job:
        # Show specific job status
        job_state = state_manager.get_job_state(args.job)
        if not job_state:
            print(f"❌ Job not found: {args.job}")
            sys.exit(1)
        
        job_info = job_state['job_info']
        
        print(f"📋 Job Status: {args.job}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📊 Dataset: {job_info['dataset_name']}")
        print(f"🎯 Target: {job_info['target_variable']}")
        print(f"⚙️  Mode: {job_info['mode']}")
        print(f"📅 Created: {job_info['timestamp']}")
        print(f"🔄 Status: {job_info['status']}")
        
        # Show best model if available
        best_model = job_state.get('model_results', {}).get('best_model', {})
        if best_model:
            print(f"\n🏆 Best Model: {best_model['model_name']}")
            print(f"📊 Test Score: {best_model.get('test_score', 'N/A')}")
            print(f"⏱️  Training Time: {best_model.get('training_time', 'N/A')}s")
        
        # Show file locations
        job_path = state_manager.get_job_path(args.job)
        print(f"\n📁 Files:")
        print(f"   Job Directory: {job_path}")
        if os.path.exists(os.path.join(job_path, 'visualizations')):
            print(f"   Visualizations: {os.path.join(job_path, 'visualizations')}")
        
    else:
        # Show all recent jobs
        jobs = state_manager.list_jobs(limit=20)
        
        if not jobs:
            print("📭 No jobs found")
            return
        
        print("📋 Recent Jobs:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"{'Job ID':<25} {'Dataset':<20} {'Target':<15} {'Status':<10} {'Created':<20}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        for job in jobs:
            job_id = job['job_id']
            dataset = job['dataset_name'][:19] if len(job['dataset_name']) > 19 else job['dataset_name']
            target = job['target_variable'][:14] if len(job['target_variable']) > 14 else job['target_variable']
            status = job['status']
            created = job['timestamp'].split('T')[0]  # Just the date
            
            print(f"{job_id:<25} {dataset:<20} {target:<15} {status:<10} {created:<20}")

def compare_models(args):
    """Compare models from a job"""
    config = load_config(args.config)
    state_manager = StateManager(config['pipeline'])
    
    job_state = state_manager.get_job_state(args.job)
    if not job_state:
        print(f"❌ Job not found: {args.job}")
        sys.exit(1)
    
    model_results = job_state.get('model_results', {})
    all_models = model_results.get('all_models_performance', {})
    
    if not all_models:
        print(f"❌ No model results found for job {args.job}")
        return
    
    print(f"📊 Model Comparison: {args.job}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Convert to DataFrame for better display
    results_data = []
    for model_name, results in all_models.items():
        results_data.append({
            'Model': model_name,
            'Train Score': results.get('train_score', 0),
            'Test Score': results.get('test_score', 0),
            'CV Score': results.get('cv_score', 0),
            'Training Time': results.get('training_time', 0)
        })
    
    df = pd.DataFrame(results_data).sort_values('Test Score', ascending=False)
    print(df.to_string(index=False, float_format='%.4f'))
    
    # Show best model
    best_model = model_results.get('best_model', {})
    if best_model:
        print(f"\n🏆 Best Model: {best_model['model_name']}")
        print(f"   📊 Test Score: {best_model.get('test_score', 'N/A'):.4f}")
        print(f"   🎯 From: {'Suggestions' if best_model.get('is_suggested') else 'All Models'}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="AutoML Phase 1 - All-Numeric Regression Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic auto mode
  python main.py run --file data.csv --target price
  
  # Custom configuration
  python main.py run --file data.csv --target price --config my_config.yaml
  
  # Run all models (comprehensive mode)
  python main.py run --file data.csv --target price --mode comprehensive
  
  # Make predictions
  python main.py predict --model job_20250117_143052 --file new_data.csv
  
  # Check job status
  python main.py status --job job_20250117_143052
  
  # Compare models from a job
  python main.py compare --job job_20250117_143052
        """
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run pipeline command
    run_parser = subparsers.add_parser('run', help='Run the AutoML pipeline')
    run_parser.add_argument('--file', '-f', required=True, help='Input dataset file')
    run_parser.add_argument('--target', '-t', help='Target variable name (optional - will prompt if not provided)')
    run_parser.add_argument('--mode', '-m', choices=['auto', 'comprehensive', 'quick'], default='auto',
                           help='Processing mode: auto (suggested models), comprehensive (all models), quick (top 3)')
    run_parser.add_argument('--config', '-c', help='Custom configuration file')
    run_parser.add_argument('--batch', action='store_true', help='Batch mode - no interactive prompts')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with trained model')
    predict_parser.add_argument('--model', required=True, help='Model job ID')
    predict_parser.add_argument('--file', '-f', required=True, help='Data file for prediction')
    predict_parser.add_argument('--output', '-o', help='Output file for predictions')
    predict_parser.add_argument('--config', '-c', help='Configuration file')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show job status')
    status_parser.add_argument('--job', help='Specific job ID (if not provided, shows all recent jobs)')
    status_parser.add_argument('--config', '-c', help='Configuration file')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare models from a job')
    compare_parser.add_argument('--job', required=True, help='Job ID to compare models from')
    compare_parser.add_argument('--config', '-c', help='Configuration file')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'run':
        run_pipeline(args)
    elif args.command == 'predict':
        predict_with_model(args)
    elif args.command == 'status':
        show_job_status(args)
    elif args.command == 'compare':
        compare_models(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()