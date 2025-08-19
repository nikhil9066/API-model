#!/usr/bin/env python3
"""
Complete Visualization Diagnostic and Fix Script
Run this to test and fix visualization issues
"""

import sys
import os
import warnings
import traceback
warnings.filterwarnings('ignore')

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import matplotlib
        print("✅ matplotlib available")
        
        # Set backend before importing pyplot
        matplotlib.use('Agg')
        print("✅ Set matplotlib backend to 'Agg'")
        
        import matplotlib.pyplot as plt
        print("✅ matplotlib.pyplot imported")
        
        import pandas as pd
        print("✅ pandas imported")
        
        import numpy as np
        print("✅ numpy imported")
        
        try:
            import seaborn as sns
            print("✅ seaborn imported")
        except ImportError:
            print("⚠️ seaborn not available (optional)")
        
        try:
            from scipy import stats
            print("✅ scipy.stats imported")
        except ImportError:
            print("⚠️ scipy not available (some plots may be limited)")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_plot():
    """Test basic matplotlib functionality"""
    print("\n📊 Testing basic plot creation...")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create simple plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        ax.plot(x, y, label='sin(x)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Test Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Try to save
        test_path = 'matplotlib_test.png'
        plt.savefig(test_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Check if file exists
        if os.path.exists(test_path):
            print(f"✅ Basic plot test successful - saved to {test_path}")
            os.remove(test_path)  # Clean up
            return True
        else:
            print("❌ Plot file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Basic plot test failed: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return False

def test_with_sample_data():
    """Test with sample AutoML-like data"""
    print("\n🎯 Testing with sample AutoML data...")
    
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Create sample data
        sample_data = {
            'Model': ['Linear_Regression', 'Random_Forest', 'XGBoost', 'LightGBM'],
            'Test Score': [0.7234, 0.8456, 0.8901, 0.8734],
            'Train Score': [0.7456, 0.9123, 0.9456, 0.9234],
            'Training Time': [0.5, 2.3, 5.7, 3.2]
        }
        
        df = pd.DataFrame(sample_data)
        print("✅ Sample data created")
        
        # Create simple comparison plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        bars = ax.bar(df['Model'], df['Test Score'], color='skyblue', alpha=0.7, edgecolor='navy')
        ax.set_xlabel('Models')
        ax.set_ylabel('Test Score (R²)')
        ax.set_title('Model Performance Comparison')
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate labels
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, score in zip(bars, df['Test Score']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save test plot
        test_path = 'automl_test_plot.png'
        plt.savefig(test_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        if os.path.exists(test_path):
            print(f"✅ AutoML-style plot test successful - saved to {test_path}")
            return True, test_path
        else:
            print("❌ AutoML plot file was not created")
            return False, None
            
    except Exception as e:
        print(f"❌ AutoML plot test failed: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return False, None

def create_emergency_visualizer():
    """Create a minimal emergency visualizer"""
    print("\n🚑 Creating emergency visualizer...")
    
    emergency_code = '''import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def emergency_plot(results_df, save_path='emergency_plot.png'):
    """Emergency plotting function"""
    try:
        if 'Test Score' not in results_df.columns or 'Model' not in results_df.columns:
            print("❌ Required columns missing")
            return False
        
        # Sort data
        df_sorted = results_df.sort_values('Test Score', ascending=False)
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        bars = ax.bar(range(len(df_sorted)), df_sorted['Test Score'], 
                     color='lightblue', edgecolor='darkblue', alpha=0.8)
        
        # Customize
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Score (R²)', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels(df_sorted['Model'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add scores
        for i, (bar, score) in enumerate(zip(bars, df_sorted['Test Score'])):
            ax.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Highlight best
        if len(bars) > 0:
            bars[0].set_color('gold')
            bars[0].set_edgecolor('darkgoldenrod')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Emergency plot saved to {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Emergency plot failed: {e}")
        return False

def fix_job_visualizations(job_dir):
    """Fix visualizations for a specific job"""
    try:
        # Try to load training summary
        import json
        
        summary_file = os.path.join(job_dir, 'training_summary.json')
        if not os.path.exists(summary_file):
            print(f"⚠️ No training summary found in {job_dir}")
            return False
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # Extract model results
        all_models = summary.get('all_models_performance', {})
        if not all_models:
            print(f"⚠️ No model results found in {job_dir}")
            return False
        
        # Convert to DataFrame
        results_data = []
        for model_name, results in all_models.items():
            results_data.append({
                'Model': model_name,
                'Test Score': results.get('test_score', 0),
                'Train Score': results.get('train_score', 0),
                'Training Time': results.get('training_time', 0)
            })
        
        df = pd.DataFrame(results_data)
        
        # Create visualizations directory
        viz_dir = os.path.join(job_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create emergency plot
        plot_path = os.path.join(viz_dir, 'emergency_comparison.png')
        success = emergency_plot(df, plot_path)
        
        if success:
            print(f"✅ Fixed visualizations for {job_dir}")
            return True
        else:
            print(f"❌ Failed to fix visualizations for {job_dir}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing visualizations for {job_dir}: {e}")
        return False

# Test example usage
if __name__ == "__main__":
    # Test data
    sample_results = pd.DataFrame({
        'Model': ['Linear_Regression', 'Random_Forest', 'XGBoost', 'LightGBM', 'CatBoost'],
        'Test Score': [0.7234, 0.8456, 0.8901, 0.8734, 0.8654],
        'Train Score': [0.7456, 0.9123, 0.9456, 0.9234, 0.9123],
        'Training Time': [0.5, 2.3, 5.7, 3.2, 4.1]
    })
    
    print("🧪 Testing emergency visualizer...")
    success = emergency_plot(sample_results, 'test_emergency_plot.png')
    
    if success and os.path.exists('test_emergency_plot.png'):
        print("✅ Emergency visualizer test successful!")
        print("📁 Check test_emergency_plot.png")
    else:
        print("❌ Emergency visualizer test failed")

print("📄 Emergency visualizer code written to emergency_viz.py")
'''
    
    # Write emergency visualizer to file
    with open('emergency_viz.py', 'w') as f:
        f.write(emergency_code)
    
    print("✅ Emergency visualizer created as 'emergency_viz.py'")
    return True

def find_and_fix_jobs():
    """Find existing jobs and fix their visualizations"""
    print("\n🔍 Looking for existing jobs to fix...")
    
    jobs_dir = 'jobs'
    if not os.path.exists(jobs_dir):
        print("⚠️ No jobs directory found")
        return False
    
    job_dirs = [d for d in os.listdir(jobs_dir) if d.startswith('job_')]
    
    if not job_dirs:
        print("⚠️ No job directories found")
        return False
    
    print(f"📁 Found {len(job_dirs)} job(s)")
    
    # Import the emergency functions
    try:
        import pandas as pd
        import json
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        def emergency_plot_local(results_df, save_path):
            """Local emergency plot function"""
            try:
                df_sorted = results_df.sort_values('Test Score', ascending=False)
                
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                bars = ax.bar(range(len(df_sorted)), df_sorted['Test Score'], 
                             color='lightblue', edgecolor='darkblue', alpha=0.8)
                
                ax.set_xlabel('Models', fontsize=12, fontweight='bold')
                ax.set_ylabel('Test Score (R²)', fontsize=12, fontweight='bold')
                ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
                ax.set_xticks(range(len(df_sorted)))
                ax.set_xticklabels(df_sorted['Model'], rotation=45, ha='right')
                ax.grid(axis='y', alpha=0.3)
                
                for i, (bar, score) in enumerate(zip(bars, df_sorted['Test Score'])):
                    ax.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
                
                if len(bars) > 0:
                    bars[0].set_color('gold')
                    bars[0].set_edgecolor('darkgoldenrod')
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                return True
            except Exception as e:
                print(f"Plot error: {e}")
                return False
        
        fixed_count = 0
        for job_dir in job_dirs:
            job_path = os.path.join(jobs_dir, job_dir)
            
            try:
                # Load training summary
                summary_file = os.path.join(job_path, 'training_summary.json')
                if not os.path.exists(summary_file):
                    continue
                
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                all_models = summary.get('all_models_performance', {})
                if not all_models:
                    continue
                
                # Convert to DataFrame
                results_data = []
                for model_name, results in all_models.items():
                    results_data.append({
                        'Model': model_name,
                        'Test Score': results.get('test_score', 0),
                        'Train Score': results.get('train_score', 0),
                        'Training Time': results.get('training_time', 0)
                    })
                
                df = pd.DataFrame(results_data)
                
                # Create visualizations
                viz_dir = os.path.join(job_path, 'visualizations')
                os.makedirs(viz_dir, exist_ok=True)
                
                plot_path = os.path.join(viz_dir, 'fixed_comparison.png')
                if emergency_plot_local(df, plot_path):
                    print(f"✅ Fixed: {job_dir}")
                    fixed_count += 1
                else:
                    print(f"❌ Failed: {job_dir}")
                    
            except Exception as e:
                print(f"❌ Error with {job_dir}: {e}")
        
        print(f"\n🎉 Fixed visualizations for {fixed_count} job(s)")
        return fixed_count > 0
        
    except Exception as e:
        print(f"❌ Error in fix process: {e}")
        return False

def check_system_info():
    """Check system information for debugging"""
    print("\n🖥️ System Information:")
    
    try:
        print(f"Python version: {sys.version}")
        print(f"Platform: {sys.platform}")
        
        import matplotlib
        print(f"Matplotlib version: {matplotlib.__version__}")
        print(f"Matplotlib backend: {matplotlib.get_backend()}")
        
        import pandas as pd
        print(f"Pandas version: {pd.__version__}")
        
        import numpy as np
        print(f"Numpy version: {np.__version__}")
        
        # Check display environment
        display_vars = ['DISPLAY', 'WAYLAND_DISPLAY', 'XDG_SESSION_TYPE']
        for var in display_vars:
            value = os.environ.get(var, 'Not set')
            print(f"{var}: {value}")
            
    except Exception as e:
        print(f"Error getting system info: {e}")

def main():
    """Main diagnostic function"""
    print("🔧 AutoML Visualization Diagnostic Tool")
    print("=" * 50)
    
    # Check system info
    check_system_info()
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed - install missing packages:")
        print("pip3 install matplotlib pandas numpy seaborn scipy")
        return False
    
    # Test basic plotting
    if not test_basic_plot():
        print("\n❌ Basic plotting failed - check matplotlib installation")
        return False
    
    # Test with sample data
    success, test_file = test_with_sample_data()
    if not success:
        print("\n❌ Sample data plotting failed")
        return False
    
    # Create emergency visualizer
    create_emergency_visualizer()
    
    # Try to fix existing jobs
    find_and_fix_jobs()
    
    print("\n🎉 Diagnostic complete!")
    print("\n📋 Summary:")
    print("✅ All import tests passed")
    print("✅ Basic plotting works")
    print("✅ Sample data plotting works")
    print("✅ Emergency visualizer created")
    print("\n📁 Files created:")
    print("  - emergency_viz.py (standalone visualizer)")
    if success and test_file:
        print(f"  - {test_file} (test plot)")
    
    print("\n💡 Next steps:")
    print("1. Use emergency_viz.py for manual plotting")
    print("2. Run your AutoML pipeline again")
    print("3. If issues persist, use quick mode:")
    print("   python3 main.py run --file data.csv --target column --mode quick")
    
    return True

if __name__ == "__main__":
    main()