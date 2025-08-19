# API-Model

## AutoML Phase 1 - All-Numeric Regression Pipeline

🤖 **Intelligent AutoML system for numeric regression problems with automated model selection, feature engineering, and comprehensive evaluation.**

## 🌟 Features

- **Smart Model Selection** - AI analyzes your data and recommends the best models
- **Automated Feature Engineering** - Creates new features automatically to improve predictions
- **Comprehensive Preprocessing** - Handles missing values, outliers, scaling, and correlations
- **Real-time Progress Tracking** - Visual progress bars and detailed logging
- **Interactive Target Selection** - System analyzes and recommends target variables
- **Multiple Training Modes** - Auto, Quick, and Comprehensive options
- **Prediction Engine** - Easy predictions on new data using trained models
- **Rich Visualizations** - Model comparison charts and performance plots
- **Job Management** - Save, track, and reuse training jobs

## 📋 Requirements

### System Requirements
- Python 3.7+
- 4GB+ RAM recommended
- CSV datasets with numeric data

### Dependencies

Create a `requirements.txt` file:

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
PyYAML>=5.4.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
tqdm>=4.62.0
joblib>=1.1.0
scipy>=1.7.0
statsmodels>=0.13.0
```

### Installation

```bash
# 1. Clone or download the project
# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Create required directories
mkdir -p config logs jobs

# 4. Verify installation
python3 main.py --help
```

## 🚀 Quick Start

### 1. Basic Training
```bash
# Let the system analyze your data and guide you
python3 main.py run --file your_data.csv
```

The system will:
- ✅ Analyze your dataset
- ✅ Recommend target variables
- ✅ Train the best models automatically
- ✅ Create performance visualizations
- ✅ Save everything for future use

### 2. Training Modes

```bash
# Auto mode - AI-suggested models only (recommended)
python3 main.py run --file data.csv --target price --mode auto

# Quick mode - Top 3 models only (fastest)
python3 main.py run --file data.csv --target price --mode quick

# Comprehensive mode - All available models (thorough)
python3 main.py run --file data.csv --target price --mode comprehensive
```

### 3. Making Predictions
```bash
# Use trained model for predictions
python3 main.py predict --model job_20250818_231348 --file new_data.csv
```

## 📊 Commands Reference

### Training Pipeline
```bash
# Interactive target selection
python3 main.py run --file dataset.csv

# Specify target variable
python3 main.py run --file dataset.csv --target column_name

# Batch mode (no prompts)
python3 main.py run --file dataset.csv --target column_name --batch

# Custom configuration
python3 main.py run --file dataset.csv --config custom_config.yaml
```

### Job Management
```bash
# View all recent jobs
python3 main.py status

# Check specific job details
python3 main.py status --job job_20250818_231348

# Compare models from a job
python3 main.py compare --job job_20250818_231348
```

### Making Predictions
```bash
# Basic prediction
python3 main.py predict --model job_ID --file new_data.csv

# Custom output file
python3 main.py predict --model job_ID --file new_data.csv --output predictions.csv
```

### Help
```bash
# General help
python3 main.py --help

# Command-specific help
python3 main.py run --help
python3 main.py predict --help
```

## 🎯 Target Variable Selection

### ✅ Highly Recommended
- **Numeric columns** (int64, float64)
- **Good value range** (not constant)
- **< 20% missing values**
- **Clear prediction target**

### ⚠️ Possible but Challenging
- **20-50% missing values**
- **Many unique values**
- **Requires careful consideration**

### ❌ Not Recommended
- **Text/categorical columns** (use Phase 2)
- **Constant values** (no variance)
- **ID columns** (too many unique values)
- **> 50% missing values**
- **DateTime columns** (needs feature engineering)

## 📁 Output Structure

Each training job creates a complete results folder:

```
jobs/job_20250818_231348/
├── models/                           # Trained models
│   ├── best_model.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
├── visualizations/                   # Charts and plots
│   ├── model_comparison.png
│   ├── performance_summary.png
│   └── results_table.png
├── preprocessing_pipeline.pkl        # Data preprocessing
├── feature_engineering_pipeline.pkl # Feature engineering
├── job_summary.json                 # Quick summary
└── training_summary.json            # Detailed results
```

## 🔧 Configuration

### Default Configuration
The system uses `config/default_config.yaml`. You can create custom configurations:

```yaml
pipeline:
  preprocessing:
    outlier_method: "percentile"
    outlier_threshold: 0.05
    correlation_threshold: 0.95
    
  feature_engineering:
    polynomial_features: true
    interaction_features: true
    
  model_selection:
    cv_folds: 5
    random_state: 42
    
logging:
  level: "INFO"
  log_dir: "logs"
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError
```bash
# Install missing packages
pip3 install package_name

# Or reinstall all requirements
pip3 install -r requirements.txt --force-reinstall
```

#### 2. "yeo-johnson_transformed" Error
This indicates a feature transformation issue. Try:
```bash
# Use quick mode to bypass complex transformations
python3 main.py run --file data.csv --target column --mode quick
```

#### 3. Visualization Errors
If plots aren't generating:
```bash
# Install/update visualization packages
pip3 install matplotlib seaborn plotly --upgrade

# Check if GUI backend is available
python3 -c "import matplotlib.pyplot as plt; plt.figure(); print('Matplotlib working')"
```

#### 4. Memory Issues
For large datasets:
```bash
# Use quick mode for faster processing
python3 main.py run --file large_data.csv --target column --mode quick
```

#### 5. Target Variable Issues
- Ensure target column exists in your data
- Use numeric columns only for Phase 1
- Check for constant or ID-like columns

### Debug Mode
```bash
# Run with verbose logging
python3 main.py run --file data.csv --target column --config debug_config.yaml
```

## 📈 Performance Tips

### Data Preparation
- **Remove ID columns** before training
- **Handle missing values** in advance if possible
- **Check target variable distribution**
- **Ensure sufficient data** (100+ rows recommended)

### Training Optimization
- Use **auto mode** for best speed/performance balance
- Use **quick mode** for rapid prototyping
- Use **comprehensive mode** for final production models

### System Resources
- **4GB+ RAM** for medium datasets
- **SSD storage** recommended for faster I/O
- **Multiple CPU cores** speed up training

## 🔄 Example Workflow

```bash
# 1. Analyze and train models
python3 main.py run --file sales_data.csv
# → Select target interactively
# → Returns: job_20250818_231348

# 2. Check results
python3 main.py status --job job_20250818_231348
python3 main.py compare --job job_20250818_231348

# 3. Make predictions
python3 main.py predict --model job_20250818_231348 --file new_sales.csv

# 4. View visualizations
# → Check: jobs/job_20250818_231348/visualizations/
```

## 🎨 Visualization Fixes

If visualizations aren't working, try:

### Install Backend
```bash
# For headless servers
pip3 install matplotlib --upgrade
export MPLBACKEND=Agg

# For GUI systems
pip3 install matplotlib seaborn plotly --upgrade
```

### Alternative: Generate Plots Manually
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results = pd.read_csv('jobs/your_job_id/model_results.csv')

# Create comparison plot
plt.figure(figsize=(12, 6))
sns.barplot(data=results, x='Model', y='Test Score')
plt.xticks(rotation=45)
plt.title('Model Performance Comparison')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 📝 Supported Models

- **Linear Regression** - Fast baseline
- **Random Forest** - Robust ensemble
- **XGBoost** - Gradient boosting
- **LightGBM** - Fast gradient boosting
- **CatBoost** - Categorical boosting
- **Support Vector Regression** - Non-linear patterns
- **Neural Networks** - Deep learning

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check this README and inline help
- **Examples**: See example workflows above

---

**🚀 Ready to start? Run: `python3 main.py run --file your_data.csv`**
