"""
generate_crazy_dataset_numerical.py
Creates a messy dataset with ALL NUMERICAL values and all possible flaws to test the AutoML system
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_crazy_numerical_dataset(n_rows=500):
    """Generate a crazy dataset with all numerical values and all possible flaws"""
    
    print("🔥 Generating CRAZY test dataset with ALL NUMERICAL VALUES...")
    
    data = {}
    
    # 1. GOOD TARGET VARIABLES (what we want to predict)
    print("  ✅ Adding good target variables...")
    
    # Perfect regression target - house prices
    base_price = np.random.normal(300000, 100000, n_rows)
    base_price = np.abs(base_price)  # No negative prices
    data['house_price'] = base_price
    
    # Another good target - student grades (0-100)
    data['student_grade'] = np.random.beta(2, 2, n_rows) * 100
    
    # 2. ID COLUMNS (should NOT be targets) - NOW NUMERICAL
    print("  ⚠️  Adding numerical ID columns (not suitable for targets)...")
    
    # Sequential ID
    data['customer_id'] = range(100001, 100001 + n_rows)  # Start from 100001
    
    # Another sequential ID
    data['record_id'] = range(1, n_rows + 1)
    
    # Random unique IDs
    data['transaction_id'] = np.random.choice(range(500000, 999999), n_rows, replace=False)
    
    # 3. CONSTANT COLUMNS (no variance)
    print("  ❌ Adding constant columns...")
    
    # Completely constant
    data['constant_column'] = [42] * n_rows
    
    # Another numeric constant
    data['constant_number'] = [999] * n_rows
    
    # Almost constant (99% same value)
    data['almost_constant'] = [1] * int(n_rows * 0.99) + [0] * int(n_rows * 0.01)
    
    # 4. HIGHLY MISSING DATA
    print("  🕳️  Adding columns with high missing data...")
    
    # 80% missing
    data['mostly_missing'] = [np.nan] * int(n_rows * 0.8) + list(np.random.normal(100, 20, int(n_rows * 0.2)))
    
    # 95% missing
    data['extremely_missing'] = [np.nan] * int(n_rows * 0.95) + list(np.random.normal(50, 10, int(n_rows * 0.05)))
    
    # 5. HIGHLY SKEWED FEATURES (need transformation)
    print("  📈 Adding highly skewed features...")
    
    # Extreme right skew (exponential-like)
    data['income_skewed'] = np.random.exponential(50000, n_rows)
    
    # Log-normal distribution (very skewed)
    data['wealth_skewed'] = np.random.lognormal(10, 2, n_rows)
    
    # Power law distribution
    data['popularity_skewed'] = np.random.pareto(0.5, n_rows) * 1000
    
    # Gamma distribution (moderately skewed)
    data['sales_moderate_skew'] = np.random.gamma(2, 2, n_rows) * 1000
    
    # 6. FEATURES THAT WERE TEXT - NOW NUMERICAL
    print("  🔧 Adding features converted to numerical...")
    
    # Date/time as numerical timestamps
    start_date = datetime(2020, 1, 1)
    dates = [(start_date + timedelta(days=random.randint(0, 1460))) for _ in range(n_rows)]
    data['purchase_timestamp'] = [int(d.timestamp()) for d in dates]
    
    # Extract numerical features from dates
    data['purchase_year'] = [d.year for d in dates]
    data['purchase_month'] = [d.month for d in dates]
    data['purchase_day'] = [d.day for d in dates]
    data['purchase_weekday'] = [d.weekday() for d in dates]  # 0=Monday, 6=Sunday
    
    # High cardinality categorical as numerical codes
    data['category_code'] = np.random.randint(1, 101, n_rows)  # 1-100 categories as numbers
    
    # Product types as numerical codes
    data['product_type_code'] = np.random.randint(1, 8, n_rows)  # 1-7 product types
    
    # Boolean as 0/1
    data['is_premium'] = np.random.choice([0, 1], n_rows, p=[0.7, 0.3])
    
    # Quality levels as numbers
    data['quality_level'] = np.random.choice([1, 2, 3, 4], n_rows, p=[0.1, 0.3, 0.4, 0.2])
    
    # 7. HIGHLY CORRELATED FEATURES (redundant)
    print("  🔄 Adding highly correlated features...")
    
    # Base feature
    base_feature = np.random.normal(1000, 200, n_rows)
    data['base_metric'] = base_feature
    
    # Almost identical features (0.95+ correlation)
    data['duplicate_metric_1'] = base_feature + np.random.normal(0, 10, n_rows)  # 99% correlation
    data['duplicate_metric_2'] = base_feature * 1.1 + np.random.normal(0, 20, n_rows)  # 98% correlation
    data['duplicate_metric_3'] = base_feature + 50 + np.random.normal(0, 15, n_rows)  # 97% correlation
    
    # Perfectly correlated (one is just transformation of other)
    data['temperature_celsius'] = np.random.normal(20, 10, n_rows)
    data['temperature_fahrenheit'] = data['temperature_celsius'] * 9/5 + 32  # Perfect correlation
    
    # 8. FEATURES PERFECT FOR INTERACTION TERMS
    print("  🤝 Adding features that need interactions...")
    
    # Features that should be multiplied
    data['length'] = np.random.normal(10, 2, n_rows)
    data['width'] = np.random.normal(8, 1.5, n_rows)
    # area = length * width (interaction needed)
    
    data['price_per_unit'] = np.random.normal(50, 10, n_rows)
    data['quantity'] = np.random.randint(1, 100, n_rows)
    # total_cost = price_per_unit * quantity (interaction needed)
    
    # 9. FEATURES WITH OUTLIERS
    print("  🎯 Adding features with extreme outliers...")
    
    # Normal feature with extreme outliers
    normal_feature = np.random.normal(100, 20, n_rows)
    # Add 10 extreme outliers
    outlier_indices = np.random.choice(n_rows, 10, replace=False)
    normal_feature[outlier_indices] = np.random.normal(1000, 100, 10)  # 10x normal values
    data['feature_with_outliers'] = normal_feature
    
    # Another with negative outliers
    positive_feature = np.abs(np.random.normal(50, 10, n_rows))
    outlier_indices = np.random.choice(n_rows, 5, replace=False)
    positive_feature[outlier_indices] = np.random.normal(-100, 20, 5)  # Negative outliers
    data['positive_with_negative_outliers'] = positive_feature
    
    # 10. POLYNOMIAL RELATIONSHIPS (need polynomial features)
    print("  📊 Adding polynomial relationships...")
    
    # Quadratic relationship
    x = np.random.normal(0, 1, n_rows)
    data['x_linear'] = x
    data['y_quadratic'] = 2 * x**2 + 3 * x + np.random.normal(0, 0.5, n_rows)
    
    # Cubic relationship  
    data['z_cubic'] = x**3 - 2 * x**2 + x + np.random.normal(0, 1, n_rows)
    
    # 11. MIXED SCALE VALUES - ALL NUMERICAL NOW
    print("  🔀 Adding numerical values with different meanings...")
    
    # Values that could be numeric or categorical (but all numeric now)
    data['score_or_category'] = np.random.choice([1, 2, 3, 10, 25, 50, 99, 100], n_rows)
    
    # 12. FEATURES WITH DIFFERENT SCALES (need scaling)
    print("  ⚖️  Adding features with different scales...")
    
    data['tiny_feature'] = np.random.normal(0.001, 0.0001, n_rows)  # Very small scale
    data['huge_feature'] = np.random.normal(1000000, 100000, n_rows)  # Very large scale
    data['percentage'] = np.random.uniform(0, 100, n_rows)  # 0-100 scale
    data['normalized'] = np.random.uniform(-1, 1, n_rows)  # -1 to 1 scale
    
    # 13. FEATURES THAT NEED BINNING/GROUPING
    print("  📦 Adding features that need binning...")
    
    # Age that should be binned into groups
    data['customer_age'] = np.random.randint(18, 80, n_rows)
    
    # Income that should be grouped
    data['annual_income'] = np.random.lognormal(11, 0.5, n_rows)
    
    # 14. COMPLEX TARGET RELATIONSHIPS
    print("  🎯 Creating complex target relationships...")
    
    # Make house_price depend on multiple features with interactions
    data['house_price'] = (
        data['length'] * data['width'] * 1000 +  # Area effect
        data['customer_age'] * 2000 +  # Age effect
        (data['annual_income'] * 0.3) +  # Income effect
        (data['temperature_celsius'] * 500) +  # Climate effect
        np.random.normal(0, 10000, n_rows)  # Noise
    )
    
    # Make sure prices are positive
    data['house_price'] = np.abs(data['house_price'])
    
    # Student grade based on study time and IQ with polynomial relationship
    study_time = np.random.normal(5, 2, n_rows)  # Hours per day
    iq = np.random.normal(100, 15, n_rows)
    data['study_hours'] = np.abs(study_time)
    data['iq_score'] = iq
    data['student_grade'] = (
        30 + 
        data['study_hours'] * 8 + 
        (data['iq_score'] - 100) * 0.5 + 
        data['study_hours']**2 * 0.5 +  # Polynomial term
        np.random.normal(0, 5, n_rows)
    )
    data['student_grade'] = np.clip(data['student_grade'], 0, 100)  # Keep in 0-100 range
    
    # 15. MORE PROBLEMATIC FEATURES
    print("  💥 Adding more problematic features...")
    
    # Feature with only 2 unique values (almost binary)
    data['binary_like'] = np.random.choice([0, 1], n_rows, p=[0.95, 0.05])  # Very unbalanced
    
    # Feature with strange distribution (multimodal)
    data['weird_distribution'] = np.concatenate([
        np.random.normal(-50, 5, n_rows//3),
        np.random.normal(0, 1, n_rows//3),
        np.random.normal(100, 20, n_rows - 2*(n_rows//3))
    ])
    
    # Feature with periodic pattern
    data['periodic_feature'] = np.sin(np.linspace(0, 4*np.pi, n_rows)) + np.random.normal(0, 0.1, n_rows)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # 16. INTRODUCE SOME MISSING VALUES RANDOMLY
    print("  🕳️  Adding random missing values...")
    
    # Add random missing values to numeric columns (5-15% missing)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if 'constant' not in col and 'missing' not in col:  # Don't add to already problematic columns
            missing_pct = random.uniform(0.05, 0.15)
            missing_indices = np.random.choice(df.index, int(len(df) * missing_pct), replace=False)
            df.loc[missing_indices, col] = np.nan
    
    # 17. DUPLICATE ROWS
    print("  📋 Adding duplicate rows...")
    
    # Add 10 duplicate rows
    duplicate_indices = np.random.choice(df.index, 10, replace=False)
    duplicate_rows = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    print(f"✅ Dataset created with {len(df)} rows and {len(df.columns)} columns")
    print(f"🔢 ALL VALUES ARE NUMERICAL!")
    
    return df

def analyze_numerical_dataset_flaws(df):
    """Analyze and report all the flaws in the numerical dataset"""
    
    print("\n🔍 NUMERICAL DATASET FLAW ANALYSIS:")
    print("=" * 60)
    
    # Verify all columns are numerical
    print("📊 Data Types:")
    non_numeric = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) == 0:
        print("   ✅ ALL columns are numerical!")
    else:
        print(f"   ❌ Non-numerical columns found: {list(non_numeric)}")
    
    # 1. ID Columns (high unique ratios)
    id_cols = []
    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.95:
            id_cols.append(col)
    
    print(f"\n🆔 ID-like Columns (too many unique values): {len(id_cols)}")
    for col in id_cols:
        print(f"   • {col}: {df[col].nunique()}/{len(df)} unique ({df[col].nunique()/len(df)*100:.1f}%)")
    
    # 2. Constant columns
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 2:
            constant_cols.append(col)
    
    print(f"\n🔒 Constant/Near-constant columns: {len(constant_cols)}")
    for col in constant_cols:
        print(f"   • {col}: {df[col].nunique()} unique values")
    
    # 3. High missing data
    high_missing = []
    for col in df.columns:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        if missing_pct > 20:
            high_missing.append((col, missing_pct))
    
    print(f"\n🕳️  High missing data columns: {len(high_missing)}")
    for col, pct in high_missing:
        print(f"   • {col}: {pct:.1f}% missing")
    
    # 4. Skewed features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewed_features = []
    for col in numeric_cols:
        try:
            skewness = df[col].skew()
            if abs(skewness) > 1:
                skewed_features.append((col, skewness))
        except:
            pass
    
    print(f"\n📈 Highly skewed features: {len(skewed_features)}")
    for col, skew in sorted(skewed_features, key=lambda x: abs(x[1]), reverse=True)[:10]:
        print(f"   • {col}: skewness = {skew:.2f}")
    
    # 5. Correlation analysis
    try:
        corr_matrix = df[numeric_cols].corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.8 and not np.isnan(corr_val):
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        print(f"\n🔄 Highly correlated feature pairs: {len(high_corr_pairs)}")
        for col1, col2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:5]:
            print(f"   • {col1} ↔ {col2}: {corr:.3f}")
    except:
        print("\n🔄 Could not analyze correlations")
    
    # 6. Scale differences
    scale_info = []
    for col in numeric_cols[:10]:  # Check first 10 for brevity
        try:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if not np.isnan(mean_val) and not np.isnan(std_val):
                scale_info.append((col, mean_val, std_val))
        except:
            pass
    
    print(f"\n⚖️  Scale differences (sample):")
    for col, mean, std in scale_info:
        print(f"   • {col}: mean={mean:.2e}, std={std:.2e}")
    
    # 7. Target recommendations
    print(f"\n🎯 TARGET VARIABLE RECOMMENDATIONS:")
    print("=" * 40)
    print("✅ GOOD TARGETS (use these for testing):")
    print(f"   🏠 house_price: Main target - predicts house prices")
    print(f"      Range: ${df['house_price'].min():,.0f} - ${df['house_price'].max():,.0f}")
    print(f"   📚 student_grade: Secondary target - predicts grades (0-100)")
    print(f"      Range: {df['student_grade'].min():.1f} - {df['student_grade'].max():.1f}")
    print()
    print("❌ BAD TARGETS (system should reject these):")
    print(f"   🆔 customer_id: ID column (too many unique values)")
    print(f"   🔒 constant_column: No variance (all values = 42)")
    print(f"   🕳️  mostly_missing: 80% missing data")
    print()
    print("🎯 PRIMARY TARGET FOR TESTING: house_price")

def main():
    """Generate the crazy numerical test dataset"""
    
    print("🚀 Creating CRAZY TEST DATASET with ALL NUMERICAL VALUES...")
    print("This dataset will test EVERY edge case with numerical data only!\n")
    
    # Generate dataset
    df = generate_crazy_numerical_dataset(n_rows=500)
    
    # Save to CSV
    filename = "crazy_numerical_dataset.csv"
    df.to_csv(filename, index=False)
    print(f"\n💾 Dataset saved as: {filename}")
    
    # Analyze flaws
    analyze_numerical_dataset_flaws(df)
    
    print(f"\n🎯 TARGET SELECTION GUIDE:")
    print("=" * 60)
    print("📋 When prompted 'Select target variable:', use:")
    print()
    print("🏆 PRIMARY TARGET (recommended):")
    print("   → house_price")
    print("     This predicts house prices based on size, location, age, etc.")
    print()
    print("🥈 SECONDARY TARGET (alternative):")
    print("   → student_grade") 
    print("     This predicts student grades based on study hours, IQ, etc.")
    print()
    print("🚫 TEST REJECTION HANDLING:")
    print("   → customer_id (should be rejected as ID)")
    print("   → constant_column (should be rejected - no variance)")
    print("   → mostly_missing (should warn about missing data)")
    print()
    print("💡 USAGE: python main.py run --file crazy_numerical_dataset.csv")
    print("   Then type: house_price")
    print()
    print("🔥 Your AutoML system will be tested with PURE NUMERICAL data!")
    print(f"Dataset shape: {df.shape}")
    print(f"All {df.shape[1]} columns are numerical!")
    print(f"PRIMARY TARGET: house_price")

if __name__ == "__main__":
    main()