import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
import os
import ipp

# Task Detection
def detect_task_type(df, target_column):
    target = df[target_column]
    unique_vals = target.nunique()
    if pd.api.types.is_numeric_dtype(target) and unique_vals > 10:
        return "regression"
    else:
        return "classification"

# Profile Extraction
def extract_data_profile(df, target_column):
    df_copy = df.copy()
    n_samples, n_features = df_copy.drop(columns=[target_column]).shape
    num_cols = df_copy.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_copy.select_dtypes(exclude=np.number).columns.tolist()

    corr_matrix = df_copy[num_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(col, row) for col in upper_tri.columns for row in upper_tri.index if upper_tri.loc[row, col] > 0.8]

    outlier_ratio = 0
    if len(num_cols) > 0:
        z_scores = np.abs((df_copy[num_cols] - df_copy[num_cols].mean()) / df_copy[num_cols].std())
        outlier_flags = (z_scores > 3).sum(axis=1)
        outlier_ratio = (outlier_flags > 0).sum() / len(df_copy)

    target_skew = df_copy[target_column].skew() if pd.api.types.is_numeric_dtype(df_copy[target_column]) else None
    feature_variances = df_copy.drop(columns=[target_column]).var().mean()
    target_kurtosis = df_copy[target_column].kurt() if pd.api.types.is_numeric_dtype(df_copy[target_column]) else None

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "numeric_features": len(num_cols),
        "categorical_features": len(cat_cols),
        "high_corr_pairs": len(high_corr_pairs),
        "outlier_ratio": outlier_ratio,
        "target_skew": target_skew,
        "feature_variance": feature_variances,
        "target_kurtosis": target_kurtosis
    }

# Confidence Scoring
def score_regression_models(profile):
    scores = {}
    n_samples = profile["n_samples"]
    n_features = profile["n_features"]
    corr = profile["high_corr_pairs"]
    outliers = profile["outlier_ratio"]
    skew = profile["target_skew"]
    cat_features = profile["categorical_features"]
    feature_variance = profile.get("feature_variance", 0)
    kurtosis = profile.get("target_kurtosis", 0)

    scores["Linear Regression"] = (corr < 5) * 20 + (outliers < 0.3) * 20 + (skew is not None and abs(skew) < 1) * 30 + (feature_variance < 50) * 30
    scores["Ridge Regression"] = (outliers < 0.4) * 35 + (n_features > 2) * 30 + (kurtosis is not None and abs(kurtosis) < 3) * 35
    scores["Lasso Regression"] = (corr > 2) * 35 + (n_features > 5) * 35 + (feature_variance > 50) * 30
    scores["ElasticNet Regression"] = (corr > 2) * 30 + (outliers < 0.4) * 40 + (kurtosis is not None and abs(kurtosis) < 3) * 30
    scores["Polynomial Regression"] = (n_features <= 3) * 40 + (skew is not None and abs(skew) < 1.5) * 30 + (feature_variance < 50) * 30
    scores["Support Vector Regression (SVR)"] = (n_samples <= 100) * 40 + (skew is not None and abs(skew) < 2) * 30 + (feature_variance < 30) * 30
    scores["Decision Tree"] = (outliers >= 0.2) * 40 + (skew is not None and abs(skew) >= 1) * 40 + (kurtosis is not None and abs(kurtosis) > 2) * 30
    scores["Random Forest"] = (outliers >= 0.2) * 30 + (skew is not None and abs(skew) >= 1) * 30 + (n_samples > 50) * 40
    scores["KNN Regressor"] = (n_samples < 200) * 40 + (outliers < 0.2) * 30 + (feature_variance < 50) * 30
    scores["Gradient Boosting"] = (skew is not None and abs(skew) >= 0.5) * 40 + (kurtosis is not None and abs(kurtosis) > 2) * 30 + (n_samples > 50) * 30
    return scores

# Generalized Categorical Preprocessor
def get_categorical_only_preprocessor(df, target_column):
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ], remainder='passthrough')

# Build Pipeline
def build_pipeline(model, df, target_column):
    preprocessor = get_categorical_only_preprocessor(df, target_column)
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

# Evaluate Models
def evaluate_models(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "ElasticNet Regression": ElasticNet(),
        "SVR": SVR(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "KNN Regressor": KNeighborsRegressor(),
        "Gradient Boosting": GradientBoostingRegressor()
    }

    performance = {}
    for name, model in models.items():
        try:
            pipeline = build_pipeline(model, X, target_column)
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            performance[name] = {
                "r2_score": r2_score(y_test, preds),
                "rmse": np.sqrt(mean_squared_error(y_test, preds))
            }
        except Exception as e:
            performance[name] = {"error": str(e)}

    return performance

# Save to CSV

def save_summary_with_top_models(profile, task_type, performance, filename="model_training_summary.csv"):
    sorted_models = sorted(
        [(k, v["r2_score"]) for k, v in performance.items() if "r2_score" in v],
        key=lambda x: x[1],
        reverse=True
    )
    top_models = [m[0] for m in sorted_models[:3]]
    confidence_scores = score_regression_models(profile)
    top_confidence = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": profile["n_samples"],
        "n_features": profile["n_features"],
        "numeric_features": profile["numeric_features"],
        "categorical_features": profile["categorical_features"],
        "high_corr_pairs": profile["high_corr_pairs"],
        "outlier_ratio": round(profile["outlier_ratio"], 3),
        "target_skew": round(profile["target_skew"], 3) if profile["target_skew"] is not None else None,
        "feature_variance": round(profile["feature_variance"], 3),
        "target_kurtosis": round(profile["target_kurtosis"], 3) if profile["target_kurtosis"] is not None else None,
        "task_type": task_type,
        "actual_top_model_1": top_models[0],
        "actual_top_model_2": top_models[1],
        "actual_top_model_3": top_models[2],
        "predicted_top_model_1": top_confidence[0][0],
        "predicted_top_model_2": top_confidence[1][0],
        "predicted_top_model_3": top_confidence[2][0]
    }

    df_log = pd.DataFrame([row])
    file_exists = os.path.isfile(filename)
    df_log.to_csv(filename, mode='a', header=not file_exists, index=False)

    result_summary = {
        "actual_top_model_1": top_models[0],
        "actual_r2_score_1": round(performance[top_models[0]]["r2_score"], 3),
        "actual_top_model_2": top_models[1],
        "actual_r2_score_2": round(performance[top_models[1]]["r2_score"], 3),
        "actual_top_model_3": top_models[2],
        "actual_r2_score_3": round(performance[top_models[2]]["r2_score"], 3),
        "predicted_top_model_1": top_confidence[0][0],
        "confidence_score_1": top_confidence[0][1],
        "predicted_top_model_2": top_confidence[1][0],
        "confidence_score_2": top_confidence[1][1],
        "predicted_top_model_3": top_confidence[2][0],
        "confidence_score_3": top_confidence[2][1]
    }
    # print(f"\n✅ Metadata + Top Models saved to {filename}")
    # print("\n📋 Summary:")
    # print(result_summary)
    return result_summary

# --- Encrypting and Updating JSON ---
import json
import base64
import os
from cryptography.fernet import Fernet

def encrypt_data(data, fernet):
    """Encrypts and encodes a string using Fernet."""
    return fernet.encrypt(data.encode()).decode()

def update_model_mind(model_mind_output):
    # Generate encryption key
    key = Fernet.generate_key()
    fernet = Fernet(key)

    # Prepare actual_model and predicted_model lists
    actual_model = []
    predicted_model = []

    # Actual models
    for i in range(1, 4):
        model = model_mind_output.get(f'actual_top_model_{i}')
        r2 = model_mind_output.get(f'actual_r2_score_{i}')
        if model and r2 is not None:
            encrypted_model = encrypt_data(model, fernet)
            encrypted_r2 = encrypt_data(str(r2), fernet)
            actual_model.append({ "model": encrypted_model, "r2_score": encrypted_r2 })

    # Predicted models
    for i in range(1, 4):
        model = model_mind_output.get(f'predicted_top_model_{i}')
        score = model_mind_output.get(f'confidence_score_{i}')
        if model and score is not None:
            encrypted_model = encrypt_data(model, fernet)
            encrypted_score = encrypt_data(str(score), fernet)
            predicted_model.append({ "model": encrypted_model, "confidence_score": encrypted_score })

    # Load existing status.json or initialize
    try:
        with open(ipp.json_file_path, "r") as file:
            status_data = ipp.json.load(file)
    except FileNotFoundError:
        status_data = {}

    # Update Model_Mind and append encryption key
    status_data["Model_Mind"] = {
        "actual_model": actual_model,
        "predicted_model": predicted_model
    }
    status_data["key"] = key.decode()

    # Save back to JSON
    with open(ipp.json_file_path, "w") as file:
        ipp.json.dump(status_data, file, indent=4)

def workflow(df, predictor_variable):
    # Run workflow
    task_type = detect_task_type(df, predictor_variable)
    # print(f"\U0001F4CC Task Type: {task_type}")
    profile = extract_data_profile(df, predictor_variable)
    performance = evaluate_models(df, predictor_variable)
    model_mind_output = save_summary_with_top_models(profile, task_type, performance)
    update_model_mind(model_mind_output)

# print(workflow(df, predictor_variable))


