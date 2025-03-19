import ipp
def remove_outliers(df, predictor_variable):
    """
    Remove outliers from a specified predictor variable using different methods and
    choose the method that removes the maximum number of outliers.
    
    Parameters:
        df (ipp.pd.DataFrame): The input dataframe.
        predictor_variable (str): The column name for the variable to remove outliers from.
    
    Returns:
        ipp.pd.DataFrame: The dataframe with outliers removed for the specified predictor variable.
        str: The method used to remove outliers.
        int: The number of outliers removed.
    """
    
    # Method results
    methods = {
        'iqr': remove_outliers_iqr(df, predictor_variable),
        'sd3': remove_outliers_sd3(df, predictor_variable),
        'percentile': remove_outliers_percentile(df, predictor_variable)
    }
    
    # Calculate the number of outliers removed for each method
    outliers_removed = {method: len(df) - len(methods[method]) for method in methods}
    
    # Print number of outliers removed by each method
    for method in methods:
        print(f"Method: {method}, Outliers removed: {outliers_removed[method]}")
    
    # Get the method that removes the maximum number of outliers
    max_outliers_removed_method = max(outliers_removed, key=outliers_removed.get)
    
    # Print the method used and the number of outliers removed
    print(f"\nBest method used: {max_outliers_removed_method}")
    print(f"Number of outliers removed: {outliers_removed[max_outliers_removed_method]}")
    
    # Return the cleaned dataframe (no transformation, just removing outliers)
    return methods[max_outliers_removed_method], max_outliers_removed_method, outliers_removed[max_outliers_removed_method]

def remove_outliers_iqr(df, predictor_variable) -> ipp.pd.DataFrame:
    """
    Removes outliers using the Interquartile Range (IQR) method.
    """
    Q1 = df[predictor_variable].quantile(0.25)
    Q3 = df[predictor_variable].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[predictor_variable] < lower_bound) | (df[predictor_variable] > upper_bound)]
    print(f"IQR method removes: {len(outliers)} outliers")
    return df[(df[predictor_variable] >= lower_bound) & (df[predictor_variable] <= upper_bound)]

def remove_outliers_sd3(df, predictor_variable) -> ipp.pd.DataFrame:
    """
    Removes outliers using the Standard Deviation method (SD3).
    """
    mean = df[predictor_variable].mean()
    std_dev = df[predictor_variable].std()
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev
    outliers = df[(df[predictor_variable] < lower_bound) | (df[predictor_variable] > upper_bound)]
    print(f"SD3 method removes: {len(outliers)} outliers")
    return df[(df[predictor_variable] >= lower_bound) & (df[predictor_variable] <= upper_bound)]

def remove_outliers_percentile(df, predictor_variable) -> ipp.pd.DataFrame:
    """
    Removes outliers using Percentile method.
    """
    lower_percentile = df[predictor_variable].quantile(0.01)
    upper_percentile = df[predictor_variable].quantile(0.99)
    outliers = df[(df[predictor_variable] < lower_percentile) | (df[predictor_variable] > upper_percentile)]
    print(f"Percentile method removes: {len(outliers)} outliers")
    return df[(df[predictor_variable] >= lower_percentile) & (df[predictor_variable] <= upper_percentile)]

## other things added



# Example use case inside the remove_outliers function
def main_outliers(df, predictor_variable):
    
    '''
    Call the outlier removal methods (IQR, SD3, Percentile) and choose the one
    that removes the most outliers, as shown in previous example...
    Let's assume the method removes outliers and returns the count 
    '''
    
    df_filtered, method_used, outliers_removed = ipp.remove_outliers(df, predictor_variable)

    # Load the existing status.json data
    with open(ipp.json_file_path, 'r') as f:
        status_data = ipp.json.load(f)
    
    # Update outlier detection status
    if outliers_removed > 0:
        # If outliers were removed, update detection status and method details
        status_data["pre_processing"]["outliers"]["detection"] = True
        status_data["pre_processing"]["outliers"]["method"] = method_used
        status_data["pre_processing"]["outliers"]["outliers_removed"] = outliers_removed
    else:
        # If no outliers were removed, ensure detection remains False
        status_data["pre_processing"]["outliers"]["detection"] = False
    
    # Save the updated status data back to the file
    with open(ipp.json_file_path, 'w') as f:
        ipp.json.dump(status_data, f, indent=4)
    
    return df_filtered


## adding the CFS status update part
def update_low_correlation_features(lower_threshold, low_loss):
    """
    Updates the status.json file when low-correlation features are removed.

    Parameters:
    - lower_threshold (float): The threshold used for low correlation feature removal.
    - low_loss (dict): Features removed due to low correlation (below lower_threshold).
    """
    
    if not low_loss:
        return

    try:
        with open(ipp.json_file_path, "r") as file:
            status_data = ipp.json.load(file)
    except FileNotFoundError:
        status_data = {}

    # Ensure "pre_processing" and "CFS" sections exist
    status_data.setdefault("pre_processing", {}).setdefault("CFS", {"feature_selection": False})

    # Set feature_selection to True since we're removing features
    status_data["pre_processing"]["CFS"]["feature_selection"] = True

    # Create an entry with timestamp
    timestamp = ipp.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_entry = {
        "timestamp": timestamp,
        "lower_threshold": lower_threshold,
        "Low_loss": low_loss
    }

    # Append new entry without overwriting previous ones
    status_data["pre_processing"]["CFS"].setdefault("history", []).append(update_entry)

    # Write the updated status back to the file
    with open(ipp.json_file_path, "w") as file:
        ipp.json.dump(status_data, file, indent=4)


def update_high_correlation_features(high_loss):
    """
    Updates the status.json file when high-correlation features are removed.

    Parameters:
    - high_loss (dict): Features removed due to high correlation (above 0.90).
    """
    
    if not high_loss:
        return

    try:
        with open(ipp.json_file_path, "r") as file:
            status_data = ipp.json.load(file)
    except FileNotFoundError:
        status_data = {}

    # Ensure "pre_processing" and "CFS" sections exist
    status_data.setdefault("pre_processing", {}).setdefault("CFS", {"feature_selection": False})

    # Set feature_selection to True since we're removing features
    status_data["pre_processing"]["CFS"]["feature_selection"] = True

    # Create an entry with timestamp
    timestamp = ipp.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_entry = {
        "timestamp": timestamp,
        "High_loss": high_loss,
    }

    # Append new entry without overwriting previous ones
    status_data["pre_processing"]["CFS"].setdefault("history", []).append(update_entry)

    # Write the updated status back to the file
    with open(ipp.json_file_path, "w") as file:
        ipp.json.dump(status_data, file, indent=4)

# Section Ends

## CFS Section
def remove_low_correlation_features(df, lower_threshold, target_variable):
    """Removes features with correlation below the given lower_threshold with the target variable."""
    correlation_matrix = df.corr()
    target_corr = correlation_matrix[target_variable]

    # Identify low correlation features
    low_correlation_features = {
        col: round(target_corr[col], 4)
        for col in target_corr.index
        if abs(target_corr[col]) < lower_threshold and col != target_variable
    }

    # Drop low correlation features
    df_selected = df.drop(columns=list(low_correlation_features.keys()))

    # Return DataFrame, removed features, and threshold
    return df_selected, {"Low_loss": low_correlation_features, "Threshold": lower_threshold}


def remove_high_correlation_features(df, target_variable):
    """Removes one feature from each highly correlated pair (correlation above 0.90)."""
    HIGHER_THRESHOLD = 0.90
    correlation_matrix = df.corr()
    target_corr = correlation_matrix[target_variable]

    high_correlation_features = {}
    processed = set()

    # Identify highly correlated features
    for col in correlation_matrix.columns:
        if col == target_variable:
            continue
        for row in correlation_matrix.columns:
            if row != col and row != target_variable and (row, col) not in processed:
                corr_value = abs(correlation_matrix.loc[col, row])
                if corr_value > HIGHER_THRESHOLD:
                    # Drop the feature with lower correlation to the target variable
                    if abs(target_corr[col]) < abs(target_corr[row]):
                        high_correlation_features[col] = round(corr_value, 4)
                    else:
                        high_correlation_features[row] = round(corr_value, 4)
                processed.add((col, row))

    # Drop high correlation features
    df_selected = df.drop(columns=list(high_correlation_features.keys()))

    # Return DataFrame and removed features
    return df_selected, {"High_loss": high_correlation_features, "Threshold": HIGHER_THRESHOLD}

## Section Ends

## Normality Check
# Function to check normality using multiple tests
def check_normality(df):
    results = []
    
    for col in df.columns:
        data = df[col].dropna()
        shapiro_p = ipp.stats.shapiro(data)[1] if len(data) < 5000 else ipp.np.nan  # Shapiro fails for large samples
        ks_p = ipp.stats.kstest(data, 'norm')[1]
        dagostino_p = ipp.stats.normaltest(data)[1]
        
        skewness = data.skew()
        
        results.append({
            "Feature": col,
            "Skewness": skewness,
            "Shapiro-Wilk p": shapiro_p,
            "K-S Test p": ks_p,
            "D’Agostino p": dagostino_p,
            "Needs Transformation": (abs(skewness) > 0.5) or (dagostino_p < 0.05) or (ks_p < 0.05)
        }) 
    
    return ipp.pd.DataFrame(results)
## end of normality check


## skewed features transformation
# Function to handle highly skewed features
def handle_high_skew(df_selected_3, highly_skewed):
    print("\n🔹 Transforming Highly Skewed Features...")
    transformed_features = []
    def transform_column(col, transformations):
        original_skew = df_selected_3[col].skew()
        best_method, best_data, best_skew = None, df_selected_3[col], original_skew

        for name, transformed in transformations.items():
            if transformed is not None:
                new_skew = ipp.pd.Series(transformed).skew()
                if abs(new_skew) < abs(best_skew) * 0.9:  # Accept if at least 10% better
                    best_method, best_data, best_skew = name, transformed, new_skew

        return best_method, best_data, best_skew

    for col in highly_skewed:
        original_data = df_selected_3[col].copy()  # Save original before dropping

        shift_val = abs(df_selected_3[col].min()) + 1 if (df_selected_3[col] <= 0).any() else 0

        transformations = {
            'log': ipp.np.log1p(df_selected_3[col] + shift_val) if (df_selected_3[col] + shift_val > 0).all() else None,
            'sqrt': ipp.np.sqrt(df_selected_3[col] + shift_val) if (df_selected_3[col] + shift_val >= 0).all() else None,
            'boxcox': None,
            'power': ipp.PowerTransformer(method='yeo-johnson').fit_transform(df_selected_3[[col]]).flatten()
        }

        try:
            if (df_selected_3[col] + shift_val > 0).all():
                transformations['boxcox'] = ipp.boxcox(df_selected_3[col] + shift_val)[0]
        except Exception as e:
            print(f"⚠️ Box-Cox failed for {col}: {e}")

        best_method, best_data, best_skew = transform_column(col, transformations)

        if best_method:
            new_col_name = f"{col}_{best_method}"
            df_selected_3[new_col_name] = best_data  
            df_selected_3.drop(columns=[col], inplace=True)
            transformed_features.append(col)

            # Plot comparison (Overlayed)
            ipp.plt.figure(figsize=(8, 5))
            ipp.sns.histplot(original_data, bins=30, kde=True, color="red", label="Original", alpha=0.5)
            ipp.sns.histplot(df_selected_3[new_col_name], bins=30, kde=True, color="blue", label="Transformed", alpha=0.5)
            ipp.plt.title(f"Overlayed Histogram: {col} (Original vs. {best_method})")
            ipp.plt.legend()
            ipp.plt.show()
        
        try:
            with open(ipp.json_file_path, "r") as file:
                status_data = ipp.json.load(file)
        except FileNotFoundError:
            status_data = {}

        # Set feature_selection to True since we're removing features
        if transformed_features:
            status_data["pre_processing"]["Skew"]["High"]["handling"] = True
            status_data["pre_processing"]["Skew"]["High"]["features"] = transformed_features

        # Write the updated status back to the file
        with open(ipp.json_file_path, "w") as file:
            ipp.json.dump(status_data, file, indent=4)

    print("\n✅ Completed Transformations for Highly Skewed Features.")
    return df_selected_3
























# Function to handle moderately skewed features
def handle_moderate_skew(df_selected_3, moderately_skewed):
    print("\n🔹 Applying Winsorization to Moderately Skewed Features...")
    transformed_features = []  # Track successfully transformed features

    for col in moderately_skewed:
        original_data = df_selected_3[col].copy()
        df_selected_3[col] = ipp.stats.mstats.winsorize(df_selected_3[col], limits=[0.05, 0.05])  # Capping extreme 5% values
        transformed_features.append(col)

        # Plot comparison (Before vs. After)
        fig, axes = ipp.plt.subplots(1, 2, figsize=(12, 5))

        ipp.sns.boxplot(x=original_data, ax=axes[0], color="red")
        axes[0].set_title(f"Original Boxplot: {col}")

        ipp.sns.boxplot(x=df_selected_3[col], ax=axes[1], color="green")
        axes[1].set_title(f"Winsorized Boxplot: {col}")

        ipp.plt.show()


    try:
        with open(ipp.json_file_path, "r") as file:
            status_data = ipp.json.load(file)
    except FileNotFoundError:
        status_data = {}

    # Update JSON for moderate skew
    if transformed_features:
        status_data["pre_processing"]["Skew"]["Moderate"]["handling"] = True
        status_data["pre_processing"]["Skew"]["Moderate"]["features"] = transformed_features

    # Write the updated JSON back to file
    with open(ipp.json_file_path, "w") as file:
        ipp.json.dump(status_data, file, indent=4)

    print("\n✅ Completed Winsorization for Moderately Skewed Features.")
    return df_selected_3
## End of skewed features transformation