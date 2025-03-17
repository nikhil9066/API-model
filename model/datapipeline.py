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


## adding the CFS part
def CFS_update(lower_threshold, high_loss, low_loss):
    """
    Updates the status.json file by modifying the "CFS" section when feature selection occurs.

    Parameters:
    - status_file (str): Path to the status.json file.
    - lower_threshold (float): The threshold used for low correlation feature removal.
    - high_loss (dict): Features removed due to high correlation (above 0.90).
    - low_loss (dict): Features removed due to low correlation (below lower_threshold).
    """
    
    # If no features are removed, do not update the file
    if not high_loss and not low_loss:
        # print("No features removed. Skipping status.json update.")
        return

    try:
        # Load existing status.json
        with open(ipp.json_file_path, "r") as file:
            status_data = ipp.json.load(file)
    except FileNotFoundError:
        status_data = {}

    # Ensure "pre_processing" and "CFS" sections exist
    if "pre_processing" not in status_data:
        status_data["pre_processing"] = {}
    if "CFS" not in status_data["pre_processing"]:
        status_data["pre_processing"]["CFS"] = {"feature_selection": False}

    # Set feature_selection to True since we're removing features
    status_data["pre_processing"]["CFS"]["feature_selection"] = True

    # Create an entry with timestamp
    timestamp = ipp.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_entry = {
        "timestamp": timestamp,
        "lower_threshold": lower_threshold,
        "High_loss": high_loss,
        "Low_loss": low_loss
    }

    # Append new entry without overwriting previous ones
    if "history" not in status_data["pre_processing"]["CFS"]:
        status_data["pre_processing"]["CFS"]["history"] = []
    
    status_data["pre_processing"]["CFS"]["history"].append(update_entry)

    # Write the updated status back to the file
    with open(ipp.json_file_path, "w") as file:
        ipp.json.dump(status_data, file, indent=4)

    # print("✅ status.json updated successfully!")

def correlation_feature_selection(df, lower_threshold, target_variable="medv"):
    HIGHER_THRESHOLD = 0.90  # Constant for higher threshold
    correlation_matrix = df.corr()  # Compute correlation matrix
    target_corr = correlation_matrix[target_variable]  # Correlations with target

    # Identifying features to drop due to low correlation with the predictive variable
    low_correlation_features = {
        col: round(target_corr[col], 4)
        for col in target_corr.index
        if abs(target_corr[col]) < lower_threshold and col != target_variable
    }

    # Identifying highly correlated features (above 0.90) and keeping only one from each pair
    high_correlation_features = {}
    processed = set()

    for col in correlation_matrix.columns:
        if col == target_variable:
            continue
        for row in correlation_matrix.columns:
            if row != col and row != target_variable and (row, col) not in processed:
                corr_value = abs(correlation_matrix.loc[col, row])
                if corr_value > HIGHER_THRESHOLD:
                    # Drop the feature that has lower correlation with the target variable
                    if abs(target_corr[col]) < abs(target_corr[row]):
                        high_correlation_features[col] = round(corr_value, 4)
                    else:
                        high_correlation_features[row] = round(corr_value, 4)
                processed.add((col, row))

    # Logging features being dropped
    # print(high_correlation_features)
    # if high_correlation_features:
    #     print("High_loss = {", ", ".join(f"{k}: {v}" for k, v in high_correlation_features.items()), "} (Highly correlated features dropped)")

    # if low_correlation_features:
    #     print("Low_loss = {", ", ".join(f"{k}: {v}" for k, v in low_correlation_features.items()), "} (Low correlation features dropped)")

    # Drop the selected features
    df_selected = df.drop(columns=list(high_correlation_features.keys()) + list(low_correlation_features.keys()))
    CFS_update(lower_threshold, high_correlation_features, low_correlation_features)
    return df_selected

## CFS ends here