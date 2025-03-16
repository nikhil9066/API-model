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

import json
import os

def update_outlier_status_json(model_dir, outliers_removed, method_used):
    """
    Updates the status.json file with the outlier detection status and details.
    
    Parameters:
        model_dir (str): Directory where the model and status.json are stored.
        outliers_removed (int): Number of outliers removed.
        method_used (str): The method used for outlier detection and removal.
    """
    
    

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
        status_data = json.load(f)
    
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
        json.dump(status_data, f, indent=4)
    
    return df_filtered