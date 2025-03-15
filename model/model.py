import ipp

# Function to segregate skewness values
def segregate_skewness(skewness_values):
    approx_symmetry = skewness_values[(skewness_values >= -0.5) & (skewness_values <= 0.5)].index.tolist()
    slightly_skewed = skewness_values[((skewness_values >= -1) & (skewness_values < -0.5)) | 
                                      ((skewness_values > 0.5) & (skewness_values <= 1))].index.tolist()
    highly_skewed = skewness_values[(skewness_values < -1) | (skewness_values > 1)].index.tolist()
    
    # Print the results
    print("For skewness values between -0.5 and 0.5, the data exhibit approximate symmetry:")
    print(approx_symmetry)

    print("\nSkewness values within the range of -1 and -0.5 (negative skewed) or 0.5 and 1 (positive skewed) indicate slightly skewed data distributions:")
    print(slightly_skewed)

    print("\nData with skewness values less than -1 (negative skewed) or greater than 1 (positive skewed) are considered highly skewed:")
    print(highly_skewed)    

    return approx_symmetry, slightly_skewed, highly_skewed

def get_high_corr_columns(data, threshold):
    corr_matrix = data.corr()
    print(threshold)
    # print(type(corr_matrix))
    high_corr_columns = []

    for row in corr_matrix.index:
        for col in corr_matrix.columns:
            if row != col and abs(corr_matrix.loc[row, col]) > threshold:
                high_corr_columns.append((row, col))

    my_list = []
    # Add correlated pairs to my_list
    for x, y in high_corr_columns:
        my_list.append(x)
        my_list.append(y)

    # Make my_list unique
    unique_list = list(set(my_list))

    # print(unique_list)
    # print(len(unique_list))

    return unique_list

# Define Gradient Descent function
def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m = len(y)
    theta = ipp.np.random.randn(X.shape[1], 1)
    for iteration in range(n_iterations):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradients
    return theta

def corr_LinearRegModel(th_value, data, predictor_variable, use_gradient_descent=False):
    prices = data[predictor_variable].values.reshape(-1, 1)
    unique_list = ipp.get_high_corr_columns(data, th_value)
    unique_list = [col for col in unique_list if col != predictor_variable]

    if predictor_variable not in unique_list:
        print("Success")
        features = data[unique_list]
    else:
        print("Failure")
        exit()

    X_train, X_test, Y_train, Y_test = ipp.train_test_split(features, prices, train_size=0.8, random_state=42)

    if use_gradient_descent:
        # Adding a bias term to the features
        X_train_b = ipp.np.c_[ipp.np.ones((X_train.shape[0], 1)), X_train]
        X_test_b = ipp.np.c_[ipp.np.ones((X_test.shape[0], 1)), X_test]

        theta = gradient_descent(X_train_b, Y_train)
        Y_train_pred = X_train_b.dot(theta)
        Y_test_pred = X_test_b.dot(theta)

        Train = 1 - ipp.np.sum((Y_train_pred - Y_train) ** 2) / ipp.np.sum((Y_train - Y_train.mean()) ** 2)
        Test = 1 - ipp.np.sum((Y_test_pred - Y_test) ** 2) / ipp.np.sum((Y_test - Y_test.mean()) ** 2)
    else:
        # Fitting the linear regression model using scikit-learn
        reg = ipp.LinearRegression()
        model_1 = reg.fit(X_train, Y_train)
        Train = model_1.score(X_train, Y_train)
        Test = model_1.score(X_test, Y_test)

        # Reshape model_1.coef_ to match the shape implied by the indices
        coef_reshaped = model_1.coef_.reshape(-1, 1)
        slope = ipp.pd.DataFrame(coef_reshaped, index=X_train.columns, columns=["Slope"])

        # Using statsmodels for OLS regression
        y = Y_train
        x = ipp.sm.add_constant(X_train)
        mod = ipp.sm.OLS(y, x).fit()

    objects_dict = {
        "Linear_Model": {
            "Threshold value": th_value,
            "Train": Train,
            "Test": Test,
            "OLS_model": mod if not use_gradient_descent else None,
            "Theta": theta if use_gradient_descent else None,
        }
    }

    return objects_dict

import numpy as np

def extract_info_model(mmodel):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    Y_pred = []
    
    for key, model_data in mmodel.items():
        ols_model = model_data["OLS_model"]
        
        # Extracting X_train and Y_train
        X_train.append(ols_model.model.exog)
        Y_train.append(ols_model.model.endog)
        
        # Extracting X_test and Y_test (assuming you have separate test data)
        # Replace with actual X_test and Y_test if available, otherwise skip this part
        X_test.append(ols_model.model.exog)  # Replace with actual X_test
        Y_test.append(ols_model.model.endog)  # Replace with actual Y_test
        
        # Predict using the fitted model on X_test (optional, if needed)
        ols_results = ols_model.predict()  # This will predict on the same dataset used for training
        
        Y_pred.append(ols_results)  # Append predicted values
    
    # Convert lists to numpy arrays (optional, if needed)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    Y_pred = np.array(Y_pred)

    X_train = X_train[0]
    Y_train = Y_train[0]
    X_test = X_test[0]
    Y_test = Y_test[0]
    Y_pred = Y_pred[0]

    
    return X_train, X_test, Y_train, Y_test, Y_pred


# def corr_LinearRegModel(th_value, data, predictor_variable):
#     prices = data[predictor_variable]
#     unique_list = ipp.get_high_corr_columns(data, th_value)
#     unique_list = [col for col in unique_list if col != predictor_variable]
#     # print(unique_list)
#     if predictor_variable not in unique_list:
#         print("Success")
#         features = data[unique_list]
#     else:
#         print("Failure")
#         exit()

#     X_train, X_test, Y_train, Y_test = ipp.train_test_split(features, prices, train_size=0.8, random_state=42)

#     # Fitting the linear regression model
#     reg = ipp.LinearRegression()
#     model_1 = reg.fit(X_train, Y_train)
#     Train = model_1.score(X_train, Y_train)
#     # print("Train data R squared value is :", Train)
#     Test = model_1.score(X_test, Y_test)
#     # print("Test data R squared values is :", Test)

#     slope = ipp.pd.DataFrame(model_1.coef_, index=X_train.columns, columns=["Slope"])

#     y = Y_train
#     x = ipp.sm.add_constant(X_train)
#     mod = ipp.sm.OLS(y, x)

#     objects_dict = {
#         "Linear_Model": {
#             "Threshold value": th_value,
#             "Train": Train,
#             "Test": Test,
#             "OLS_model": mod,
#         }
#     }

#     # print("-------------------------------------------------------------------------------------------------------")
#     return objects_dict

def filter_def(compModel):
    best_model_key = None
    best_test_value = float('-inf')
    best_train_value = float('-inf')

    # Iterate over the compModel dictionary
    for key, model_data in compModel.items():
        test_value = model_data["Test"]
        train_value = model_data["Train"]

        # Compare the test and train values with the best values found so far
        if test_value > best_test_value or train_value > best_train_value:
            # Update the best model and its test/train values
            best_model_key = key
            best_test_value = test_value
            best_train_value = train_value

    # Remove all models except the best one
    for key in list(compModel.keys()):
        if key != best_model_key:
            del compModel[key]

    # Print the best model
    print("Best Model:")
    print("Key:", best_model_key)
    print("Value:", compModel[best_model_key])

    return compModel  # Return the updated compModel dictionary

def choose_best_transformation(data, list_column_name):
    # Helper function to apply transformations
    def apply_transformations(column):
        transformations = {}
        transformations['original'] = column
        # Log Transformation
        if (column <= 0).any():
            transformations['log'] = ipp.np.log1p(column - column.min() + 1)
        else:
            transformations['log'] = ipp.np.log1p(column)
        # Square Root Transformation
        if (column >= 0).all():  # Ensure no negative values for sqrt
            transformations['sqrt'] = ipp.np.sqrt(column)
        else:
            transformations['sqrt'] = ipp.np.nan * len(column)
        # Box-Cox Transformation (only positive data)
        if (column > 0).all():  # Ensure all values are positive
            transformations['box_cox'], _ = ipp.stats.boxcox(column)
        else:
            transformations['box_cox'] = ipp.np.nan * len(column)
        # Yeo-Johnson Transformation
        transformations['yeo_johnson'], _ = ipp.stats.yeojohnson(column)
        # Reciprocal Transformation
        if (column <= 0).any():
            transformations['reciprocal'] = 1 / (column - column.min() + 1)
        else:
            transformations['reciprocal'] = 1 / (column + 1)
        # Exponential Transformation
        transformations['exp'] = ipp.np.exp(column - column.min())
        return transformations
    
    best_transformations = {}
    
    for column in list_column_name:
        col_data = data[column]
        transformations = apply_transformations(col_data)
        
        # Plot the distributions
        fig, axes = ipp.plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f"Transformations for {column}", fontsize=16)
        for ax, (trans_name, trans_data) in zip(axes.flatten(), transformations.items()):
            if not ipp.np.any(ipp.np.isnan(trans_data)):
                ipp.sns.histplot(trans_data, bins=30, kde=True, ax=ax)
                ax.set_title(f"{trans_name} transformation")
            else:
                ax.set_title(f"{trans_name} transformation (not applicable)")
        ipp.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        ipp.plt.show()
        
        # Calculate skewness for each transformation
        skewness = {trans_name: ipp.pd.Series(trans_data).skew() for trans_name, trans_data in transformations.items() if not ipp.np.any(ipp.np.isnan(trans_data))}
        print(f"Skewness for different transformations of {column}:\n", skewness)
        
        # Select the best transformation (closest to 0 skewness)
        best_trans_name = min(skewness, key=lambda k: abs(skewness[k]))
        best_transformations[column] = transformations[best_trans_name]
        
        # Compare with original and replace if there's an improvement
        if abs(skewness[best_trans_name]) < abs(col_data.skew()):
            new_column_name = f"{column}_{best_trans_name}"
            data[new_column_name] = best_transformations[column]
            print(f"Replaced {column} with {best_trans_name} transformation. Renamed to {new_column_name}.\n")
        else:
            print(f"No improvement for {column}. Keeping original.\n")
    
    # Drop original columns
    data.drop(columns=list_column_name, inplace=True)
    
    return data

def apply_log_transformation(data, columns):
    # Copy the original data to avoid modifying it directly
    transformed_data = data.copy()
    
    for column in columns:
        skewness = ipp.skew(transformed_data[column])
        if skewness > 0.5:  # Positively skewed
            transformed_column = f'log_{column}'
            # Apply log transformation
            transformed_data[transformed_column] = ipp.np.log1p(transformed_data[column])
            # Recalculate skewness after transformation
            new_skewness = ipp.skew(transformed_data[transformed_column])
            if abs(new_skewness) < abs(skewness):
                # If the new skewness is reduced, keep the transformed column
                transformed_data = transformed_data.drop(columns=[column])
            else:
                # Otherwise, drop the transformed column
                transformed_data = transformed_data.drop(columns=[transformed_column])
        elif skewness < -0.5:  # Negatively skewed
            transformed_column = f'exp_{column}'
            # Apply exponential transformation (and add 1 to avoid issues with zero values)
            transformed_data[transformed_column] = ipp.np.expm1(transformed_data[column])
            # Recalculate skewness after transformation
            new_skewness = ipp.skew(transformed_data[transformed_column])
            if abs(new_skewness) < abs(skewness):
                # If the new skewness is reduced, keep the transformed column
                transformed_data = transformed_data.drop(columns=[column])
            else:
                # Otherwise, try log transformation and compare
                transformed_column_log = f'log_{column}'
                transformed_data[transformed_column_log] = ipp.np.log1p(transformed_data[column])
                new_skewness_log = ipp.skew(transformed_data[transformed_column_log])
                if abs(new_skewness_log) < abs(skewness):
                    # If log transformation reduces skewness, keep it
                    transformed_data = transformed_data.drop(columns=[column, transformed_column])
                else:
                    # Otherwise, drop both transformations
                    transformed_data = transformed_data.drop(columns=[transformed_column, transformed_column_log])
    
    return transformed_data

def analysee_model(model):
    for key, model_data in model.items():
        try:
            # Extract necessary information from the model's dictionary
            ols_model = model_data["OLS_model"]
            
            # Extract training and test data from the fitted model
            X_train, X_test, y_train, y_test, y_pred = ipp.extract_info_model(model)
            
            # Get the fitted values and residuals directly from ols_model
            fitted_values = ols_model.fittedvalues
            residuals = ols_model.resid
            
            # Residuals vs. Fitted Values Plot
            ipp.plt.scatter(fitted_values, residuals)
            ipp.plt.xlabel('Fitted values')
            ipp.plt.ylabel('Residuals')
            ipp.plt.title(f'Residuals vs Fitted Values for {key}')
            ipp.plt.axhline(y=0, color='r', linestyle='--')
            ipp.plt.show()

            # Breusch-Pagan Test
            bp_test = ipp.sms.het_breuschpagan(residuals, ols_model.model.exog)
            bp_test_statistic = bp_test[0]
            bp_p_value = bp_test[1]

            print(f'{key} - Breusch-Pagan test statistic: {bp_test_statistic}')
            print(f'{key} - P-value: {bp_p_value}')

            # White’s Test
            white_test = ipp.smd.het_white(residuals, ols_model.model.exog)
            white_test_statistic = white_test[0]
            white_p_value = white_test[1]

            print(f'{key} - White test statistic: {white_test_statistic}')
            print(f'{key} - P-value: {white_p_value}')

            # Residual Plot for Heteroscedasticity
            abs_residuals = abs(residuals)
            ipp.plt.scatter(fitted_values, abs_residuals)
            ipp.plt.xlabel('Fitted values')
            ipp.plt.ylabel('Absolute Residuals')
            ipp.plt.title(f'Absolute Residuals vs Fitted Values for {key}')
            ipp.plt.show()

            # Conclusion
            conclusion = "Conclusion: "
            if bp_p_value < 0.05 or white_p_value < 0.05:
                conclusion += "There is evidence of heteroscedasticity."
            else:
                conclusion += "No evidence of heteroscedasticity detected."

            print(conclusion)

        except KeyError as e:
            print(f"KeyError: {e} - Check the structure of compModel for {key}")
        except Exception as e:
            print(f"An unexpected error occurred for {key}: {e}")
    
    return fitted_values,residuals


# def extract_info_model(mmodel):
#     for key, model_data in mmodel.items():
#         # Extract necessary information from the model's dictionary
#         ols_model = model_data["OLS_model"]
#         # train_value = model_data["Train"]
#         # test_value = model_data["Test"]
    
#         # Get X_train and Y_train
#         X_train = ols_model.exog
#         Y_train = ols_model.endog
    
#         # Get X_test and Y_test
#         X_test = ols_model.exog
#         Y_test = ols_model.endog
#         ols_results = ols_model.fit()
#         Y_pred = ols_results.predict(X_test)
#     return(X_train,X_test,Y_train,Y_test,Y_pred)

def add_model_info(model):
    model_names = []
    train_scores = []
    test_scores = []
    for model_name, model_data in model.items():
        model_names.append(model_data['Threshold value'])
        train_scores.append(model_data['Train'])
        test_scores.append(model_data['Test'])

    # Creating DataFrame
    Eval = ipp.pd.DataFrame({
        'Model': model_names,
        'Train Score': train_scores,
        'Test Score': test_scores
    })

    return Eval

def corr_Linear_RegModel(th_value, data, predictor_variable,split):
    prices = data[predictor_variable]
    unique_list = ipp.get_high_corr_columns(data, th_value)
    unique_list = [col for col in unique_list if col != predictor_variable]
    # print(unique_list)
    if predictor_variable not in unique_list:
        print("Success")
        features = data[unique_list]
    else:
        print("Failure")
        exit()

    X_train, X_test, Y_train, Y_test = ipp.train_test_split(features, prices, train_size=split, random_state=10)

    # Fitting the linear regression model
    reg = ipp.LinearRegression()
    model_1 = reg.fit(X_train, Y_train)
    Train = model_1.score(X_train, Y_train)
    # print("Train data R squared value is :", Train)
    Test = model_1.score(X_test, Y_test)
    # print("Test data R squared values is :", Test)

    slope = ipp.pd.DataFrame(model_1.coef_, index=X_train.columns, columns=["Slope"])

    y = Y_train
    x = ipp.sm.add_constant(X_train)
    mod = ipp.sm.OLS(y, x)

    objects_dict = {
        "Linear_Model": {
            "Threshold value": th_value,
            "Train": Train,
            "Test": Test,
            # "slope_df": slope,
            "OLS_model": mod,
        }
    }

    return objects_dict

## -----------------------------------------------------------------------------
# Function for REGRESSION MODEL

def Reg_model(df, predictor_variable, name):
    # Extracting features (X) and target (y)
    X = df.drop(columns=[predictor_variable])
    y = df[predictor_variable]

    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = ipp.train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initializing and training the model
    model = ipp.LinearRegression()
    model.fit(X_train, y_train)
    
    # Predicting values for train and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculating R2 score for both train and test
    train_score = ipp.r2_score(y_train, y_train_pred)
    test_score = ipp.r2_score(y_test, y_test_pred)
    
    # print("----------------------TESTING PURPOSE--------------------------------------")
    # print(train_score, test_score)
    # print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
    # print("y_train shape:", y_train.shape, "y_test shape:", y_test.shape)
    # print("Unique values in y_test:", np.unique(y_test))
    # y_pred = model.predict(X_test)
    # print("Predictions:", y_pred)
    # print("Actual y_test:", y_test.values)
    # print("------------------------------------------------------------------")
    
    # Return the results
    return train_score, test_score, ipp.r2_score(y_test, y_test_pred), model

# Function to update the JSON structure with model results
def interModel(df, predictor_variable, name):
    # Call the Reg_model function and store the results
    model_dir = "model_dump"
    json_file_path = ipp.os.path.join(model_dir, "status.json")
    train_score, test_score, r2_score_value, model = ipp.Reg_model(df, predictor_variable, name)

    if ipp.os.path.exists(json_file_path):
        with open(json_file_path, "r") as json_file:
            json_data = ipp.json.load(json_file)
    else:
        print("issue with json file ")
        ipp.sys.exit()
    
    # Save the model and get the file path
    model_file = ipp.save_model(model, name)
    
    # Store results under the given model name inside "modeling"
    json_data["modeling"][name] = {
        "train": train_score,
        "test": test_score,
        "r2_score": r2_score_value,
        "model": model_file  # Storing only the filename
    }

    json_data["modeling"]["efficiency"] = False # Set to True if efficiency is achieved
    
    # Print updated JSON (optional)
    # Save the updated JSON data back to status.json
    with open(json_file_path, "w") as json_file:
        ipp.json.dump(json_data, json_file, indent=4)
    # print(json.dumps(json_data, indent=4))



# Function to pickle the model and return the file path
def save_model(model, name):
    timestamp = ipp.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{name}_{timestamp}.pkl"
    file_path = ipp.os.path.join(ipp.model_dir, file_name)
    
    with open(file_path, "wb") as f:
        ipp.pickle.dump(model, f)

    return file_name  # Returning only the file name to store in JSON