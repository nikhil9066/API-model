import ipp

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
    return train_score, test_score, test_score, model

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



def log_model_result(result_entry: dict):
    import json, os

    if not os.path.exists(ipp.json_file_path):
        raise FileNotFoundError(f"{ipp.json_file_path} not found")

    with open(ipp.json_file_path, "r") as f:
        json_data = json.load(f)

    section = result_entry.get("section")
    model_results = result_entry.get("models", {})
    flags = result_entry.get("flags", {})

    # Exit if nothing to log
    if section not in ["modeling", "hyperparameter_tuning"] or not model_results:
        return  # No-op

    if section not in json_data:
        json_data[section] = {}

    for model_name, metrics in model_results.items():
        # If required keys are missing, skip
        if not {"train", "test", "r2_score"}.issubset(metrics.keys()):
            continue

        json_data[section][model_name] = metrics

    # Apply section-level flags if any
    for key, value in flags.items():
        json_data[section][key] = value

    with open(ipp.json_file_path, "w") as f:
        json.dump(json_data, f, indent=4)
