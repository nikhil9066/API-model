import ipp

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

# def all_LinearRegModel(data, predictor_variable):

#     prices = data[predictor_variable]
#     features = data.drop(predictor_variable, axis=1)

#     # Splitting the data into training and testing sets
#     X_train, X_test, Y_train, Y_test = ipp.train_test_split(features, prices, train_size=0.8, random_state=10)

#     # Fitting the linear regression model
#     reg = ipp.LinearRegression()
#     model_1 = reg.fit(X_train, Y_train)
#     print("Train data R squared value is :", model_1.score(X_train, Y_train))
#     print("Test data R squared values is :", model_1.score(X_test, Y_test))

#     # Storing the coefficients in a DataFrame
#     slope = ipp.pd.DataFrame(model_1.coef_, index=X_train.columns, columns=["Slope"])

#     # Fitting OLS model
#     y = Y_train
#     x = ipp.sm.add_constant(X_train)
#     mod = ipp.sm.OLS(y, x)
#     # res = mod.fit()
#     # res_summary = res.summary()

#     # Storing in a dictionary
#     objects_dict = {
#         "Linear_Model": {
#             "slope_df": slope,
#             "OLS_model": mod,
#             }
#         }

#     print("-------------------------------------------------------------------------------------------------------")
#     # print(type(slope))
#     # print(type(mod))
#     # print(type(res))

#     # # Printing out the dictionary
#     # print(objects_dict)
    
#     return objects_dict

def corr_LinearRegModel(th_value, data, predictor_variable):
    prices = data[predictor_variable]
    unique_list = ipp.get_high_corr_columns(data, th_value)
    unique_list = [col for col in unique_list if col != predictor_variable]

    if predictor_variable not in unique_list:
        print("Success")
        features = data[unique_list]
    else:
        print("Failure")
        exit()

    X_train, X_test, Y_train, Y_test = ipp.train_test_split(features, prices, train_size=0.8, random_state=10)

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

    # print("-------------------------------------------------------------------------------------------------------")
    return objects_dict

def filter_def(compModel):
    
    best_model_key = None
    best_test_value = float('-inf')

    # Iterate over the compModel dictionary
    for key, model_data in compModel.items():
        test_value = model_data["Test"]
        train_value = model_data["Train"]
    
        # Compare the test value with the best test value found so far
        if test_value > best_test_value:
            # Update the best model and its test value
            best_model_key = key
            best_test_value = test_value

    # Remove all models except the best one
    for key in list(compModel.keys()):
        if key != best_model_key:
            del compModel[key]

    # Print the best model
    print("Best Model:")
    print("Key:", best_model_key)
    print("Value:", compModel[best_model_key])

    return dict