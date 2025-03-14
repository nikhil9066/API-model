import ipp
compModel = {}
model_Eval = ipp.pd.DataFrame()
ipp.warnings.filterwarnings("ignore")

df = ipp.pd.read_csv('/Users/nikhilprao/Documents/Data/Boston.csv', index_col=0)
df.reset_index(drop=True)
df.isnull().sum()

# applying the method
nan_in_df = df.isnull().sum().any()
print(nan_in_df)

df.info()

print(type(df))

df.describe()

### Ask user which is Predictive column

# Display column names to the user
print("Available predictor variables:")
for idx, col in enumerate(df.columns):
    print(f"{idx + 1}. {col}")

# Ask the user to choose a predictor variable
selected_index = int(input("Enter the index of the predictor variable you want to choose: ")) - 1

# Validate user input
if 0 <= selected_index < len(df.columns):
    pattern = df.columns[selected_index]
    print(f"Selected predictor variable: {pattern}")
else:
    print("Invalid index selected. Please choose a valid index.")


predictor_variable = df.filter(regex=f'^{pattern}').columns[0]

## Model definition
def interModelWork(data, name, compModel):

    predictor_variable = data.filter(regex=f'^{pattern}').columns[0]
    new_model = ipp.corr_LinearRegModel(0, data, predictor_variable)

    next_model_key = f'Linear_Model_{name}'
    compModel[next_model_key] = new_model['Linear_Model']
    print(new_model['Linear_Model'])
    compModel[next_model_key]['Threshold value'] = name
    print(compModel)
    ipp.filter_def(compModel)
    
    return new_model

### Model starts here
x = interModelWork(df,'Original',compModel)
print(x)
model_Eval = ipp.pd.concat([model_Eval,ipp.add_model_info(x)],ignore_index=True)
print(model_Eval)

### Pipeline for OUTLIERS
# Create a pipeline for each method
pipeline_iqr = ipp.Pipeline([
    ('outlier_remover', ipp.OutlierRemover(method='iqr'))
])

pipeline_sd3 = ipp.Pipeline([
    ('outlier_remover', ipp.OutlierRemover(method='sd3'))
])

pipeline_percentile = ipp.Pipeline([
    ('outlier_remover', ipp.OutlierRemover(method='sd2'))
])

pipeline_zscore = ipp.Pipeline([
    ('outlier_remover', ipp.OutlierRemover(method='zscore'))
])

pipeline_percentile = ipp.Pipeline([
    ('outlier_remover', ipp.OutlierRemover(method='percentile'))
])
# Apply the pipelines to your DataFrame
df_iqr_no_outliers = pipeline_iqr.fit_transform(df)
df_sd3_no_outliers = pipeline_sd3.fit_transform(df)
df_z3_no_outliers = pipeline_zscore.fit_transform(df)
df_per_no_outliers = pipeline_percentile.fit_transform(df)

# Output the results
print("IQR Method:")
print("Number of outliers detected:", len(df)-len(df_iqr_no_outliers))
print("Number of rows after removing outliers:", len(df_iqr_no_outliers))

print("\n3 Standard Deviations Method:")
print("Number of outliers detected:", len(df)-len(df_sd3_no_outliers))
print("Number of rows after removing outliers:", len(df_sd3_no_outliers))

print("\nZ-Score Method:")
print("Number of outliers detected:", len(df)-len(df_z3_no_outliers))
print("Number of rows after removing outliers:", len(df_z3_no_outliers))

print("\nPercentile Method:")
print("Number of outliers detected:", len(df)-len(df_per_no_outliers))
print("Number of rows after removing outliers:", len(df_per_no_outliers))


# Collect the number of rows after removing outliers
rows_after_removal = {
    'iqr': len(df_iqr_no_outliers),
    'sd3': len(df_sd3_no_outliers),
    'zscore': len(df_z3_no_outliers),
    'percentile': len(df_per_no_outliers)
}

# Find the median value
median_rows = ipp.np.median(list(rows_after_removal.values()))

# Find the method that has the number of rows closest to the median
chosen_method = min(rows_after_removal, key=lambda k: abs(rows_after_removal[k] - median_rows))

print(f"\nChosen method: {chosen_method}")
print(f"Number of rows after removing outliers using {chosen_method} method: {rows_after_removal[chosen_method]}")

# Set the DataFrame to the one chosen by the median value
if chosen_method == 'iqr':
    data = df_iqr_no_outliers
elif chosen_method == 'sd3':
    data = df_sd3_no_outliers
elif chosen_method == 'zscore':
    data = df_z3_no_outliers
else:
    data = df_per_no_outliers

# Now `data` contains the DataFrame with outliers removed according to the chosen method
print("\nDataFrame stored in 'data' variable for further analysis or modeling.")
print(data.describe())

ipp.plot_outliers(df)

fig, axs = ipp.plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for col, value in df.items() :
    ipp.sns.distplot(value, ax=axs[index])
    index += 1
ipp.plt. tight_layout (pad=0.5, w_pad=0.7, h_pad=5.0)

### NEED to ad this box plot to all features to show outliers
#box plot to all features to show outliers
fig, axs = ipp.plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for col, value in data.items():
    ipp.sns.boxplot(y=col, data=data, ax=axs[index])
    index += 1
ipp.plt.tight_layout (pad=0.5, w_pad=0.7, h_pad=5.0)

ipp.plot_outliers(data)

x = interModelWork(data,'sd3',compModel)
model_Eval = ipp.pd.concat([model_Eval,ipp.add_model_info(x)],ignore_index=True)
print(model_Eval)

ipp.plot_all_numerical_columns(df)

corr_mat=data.corr()
correlation_matrix = corr_mat
print(type(corr_mat))

ipp.corrplot(corr_mat)

ipp.sns.pairplot(data, kind='scatter',plot_kws={'alpha': 0.4})
# <center>END OF SECTION - 1</center>







## Feature scaling 
# df_dropped = data.drop(columns=[predictor_variable])
ipp.plt.hist(data.values, label='Before Scaling')
ipp.plt.show()

scaler = ipp.StandardScaler ()
x = scaler.fit_transform(data)
df_scaled = ipp.pd.DataFrame(x, columns=data.columns)

ipp.plt.hist(df_scaled, label='After Scaling')
ipp.plt.show()

df_scaled[predictor_variable] = data[predictor_variable].values
x = interModelWork(df_scaled,'AF_Stand',compModel)
model_Eval = ipp.pd.concat([model_Eval,ipp.add_model_info(x)],ignore_index=True)
print(model_Eval)
data = df_scaled

## Splitting dataset into train and test data
### Model with Corr Model 
compModel = {}
threshold_list = [0.0,0.3,0.4,0.5,0.6,0.7]
for i in threshold_list:
    print(i)
    model_dict = {}
    model_dict = ipp.corr_LinearRegModel(i, data, predictor_variable)
    compModel.update({f"Model_{i}": model_dict["Linear_Model"]})
model_Eval = ipp.pd.concat([model_Eval,ipp.add_model_info(compModel)],ignore_index=True)
ipp.filter_def(compModel)

## Handling the skewness
skewness_values = data.skew()
approx_symmetry, slightly_skewed, highly_skewed = ipp.segregate_skewness(skewness_values)

### slightly_skewed Tranformation
data = ipp.apply_log_transformation(data.copy(), slightly_skewed)
# slightly_skewed Model
x = interModelWork(data,'slightly_skewed',compModel)
model_Eval = ipp.pd.concat([model_Eval,ipp.add_model_info(x)],ignore_index=True)
print(model_Eval)

### Highly_skewed Tranformation
data = ipp.choose_best_transformation(data.copy(), highly_skewed)
# Highly skew model 
x = interModelWork(data,'Highly_skewed',compModel)
model_Eval = ipp.pd.concat([model_Eval,ipp.add_model_info(x)],ignore_index=True)
print(model_Eval)


# Need to loop once more to clear off the skwness
skewness_values = data.skew()
approx_symmetry, slightly_skewed, highly_skewed = ipp.segregate_skewness(skewness_values)
ipp.filter_def(compModel)

# <center>END OF SECTION - 2 </center>










fitted_values,residuals = ipp.analyse_model(compModel)
X_train,X_test,y_train,y_test,y_pred = ipp.extract_info_model(compModel)

ipp.actVpre(y_test,y_pred)
# Clear the current figure
ipp.pylab.figure()

ipp.sns.displot(residuals, bins=40, kde=True)
# Clear the current figure
ipp.pylab.figure()

# Q-Q Plot for Residuals
ipp.stats.probplot(residuals, dist="norm", plot=ipp.pylab)
ipp.pylab.title('Q-Q Plot of Residuals')
ipp.pylab.show()

### Calculate MAE and MSE
mae = ipp.mean_absolute_error(y_test, y_pred)
mse = ipp.mean_squared_error(y_test, y_pred)
rmse = ipp.np.sqrt(mse)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

## Changing the split and checking
splits = [0.5, 0.6, 0.7]
predictor_variable = data.filter(regex=f'^{pattern}').columns[0]
for split in splits:
    compModel.update(ipp.corr_Linear_RegModel(0, data, predictor_variable,split))

    split_int = int(split * 100)
    complement_int = int((1 - split) * 100)
    name = {split_int: complement_int}

    compModel['Linear_Model']['Threshold value'] = name
    ipp.filter_def(compModel)
    print(compModel)
    model_Eval = ipp.pd.concat([model_Eval,ipp.add_model_info(compModel)],ignore_index=True)

# visulize the model performance scores
print(model_Eval)
print(compModel)

ipp.plot_model_eval(model_Eval)

# ipp.reg.score(X_train,y_train)
