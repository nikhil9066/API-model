import matplotlib.pyplot as plt
import seaborn as sns
import ipp

# def plot_all_numerical_columns(data):
#     numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
#     for column in numerical_columns:
#         # setting plot size
#         plt.figure(figsize=(20, 7))
#         sns.set(style="whitegrid")

#         # plot using matplotlib
#         plt.subplot(1, 2, 1)   # arguments no of rows, columns, and index
#         plt.hist(data[column], bins=50, ec="black", color="#FFEB3B")  # ec- edge color
#         plt.xlabel(column.capitalize(), fontsize=16)
#         plt.ylabel("Frequency", fontsize=16)
#         plt.title(f"{column.capitalize()} Distribution (Matplot)", fontsize=16)

#         # plot using seaborn
#         plt.subplot(1, 2, 2)
#         sns.distplot(data[column], bins=50, color="#512DA8")
#         plt.xlabel(column.capitalize(), fontsize=16)
#         plt.ylabel("Frequency", fontsize=16)
#         plt.title(f"{column.capitalize()} Distribution (Seaborn)", fontsize=16)
#         plt.show()

def plot_all_numerical_columns(data):
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    n_cols = 2  # number of columns in the plot grid for each column
    n_rows = len(numerical_columns)  # number of rows needed

    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(20, 5 * n_rows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, column in enumerate(numerical_columns):
        # Plot using matplotlib
        axes[4 * i].hist(data[column], bins=50, ec="black", color="#FFEB3B")
        axes[4 * i].set_xlabel(column.capitalize(), fontsize=16)
        axes[4 * i].set_ylabel("Frequency", fontsize=16)
        axes[4 * i].set_title(f"{column.capitalize()} Distribution (Matplot)", fontsize=16)

        # Plot using seaborn
        sns.histplot(data[column], bins=50, color="#512DA8", ax=axes[4 * i + 1], kde=True)
        axes[4 * i + 1].set_xlabel(column.capitalize(), fontsize=16)
        axes[4 * i + 1].set_ylabel("Frequency", fontsize=16)
        axes[4 * i + 1].set_title(f"{column.capitalize()} Distribution (Seaborn)", fontsize=16)

        # Plot KDE using matplotlib
        sns.kdeplot(data[column], ax=axes[4 * i + 2], color="#FFEB3B")
        axes[4 * i + 2].set_xlabel(column.capitalize(), fontsize=16)
        axes[4 * i + 2].set_ylabel("Density", fontsize=16)
        axes[4 * i + 2].set_title(f"{column.capitalize()} KDE (Matplot)", fontsize=16)

        # Plot KDE using seaborn
        sns.kdeplot(data[column], ax=axes[4 * i + 3], color="#512DA8")
        axes[4 * i + 3].set_xlabel(column.capitalize(), fontsize=16)
        axes[4 * i + 3].set_ylabel("Density", fontsize=16)
        axes[4 * i + 3].set_title(f"{column.capitalize()} KDE (Seaborn)", fontsize=16)

    plt.tight_layout()
    plt.show()

def corrplot(mat):

    plt.figure(figsize=(15,10))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    sns.heatmap(mat,annot=True,annot_kws={"size":14},linewidth=.5)

def get_high_corr_columns(data, threshold):
    corr_matrix = data.corr()
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

def plot_scatter_pairs_and_store(data, matrix, val):
    correlated_pairs = []  # Initialize an empty list to store the correlated pairs

    # Define a function to plot scatter plots for specific columns
    def plot_scatter(column1, column2):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=column1, y=column2, data=data, color="green", alpha=0.5)
        plt.xlabel(f"{column1.capitalize()}", fontsize=12)
        plt.ylabel(f"{column2.capitalize()}", fontsize=12)
        plt.title(f"{column1.capitalize()} vs {column2.capitalize()}", fontsize=12)
        # plt.show()

    # Iterate through the upper triangle of the correlation matrix
    for i in range(len(matrix.columns)):
        for j in range(i+1, len(matrix.columns)):
            correlation = abs(matrix.iloc[i, j])
            if correlation >= val:  # Check if correlation exceeds the threshold
                correlated_pairs.append((data.columns[i], data.columns[j]))  # Store correlated pair
                plot_scatter(data.columns[i], data.columns[j])  # Plot scatter plot for the pair

    return correlated_pairs

def plot_corr_scatter(data,correlated_pairs):
    for i in correlated_pairs:
        column1, column2 = i
        sns.lmplot(x=column1,y=column2,data=data,height=8)
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.show()

def actVpre(Y_test,Y_pred):
    # Creating the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_test, Y_pred, color='blue', edgecolor='k', alpha=0.7)
    plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linewidth=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid(True)
    plt.show()

def plot_model_eval(model_Eval):
    # Convert model labels to numerical indices for plotting
    numeric_labels = list(range(len(model_Eval)))
    plt.figure(figsize=(10, 6))
    plt.plot(numeric_labels, model_Eval['Train Score'], label='Train Score', marker='o')
    plt.plot(numeric_labels, model_Eval['Test Score'], label='Test Score', marker='o')
    plt.title('Model Evaluation')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.ylim(0.4, 0.9)  # Setting y-axis limits for better visualization
    plt.legend()
    plt.grid(True)
    plt.xticks(numeric_labels, model_Eval['Model'], rotation=45)
    plt.show()