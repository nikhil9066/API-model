import ipp

class OutlierRemover(ipp.BaseEstimator, ipp.TransformerMixin):
    def __init__(self, method='iqr'):
        self.method = method
        
    def fit(self, X, y=None):
        if self.method == 'iqr':
            self.desc = X.describe()
        return self
    
    def transform(self, X, y=None):
        if self.method == 'iqr':
            Q1 = self.desc.loc['25%']
            Q3 = self.desc.loc['75%']
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ipp.pd.DataFrame()
            X_no_outliers = X.copy()
            for col in X.columns:
                col_outliers = X_no_outliers[(X_no_outliers[col] < lower_bound[col]) | (X_no_outliers[col] > upper_bound[col])]
                outliers = ipp.pd.concat([outliers, col_outliers])
                X_no_outliers = X_no_outliers.drop(col_outliers.index)
            return X_no_outliers
        
        elif self.method == 'sd3':
            outliers = ipp.pd.DataFrame()
            X_no_outliers = X.copy()
            for col in X.columns:
                if ipp.pd.api.types.is_numeric_dtype(X[col]):
                    mean = X[col].mean()
                    std_dev = X[col].std()
                    lower_bound = mean - 3 * std_dev
                    upper_bound = mean + 3 * std_dev
                    col_outliers = X_no_outliers[(X_no_outliers[col] < lower_bound) | (X_no_outliers[col] > upper_bound)]
                    outliers = ipp.pd.concat([outliers, col_outliers])
                    X_no_outliers = X_no_outliers.drop(col_outliers.index)
            return X_no_outliers
        
        elif self.method == 'zscore':
            z_scores = (X - X.mean()) / X.std()
            threshold = 3
            outliers = (ipp.np.abs(z_scores) > threshold).any(axis=1)
            return X[~outliers]
        
        elif self.method == 'percentile':
            lower_percentile = 0.01
            upper_percentile = 0.99
            outliers = ipp.pd.DataFrame()
            X_no_outliers = X.copy()
            for col in X.columns:
                if ipp.pd.api.types.is_numeric_dtype(X[col]):
                    lower_bound = X[col].quantile(lower_percentile)
                    upper_bound = X[col].quantile(upper_percentile)
                    col_outliers = X_no_outliers[(X_no_outliers[col] < lower_bound) | (X_no_outliers[col] > upper_bound)]
                    outliers = ipp.pd.concat([outliers, col_outliers])
                    X_no_outliers = X_no_outliers.drop(col_outliers.index)
            return X_no_outliers
