# Importing libraries for data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pickle
from datetime import datetime
import shutil
import os
import json
import os
import sys
import warnings
import plotly

# Importing libraries for statistical modeling and machine learning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.stats.diagnostic as smd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_goldfeldquandt 
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from scipy import stats
from scipy.stats import skew
from scipy.stats import zscore
from scipy.stats import boxcox
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Importing custom modules
from datapipeline import *  # Import preprocessing pipeline
from model import *  # Import ML models
from plot import *  # Import visualization functions
### from regression import *  # Import regression models
# from preprocessing_pipeline import *  # Import additional preprocessing
from setup import *  # Import setup functions
from autoregressx import *  # Import autoregressive models

# Set plot style
plt.style.use('ggplot')


## Paths and global variables
model_dir = "model_dump"
json_file_path = os.path.join(model_dir, "status.json")
data_folder = "data"


from cryptography.fernet import Fernet

def decrypt_model_mind_section(encrypted_dict, key):
    """
    Decrypts an encrypted 'Model_Mind' section using the provided key.
    
    Args:
        encrypted_dict (dict): The dictionary with encrypted values (actual_model and predicted_model).
        key (str): The base64-encoded Fernet key used for encryption.

    Returns:
        dict: Decrypted 'Model_Mind' section with plain-text values.
    """
    fernet = Fernet(key.encode())
    
    def decrypt_entry(entry_list, keys_to_decrypt):
        decrypted = []
        for entry in entry_list:
            decrypted_entry = {}
            for k, v in entry.items():
                if k in keys_to_decrypt:
                    decrypted_entry[k] = fernet.decrypt(v.encode()).decode()
                else:
                    decrypted_entry[k] = v
            decrypted.append(decrypted_entry)
        return decrypted

    decrypted_dict = {
        "actual_model": decrypt_entry(encrypted_dict.get("actual_model", []), ["model", "r2_score"]),
        "predicted_model": decrypt_entry(encrypted_dict.get("predicted_model", []), ["model", "confidence_score"])
    }
    
    return decrypted_dict
