import os
import ipp

## Paths and global variables
model_dir = "model_dump"
json_file_path = os.path.join(model_dir, "status.json")
data_folder = "data"

def initial_Check():
    # Perform the checks
    Cache = clear_pycache()
    Model_dir = model()
    json_dump = jsonCheck()
    data_dir = clean_data_folder()

    # Update status.json dynamically
    update_json_pre({
        "pre_check": {
            "Cache": Cache,
            "Model_dir": Model_dir,
            "data_dir": data_dir,
            "json_dump": json_dump
        }
    })

# defining json file structure
def jsonCheck():
    try:
        json_data = {
                    "pre_check": { "Cache": False, "Model_dir": False, "data_dir": False, "json_dump": False},
                    "load_data": { "csv": False, "excel": False, "other": False },
                    "Model_Mind": { },
                    "pre_processing": { "null_handling": False, "outliers": { "detection": False }, "CFS": { "feature_selection": False },
                        "Skew": { "High": { "handling": False, "features": [] }, "Moderate": { "handling": False, "features": [] }, "Low": { "handling": False, "features": []}}},
                    "modeling": {},
                    "hyperparameter_tuning": { "grid_search": False, "random_search": False },
                    "final_checks": { "evaluate_performance": False, "check_overfitting": False, "check_generalization": False}
                    }

        # Ensure model directory exists before saving JSON
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        json_file_path = os.path.join(model_dir, "status.json")
        with open(json_file_path, "w") as json_file:
            ipp.json.dump(json_data, json_file, indent=4)

        # print(f"JSON data saved to {json_file_path}")
        return True
    except Exception as e:
        # print(f"Error saving JSON: {e}")
        return False
    
# def clean_data_folder():
#     try:
#         # Ensure data folder exists
#         if not os.path.exists(data_folder):
#             os.makedirs(data_folder)

#         # Clean the data folder by removing all files
#         for filename in os.listdir(data_folder):
#             file_path = os.path.join(data_folder, filename)
#             if os.path.isfile(file_path) or os.path.isdir(file_path):
#                 ipp.shutil.rmtree(file_path) if os.path.isdir(file_path) else os.remove(file_path)

#         # print(f"Data folder cleaned: {data_folder}")
#         return True
#     except Exception as e:
#         # print(f"Error cleaning data folder: {e}")
#         return False

def clean_data_folder():
    try:
        # Ensure 'data' folder exists and clean it
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        for filename in os.listdir(data_folder):
            file_path = os.path.join(data_folder, filename)
            if os.path.isfile(file_path) or os.path.isdir(file_path):
                ipp.shutil.rmtree(file_path) if os.path.isdir(file_path) else os.remove(file_path)

        # Ensure 'Out_Put' folder inside 'model_dump' exists and clean it
        output_dir = os.path.join(model_dir, "Out_Put")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path) or os.path.isdir(file_path):
                ipp.shutil.rmtree(file_path) if os.path.isdir(file_path) else os.remove(file_path)

        return True
    except Exception as e:
        # print(f"Error cleaning folders: {e}")
        return False


# Clearing cache
def clear_pycache():
    try:
        pycache_dir = "__pycache__"
        if os.path.exists(pycache_dir):
            ipp.shutil.rmtree(pycache_dir)
            # print(f"Deleted {pycache_dir}")
        else:
            # print(f"No __pycache__ found")
            pass
        return True
    except Exception as e:
        # print(f"Error clearing __pycache__: {e}")
        return False

# Ensure the model directory exists and is empty
def model():
    try:
        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                file_path = os.path.join(model_dir, filename)
                if os.path.isfile(file_path) or os.path.isdir(file_path):
                    ipp.shutil.rmtree(file_path) if os.path.isdir(file_path) else os.remove(file_path)
        else:
            os.makedirs(model_dir)

        # print(f"Model directory is ready: {model_dir}")
        return True
    except Exception as e:
        # print(f"Error preparing model directory: {e}")
        return False

# Updates the status.json file without overwriting previous values.
def update_json_pre(update_data):

    try:
        # Load existing data if status.json exists
        if os.path.exists(json_file_path):
            with open(json_file_path, "r") as json_file:
                json_data = ipp.json.load(json_file)
        else:
            json_data = {}

        # Merge new update into the existing data
        for key, value in update_data.items():
            if isinstance(value, dict) and key in json_data:
                json_data[key].update(value)
            else:
                json_data[key] = value

        # Save updated JSON back to file
        with open(json_file_path, "w") as json_file:
            ipp.json.dump(json_data, json_file, indent=4)

        # print(f"✅ Updated JSON file: {json_file_path}")
    except Exception as e:
        # print(f"❌ Error updating JSON: {e}")
        pass

# 2. **Data Loading Function (Independent from Pre-check)**

def load_data(file_path):
    """
    Load data from the provided file path and return it as a DataFrame.
    The file will be saved in the 'data' folder and checked for its type.
    JSON status will be updated accordingly.
    """

    # Ensure the data folder exists
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Extract file name and extension
    file_name = os.path.basename(file_path)
    file_extension = os.path.splitext(file_name)[1].lower()

    # Define paths
    save_path = os.path.join(data_folder, file_name)

    # Save file to the 'data' folder
    with open(file_path, 'rb') as fsrc:
        with open(save_path, 'wb') as fdst:
            fdst.write(fsrc.read())

    # Check file extension and load the appropriate data
    if file_extension == '.csv':
        df = ipp.pd.read_csv(save_path)
        file_type = "csv"
    elif file_extension == '.xlsx':
        df = ipp.pd.read_excel(save_path)
        file_type = "excel"
    else:
        df = None
        file_type = "unknown"

    # Update the JSON file with the new status
    update_json_load_data(file_type)

    return df


def update_json_load_data(file_type):
    """
    Updates the status in the JSON file with information about the file loaded.
    """

    with open(json_file_path, 'r') as f:
        status = ipp.json.load(f)

    # Update the status based on the loaded file type
    if file_type == "csv":
        status["load_data"]["csv"] = True
    elif file_type == "excel":
        status["load_data"]["excel"] = True
    else: status["load_data"]["other"] = True

    # Write the updated status back to the JSON file
    with open(json_file_path, 'w') as f:
        ipp.json.dump(status, f, indent=4)

# Example usage:
# df = load_data('path_to_file.csv')

