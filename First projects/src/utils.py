import os
import sys
import pickle # Used for serializing and de-serializing Python objects (saving to disk)
from src.exception import CustomException

def save_object(file_path, obj):
    """
    This function is responsible for saving a Python object to a specific file path.
    It is used to save the preprocessor object and the trained model.
    """
    try:
        # 1. Extract the directory path from the full file path 
        # (e.g., from 'artifacts/preprocessor.pkl', it extracts 'artifacts')
        dir_path = os.path.dirname(file_path)

        # 2. Create the directory if it doesn't already exist
        # 'exist_ok=True' ensures the code doesn't crash if the folder is already there
        os.makedirs(dir_path, exist_ok=True)

        # 3. Open the file in 'wb' mode (Write Binary)
        # We use Binary mode because pickle objects are not plain text
        with open(file_path, "wb") as file_obj:
            # 4. Use pickle to 'dump' (save) the object into the file
            pickle.dump(obj, file_obj)

    except Exception as e:
        # 5. Capture any errors (like Permission denied) and wrap them in your custom handler
        raise CustomException(e, sys)