import os
import sys
import pickle # Used for serializing Python objects (saving models/preprocessors to disk)
import dill   # Similar to pickle, but better at serializing complex objects like lambdas
import pandas as pd 
import numpy as np    
from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves a Python object (like a trained model or scaler) to a physical file.
    This allows you to 'load' the model later in a production environment.
    """
    try:
        # 1. Identify the folder name (e.g., 'artifacts')
        dir_path = os.path.dirname(file_path)

        # 2. Create the folder if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # 3. Open file in 'wb' (Write Binary) and dump the object
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    This function automates the training and testing of multiple models.
    Input: Training/Testing data and a dictionary of model objects.
    Output: A dictionary containing the R2 score for each model.
    """
    try:
        report = {}

        # Loop through the 'models' dictionary
        # models.items() gives us both the name (key) and the algorithm (value)
        for model_name, model_obj in models.items():
            
            # 1. Train the model using the training data
            model_obj.fit(X_train, y_train)
            
            # 2. Make predictions on both datasets to check for overfitting
            # y_train_pred = model_obj.predict(X_train) # Optional: check train score
            y_test_pred = model_obj.predict(X_test)
            
            # 3. Calculate the R2 Score (Accuracy metric for regression)
            # Higher is better (1.0 is a perfect fit)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # 4. Store the result in the report dictionary
            report[model_name] = test_model_score
            
        return report
        
    except Exception as e:
        raise CustomException(e, sys)