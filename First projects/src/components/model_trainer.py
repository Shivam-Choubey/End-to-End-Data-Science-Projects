import os
import sys
from dataclasses import dataclass

# Importing various Regression algorithms
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Custom modules for logging, exception handling, and helper functions
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    """
    Configuration class to define file paths for saving the trained model.
    Using @dataclass provides a clean way to store configuration constants.
    """
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self):
        # Initialize the config to access the file path later
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        """
        Input: train_array and test_array (usually from the Data Transformation stage)
        Output: R2 Score of the best performing model
        """
        try:
            logging.info("Splitting training and test input data")
            
            # Splitting features (X) and target (y) from the input arrays
            # Assumes the target variable is in the last column [:, -1]
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define a dictionary of models to experiment with
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            
            # evaluate_models is a helper function that fits each model 
            # and returns a dictionary of {model_name: r2_score}
            models_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models
            )
            
            # Find the highest R2 score from the report
            best_model_score = max(models_report.values())
            
            # Find the name of the model that achieved that highest score
            best_model_name = list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]
            
            # Select the actual model object based on the best name
            best_model = models[best_model_name]

            # Threshold check: If the best model is poor (R2 < 0.6), stop the process
            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable accuracy")

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            # Save the best performing model as a pickle (.pkl) file in the artifacts folder
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            # Final verification: Predict on test data and calculate R2 score
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            return r2_square
        
        except Exception as e:
            # Wrap any errors in the CustomException class for detailed traceback
            raise CustomException(e, sys)