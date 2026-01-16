import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd     
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # Path where the preprocessing pickle (.pkl) file will be saved
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        # FIXED: Pointed to DataTransformationConfig instead of DataTransformation
        # (This avoids the infinite recursion loop error)
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_objects(self):
        '''
        This function defines the transformation pipelines for numerical and categorical data.
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity", # Fixed typo: "reace_ethnicity"
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            
            # Pipeline for numerical features: Handle missing values with median, then scale
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            # Pipeline for categorical features: Impute, One-Hot Encode, then scale
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)) # Fixed typo: "sdcaler"
                ]
            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            # Combine both pipelines into a single transformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read the CSV files generated in the Ingestion stage
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Reading of the train and test is completed")
            logging.info("Obtaining preprocessing objects")
            
            preprocessing_obj = self.get_data_transformer_objects()
            
            target_column_name = "math_score"
            
            # Separate Features (X) and Target (y) for both train and test
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing on training and testing dataframes.")
            
            # Fit and transform the training data; only transform the test data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Concatenate the input features and target into a single array using np.c_
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info("Saving preprocessing object")
            
            # Save the preprocessor object as a pickle file for future use in prediction
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            # Always raise CustomException instead of 'pass' to catch errors
            raise CustomException(e, sys)