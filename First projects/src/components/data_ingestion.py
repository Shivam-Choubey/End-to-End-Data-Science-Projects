import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# 1. Configuration Class
# Using @dataclass is a clean way to store file paths without needing an __init__
@dataclass
class DataIngestionConfig:
    # Defines where the processed files will be saved
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    
class DataIngestion:
    def __init__(self):
        # Initialize the config object to access the paths defined above
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or components")
        try:
            # 2. Reading the dataset
            # Note: Ensure this path matches your local folder structure
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Read the dataset as dataframe")
            
            # 3. Create 'artifacts' folder
            # We use os.path.dirname to get the folder name ('artifacts') from the full file path
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save the raw data before splitting
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # 4. Train-Test Split
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # 5. Exporting split datasets to CSV files in the artifacts folder
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            
            # Return the paths so the next pipeline stage (Data Transformation) knows where to find the data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            # Catching any error and wrapping it in your CustomException
            raise CustomException(e, sys)
            

if __name__=="__main__":
    # Create object and run the ingestion
    obj = DataIngestion()
    # Added parentheses to actually execute the method
    trian_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(trian_data, test_data)