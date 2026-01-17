import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    """
    This class is responsible for taking raw input, preprocessing it 
    using the saved pipeline, and returning a prediction.
    """
    def __init__(self):
        pass

    def predict(self, features):
        """
        Input: A pandas DataFrame of features
        Output: Model prediction (e.g., a score or category)
        """
        try:
            # 1. Define paths to the 'frozen' objects created during training
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            # 2. Load the objects back into memory
            # load_object is a helper function (usually using pickle or dill)
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # 3. Transform the raw input features
            # It is crucial to use the SAME scaling/encoding used during training
            data_scaled = preprocessor.transform(features)

            # 4. Generate the prediction
            preds = model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    This class acts as a 'Mapper'. It maps the input from a Web Form (HTML/Flask)
    directly to a structured format (Pandas DataFrame) that the pipeline expects.
    """
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        # Assigning web form inputs to class variables
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Converts the class variables into a dictionary and then a DataFrame.
        This ensures column names match exactly what the model was trained on.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)