# Import necessary modules and libraries
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize Flask application
application = Flask(__name__)
app = application  # Create an alias

## Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

## Route for handling prediction data
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Create a CustomData object with form data
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        
        # Convert the custom data object to a pandas DataFrame
        pred_df = data.get_data_as_data_frame()
        
        # Create an instance of the prediction pipeline
        predict_pipeline = PredictPipeline()
        
        # Make prediction using the pipeline
        results = predict_pipeline.predict(pred_df)
        
        # Get the prediction value
        if hasattr(results, '__iter__'):  # If it's a list/array
            prediction_value = float(results[0])
        else:  # If it's a single value
            prediction_value = float(results)
        
        # Format to 2 decimal places
        prediction_value = round(prediction_value, 2)
        
        # Render the home.html template with prediction results
        return render_template('home.html', results=prediction_value)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)