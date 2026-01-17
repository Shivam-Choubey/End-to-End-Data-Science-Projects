# Import necessary modules and libraries
from flask import Flask, request, render_template  # Flask framework components
import numpy as np  # Numerical computing library
import pandas as pd  # Data manipulation library

from sklearn.preprocessing import StandardScaler  # For feature scaling (imported but not used in this file)
from src.pipeline.predict_pipeline import CustomData, PredictPipeline  # Custom modules for data processing and prediction

# Initialize Flask application
application = Flask(__name__)

# Create an alias for easier reference (common practice in Flask applications)
app = application

## Route for the home page (main entry point of the web application)
@app.route('/')
def index():
    # Render and return the index.html template when user visits the root URL
    return render_template('index.html') 

## Route for handling prediction data (supports both GET and POST methods)
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    # If the request method is GET (user first visits the page)
    if request.method == 'GET':
        # Render and return the home.html template with empty form
        return render_template('home.html')
    
    # If the request method is POST (user submitted the form)
    else:
        # Create a CustomData object with form data extracted from the request
        # Note: There appears to be a bug here - reading_score and writing_score are swapped
        data = CustomData(
            gender=request.form.get('gender'),  # Get gender from form
            race_ethnicity=request.form.get('ethnicity'),  # Get ethnicity from form
            parental_level_of_education=request.form.get('parental_level_of_education'),  # Get parental education level
            lunch=request.form.get('lunch'),  # Get lunch type from form
            test_preparation_course=request.form.get('test_preparation_course'),  # Get test prep course status
            reading_score=float(request.form.get('writing_score')),  # BUG: Should be writing_score, gets reading_score value
            writing_score=float(request.form.get('reading_score'))  # BUG: Should be reading_score, gets writing_score value
        )
        
        # Convert the custom data object to a pandas DataFrame
        pred_df = data.get_data_as_data_frame()
        
        # Print the DataFrame for debugging purposes (visible in server console)
        print(pred_df)
        print("Before Prediction")  # Debug message

        # Create an instance of the prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")  # Debug message
        
        # Make prediction using the pipeline
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")  # Debug message
        
        # Render the home.html template with prediction results
        # results[0] accesses the first (and presumably only) prediction result
        return render_template('home.html', results=results[0])
    

# Main entry point - runs the Flask application
if __name__ == "__main__":
    # Run the app on all available network interfaces (0.0.0.0)
    # This makes it accessible from other devices on the network
    # Default port is 5000
    app.run(host="0.0.0.0")