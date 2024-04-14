import pandas as pd
import numpy as np
import joblib
import mlflow
import os
from Churn_predict.preprocess import preprocess_data_with_objects

def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    input_data_processed = preprocess_data_with_objects(input_data)
    model = joblib.load('models/model.joblib')
    mlflow.log_artifact('models/model.joblib', 'models')
    predictions = model.predict(input_data_processed)
    mlflow.log_param("num_predictions", len(predictions))

    return predictions