import pandas as pd
import numpy as np
import joblib
from Churn_predict.preprocess import preprocess_data_with_objects


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:

    input_data_processed = preprocess_data_with_objects(input_data)

    model = joblib.load('../models/model.joblib')

    predictions = model.predict(input_data_processed)

    return predictions
