import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
sys.path.append('..')
from Churn_predict import selected_features


def build_model(data: pd.DataFrame) -> dict:
    X = data.drop('Churn', axis=1)
    y = data['Churn']
   
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2, random_state=42))
    target_column = 'Churn'
    df = data[selected_features+ [target_column]]

    categorical_columns = df[ selected_features].select_dtypes(
        include='object').columns
    continuous_columns = df[ selected_features].select_dtypes(
        include='number').columns

    encoder = joblib.load('../models/encoder.joblib')
    scaler = joblib.load('../models/scaler.joblib')

    X_train_encoded = encoder.transform(X_train[categorical_columns]).toarray()
    X_train_scaled = scaler.transform(X_train[continuous_columns])
  
    model = joblib.load('../models/model.joblib')
 
    model.fit(np.concatenate(
        [X_train_scaled, X_train_encoded], axis=1), y_train)
   
    y_train_pred = model.predict(np.concatenate(
        [X_train_scaled, X_train_encoded], axis=1))
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)

    performances = {'rmse_train': rmse_train}
    return performances
