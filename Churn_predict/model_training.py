import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from Churn_predict import selected_features
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import mlflow
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
sys.path.append('..')


def pred_val(X_train_scaled, X_train_encoded, y_train):
    model = joblib.load('models/model.joblib')
    model.fit(np.concatenate(
        [X_train_scaled, X_train_encoded], axis=1), y_train)
    y_train_pred = model.predict(np.concatenate(
        [X_train_scaled, X_train_encoded], axis=1))
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    return rmse_train, mse_train


def model(X_train, categorical_columns, continuous_columns, onehot_encoder, y_train, scaler, encoder):  # noqa: E501
    model = RandomForestRegressor()
    X_train_scaled = scaler.transform(X_train[continuous_columns])
    onehot_encoded_features_train = encoder.transform(X_train[categorical_columns])  # noqa: E501
    X_train_processed = pd.concat([pd.DataFrame(X_train_scaled, columns=continuous_columns),  # noqa: E501
                               pd.DataFrame(onehot_encoded_features_train.toarray(), columns=onehot_encoder.get_feature_names_out(categorical_columns))],  # noqa: E501,E128
                              axis=1)  # noqa: E128
    model.fit(X_train_processed, y_train)
    joblib.dump(model, 'models/model.joblib')
    return X_train_scaled


def set_scaler(X_train, continuous_columns):
    scaler = StandardScaler()
    scaler.fit(X_train[continuous_columns])
    print("dumping scaler")
    joblib.dump(scaler, 'models/scaler.joblib')


def set_encoder(X_train, categorical_columns):
    onehot_encoder = OneHotEncoder(drop='first')
    onehot_encoder.fit(X_train[categorical_columns])
    print("dumping encoder")
    joblib.dump(onehot_encoder, 'models/encoder.joblib')


def build_model(data: pd.DataFrame) -> dict:
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2, random_state=42))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))

    target_column = 'Churn'
    df = data[selected_features + [target_column]]

    categorical_columns = df[selected_features].select_dtypes(
        include='object').columns
    continuous_columns = df[selected_features].select_dtypes(
        include='number').columns
    set_encoder(X_train, categorical_columns)

    encoder = joblib.load('models/encoder.joblib')

    set_scaler(X_train, continuous_columns)

    scaler = joblib.load('models/scaler.joblib')

    X_train_scaled = model(X_train, categorical_columns, continuous_columns,  encoder, y_train, scaler, encoder)  # noqa: E501
    X_train_encoded = encoder.transform(X_train[categorical_columns]).toarray()
    rmse_train, mse_train = pred_val(X_train_scaled, X_train_encoded, y_train)

    mlflow.log_metric("mse_train", mse_train)
    mlflow.log_metric("rmse_train", rmse_train)

    performances = {'rmse_train': rmse_train}
    return performances
