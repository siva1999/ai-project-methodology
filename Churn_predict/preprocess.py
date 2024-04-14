import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from Churn_predict import selected_features
import sys
sys.path.append('..')


def preprocess_data_with_objects(data: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = data[selected_features].select_dtypes(
        include='object').columns
    continuous_columns = data[selected_features].select_dtypes(
        include='number').columns
    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(data)
    data_imputed = imputer.transform(data)

    data_imputed_continuous = pd.DataFrame(data_imputed, columns=data.columns)[continuous_columns]  # noqa: E501
    data_imputed_categorical = pd.DataFrame(data_imputed, columns=data.columns)[categorical_columns]  # noqa: E501

    encoder = joblib.load('models/encoder.joblib')
    scaler = joblib.load('models/scaler.joblib')

    encoded_features = encoder.transform(data_imputed_categorical).toarray()
    scaled_features = scaler.transform(data_imputed_continuous)
    processed_data = pd.concat([
        pd.DataFrame(scaled_features, columns=continuous_columns),
        pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))  # noqa: E501
    ], axis=1)
    return processed_data
