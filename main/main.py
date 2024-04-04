import sys
sys.path.append('..')
import pandas as pd
import mlflow
from Churn_predict.model_training import build_model
from Churn_predict.inference import make_predictions
from Churn_predict import selected_features

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.start_run()


training_data_df = pd.read_csv('../data/E Commerce Dataset.csv')
model_performance_dict = build_model(training_data_df)
print(model_performance_dict)

user_data_df = pd.read_csv('../data/test.csv')
user_data_selected = user_data_df[selected_features]
predictions = make_predictions(user_data_selected)
print(predictions)

mlflow.end_run()

