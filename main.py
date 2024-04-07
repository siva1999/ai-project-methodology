import sys
import os
import pandas as pd
import mlflow

sys.path.append('..')


from Churn_predict.model_training import build_model
from Churn_predict.inference import make_predictions
from Churn_predict import selected_features



def main():
    data_path = "data/E Commerce Dataset.csv"
    test_data_path = "data/test.csv"

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.start_run()
    
    training_data_df = pd.read_csv(data_path)
    model_performance_dict = build_model(training_data_df)
    print(model_performance_dict)
    
    user_data_df = pd.read_csv(test_data_path)
    user_data_selected = user_data_df[selected_features]
    predictions = make_predictions(user_data_selected)
    print(predictions)
    
    mlflow.end_run()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <data_path> <test_data_path>")
        sys.exit(1)
    main()
