import sys
import os
import pandas as pd
import mlflow

sys.path.append('..')


from Churn_predict.model_training import build_model
from Churn_predict.inference import make_predictions
from Churn_predict import selected_features



def main():
    data_path = r"C:\Users\puthu\siva\EPITA\S2\Ai_project_methodology\ai-project-methodology\data\E Commerce Dataset.csv"
    test_data_path = r"C:\Users\puthu\siva\EPITA\S2\Ai_project_methodology\ai-project-methodology\data\test.csv"
    print("Current working directory:", os.getcwd())

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.start_run()
    
    print(f"The data path is {data_path} and test data path is {test_data_path}")
    print("Current working directory 3 :", os.getcwd())
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
