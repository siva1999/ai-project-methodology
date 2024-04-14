import sys
from Churn_predict.model_training import build_model
from Churn_predict.inference import make_predictions
from Churn_predict import selected_features
from fastapi import FastAPI, HTTPException
import pandas as pd
sys.path.append('../')

app = FastAPI()


@app.post("/predict/")
async def predict_features():
    try:
        data_path = "data/E Commerce Dataset.csv"
        test_data_path = "data/test.csv"
        training_data_df = pd.read_csv(data_path)
        model_performance_dict = build_model(training_data_df)
        print(model_performance_dict)
        user_data_df = pd.read_csv(test_data_path)
        user_data_selected = user_data_df[selected_features]
        predictions = make_predictions(user_data_selected)
        print(predictions)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error {str(e)}")  # noqa: E501
