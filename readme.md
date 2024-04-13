# Churn Prediction Project
## Project Overview
Our Churn Prediction project aims to predict customer churn for an E-commerce platform. Using machine learning techniques, specifically Random Forest Regressor, we analyze customer behavior and other relevant features to identify patterns that indicate the likelihood of churn.

## Setup and Installation
Ensure you have Python installed on your system. This project is built using Python 3.10 or higher.

1. Clone the repository to your local machine.

2. Navigate into the project directory.

3. Install the required packages using the requirements.txt file provided in the project directory.

```bash
pip install -r requirements.txt
```

The Python version used is 3.11.9. Based on the analysis and model building steps performed in the project, the requirements.txt file should include the necessary packages.

## Dataset
You can download the dataset from the repository data folder. The dataset contains the following columns:

|Column| Description |
|---|-----------|
| CustomerID |Unique customer ID  |
| Churn | Churn Flag  |
| Tenure | Tenure of customer in organization |
| PreferredLoginDevice | Preferred login device of customer |
| CityTier | City tier  |
| WarehouseToHome |  Distance in between warehouse to home of customer|
| PreferredPaymentMode  | Preferred payment method of customer |
| Gender |  Gender of customer |
| HourSpendOnApp |  Number of hours spend on mobile application or website|
| NumberOfDeviceRegistered  |  Total number of deceives is registered on particular customer |
| PreferedOrderCat |  Preferred order category of customer in last month |
| SatisfactionScore |  Satisfactory score of customer on service |
| MaritalStatus |  Marital status of customer |
| NumberOfAddress |  Total number of added added on particular customer |
| Complain |  Any complaint has been raised in last month |
| OrderAmountHikeFromlastYear |  OrderAmountHikeFromlastYear |
| CouponUsed |  Total number of coupon has been used in last month |
| OrderCount |  Total number of orders has been places in last month |
| DaySinceLastOrder |  Day Since last order by customer |
| CashbackAmount |  Average cashback in last month |

## Churn_predict Package
This is a Python package developed for predicting customer churn in E-commerce datasets. It encompasses a comprehensive workflow from data preparation to model evaluation and prediction, allowing users to efficiently process data, train machine learning models, and visualize important features influencing customer retention.

## Installation
Clone this repository to your local machine using:

```bash
git clone https://github.com/siva1999/ai-project-methodology
```
Navigate to the project directory and install the package using pip:

```bash
cd Churn_predict
pip install .
```
or
```bash
pip install Churn_predict
```
## Project Components
The project includes the following components:

1. `Preprocess.py`: Script for loading and initially cleaning the raw E-commerce dataset.Contains methods for feature extraction and preprocessing, ensuring data is suitable for model training.
2. `Model Training.py`: Facilitates the training of machine learning models using the preprocessed data, with a focus on Random Forest Regressor.
3. `Inference.py`: Provides functionality for making predictions on new data using the trained model.

## Usage
While it's recommended to explore individual scripts for granular control over the process, the package is designed to be used as follows:

Prepare your dataset and process your dataset further using the preprocess module.
Train your model with the model_training module.
Use the inference module for making predictions.
## Running the Project
To run the project:
Execute the necessary scripts or functions in the provided modules.
Use the `churn_prediction_final.ipynb` to see the model building and inference part.

## Conclusion
Our Churn Prediction project provides a comprehensive solution for predicting customer churn in E-commerce datasets. By leveraging machine learning techniques and efficient data processing, businesses can gain valuable insights into customer behavior and take proactive measures to retain customers and improve overall performance.