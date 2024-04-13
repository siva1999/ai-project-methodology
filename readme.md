# Churn Prediction Project
## Project Overview
Our Churn Prediction project aims to predict customer churn for an E-commerce platform. Using machine learning techniques, specifically Random Forest Regressor, we analyze customer behavior and other relevant features to identify patterns that indicate the likelihood of churn.

## Setup and Installation
Ensure you have Python installed on your system. This project is built using Python 3.10 or higher.

Clone the repository to your local machine.

Navigate into the project directory.

Install the required packages using the requirements.txt file provided in the project directory.

bash
Copy code
pip install -r requirements.txt

The Python version used is 3.11.9. Based on the analysis and model building steps performed in the project, the requirements.txt file should include the necessary packages.

## Dataset
You can download the dataset from the repository data folder. The dataset contains the following columns:

CustomerID: Unique customer ID
Churn: Churn Flag
Tenure: Tenure of customer in organization
PreferredLoginDevice: Preferred login device of customer
CityTier: City tier
WarehouseToHome: Distance in between warehouse to home of customer
PreferredPaymentMode: Preferred payment method of customer
Gender: Gender of customer
HourSpendOnApp: Number of hours spent on the mobile application or website
NumberOfDeviceRegistered: Total number of devices registered to a particular customer
PreferedOrderCat: Preferred order category of customer in the last month
SatisfactionScore: Satisfactory score of customer on service
MaritalStatus: Marital status of customer
NumberOfAddress: Total number of addresses added by a particular customer
Complain: Any complaint raised in the last month
OrderAmountHikeFromlastYear: Order Amount Hike From Last Year
CouponUsed: Total number of coupons used in the last month
OrderCount: Total number of orders placed in the last month
DaySinceLastOrder: Days since the last order by customer
CashbackAmount: Average cashback in the last month

## Churn_predict Package
This is a Python package developed for predicting customer churn in E-commerce datasets. It encompasses a comprehensive workflow from data preparation to model evaluation and prediction, allowing users to efficiently process data, train machine learning models, and visualize important features influencing customer retention.

## Installation
Clone this repository to your local machine using:

bash
Copy code
git clone https://github.com/siva1999/ai-project-methodology
Navigate to the project directory and install the package using pip:

bash
Copy code
cd Churn_predict
pip install .
or

bash
Copy code
pip install Churn_predict
## Project Components
The project includes the following components:

Preprocess: Script for loading and initially cleaning the raw E-commerce dataset.Contains methods for feature extraction and preprocessing, ensuring data is suitable for model training.
Model Training: Facilitates the training of machine learning models using the preprocessed data, with a focus on Random Forest Regressor.
Inference: Provides functionality for making predictions on new data using the trained model.

## Usage
While it's recommended to explore individual scripts for granular control over the process, the package is designed to be used as follows:

Prepare your dataset and process your dataset further using the preprocess module.
Train your model with the model_training module.
Use the inference module for making predictions.
## Running the Project
To run the project:
Execute the necessary scripts or functions in the provided modules.
Use the churn_prediction_final.ipynb to see the model building and inference part.

## Conclusion
Our Churn Prediction project provides a comprehensive solution for predicting customer churn in E-commerce datasets. By leveraging machine learning techniques and efficient data processing, businesses can gain valuable insights into customer behavior and take proactive measures to retain customers and improve overall performance.