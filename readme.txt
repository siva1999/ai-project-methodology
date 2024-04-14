# To run the project using MLflow, execute the below command in the root folder.

# run the mlflow server
mlflow ui

# run the fast API server

uvicorn fast_api.api:app --reload

# to run , replace the path with your local path 

mlflow run . -P data_path="C:\Users\puthu\siva\EPITA\S2\Ai_project_methodology\ai-project-methodology\data\E Commerce Dataset.csv" -P test_data_path="C:\Users\puthu\siva\EPITA\S2\Ai_project_methodology\ai-project-methodology\data\test.csv" -P model_path="dummy_value"


# AI Project Methodology

## Overview


## Project Structure
- `main.py`: Contains the main script for making predictions and logging with MLflow.
- `conda.yaml`: Conda environment specification file.
- `MLproject`: MLflow project configuration file.
- `mlruns`: Folder used by MLflow to store run information, including parameters, metrics, and artifacts logged during the execution of MLflow projects.
- `mlartifacts`: Folder intended to store artifacts generated during the execution of MLflow projects. Artifacts may include trained models, plots, visualizations, data files, or any other files generated during the machine learning workflow.
- `models`: Folder to store trained machine learning models.
- `notebooks`: Folder containing Jupyter notebooks for experimentation and analysis.
- `fast_api`: Folder for developing FastAPI-based web services for serving machine learning models.
- `Churn_predict`: Folder containing subdirectories for different stages of the machine learning workflow, including preprocessing, model training, and inference.

## Setup
1. Clone the repository:

    ```bash
    git clone git@github.com:siva1999/ai-project-methodology.git
    cd ai-project-methodology
    ```

2. Install the required dependencies:

    ```bash
    mlflow run .
    ```

## Usage
To run the project, execute the following command:

```bash

in the root folder execute the below commands:

activate the environment :

conda activate base

up the mlflow ui :

mlflow ui

up the fastAPI :

uvicorn fast_api.api:app --reload

run :

mlflow run .

