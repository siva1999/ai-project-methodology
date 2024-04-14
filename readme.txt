AI Project Methodology


Overview


This repository outlines a structured methodology for developing and deploying machine learning projects using MLflow for experiment tracking and FastAPI for serving machine learning models. The project structure provides a clear organization of code, data, and artifacts to ensure reproducibility and scalability in AI projects.

Project Structure


main.py: Contains the main script for making predictions and logging with MLflow.
conda.yaml: Conda environment specification file.
MLproject: MLflow project configuration file.
mlruns: Folder used by MLflow to store run information, including parameters, metrics, and artifacts logged during the execution of MLflow projects.
mlartifacts: Folder intended to store artifacts generated during the execution of MLflow projects. Artifacts may include trained models, plots, visualizations, data files, or any other files generated during the machine learning workflow.
models: Folder to store trained machine learning models.
notebooks: Folder containing Jupyter notebooks for experimentation and analysis.
fast_api: Folder for developing FastAPI-based web services for serving machine learning models.
Churn_predict: Folder containing subdirectories for different stages of the machine learning workflow, including preprocessing, model training, and inference.


Setup


Clone the repository:

git clone git@github.com:siva1999/ai-project-methodology.git
cd ai-project-methodology


Create and Install the required dependencies:

conda create -y python=3.9 --name aipm
conda activate aipm
pip install -r requirements.txt



To run the project, follow these steps:
Start the MLflow UI:

mlflow ui
Run the FastAPI service:

uvicorn fast_api.api:app --reload

Execute the MLflow project:

mlflow run .


After executing these commands, the MLflow UI will be accessible at http://127.0.0.1:5000, where you can view experiment runs, metrics, parameters, and artifacts logged during the project execution.
