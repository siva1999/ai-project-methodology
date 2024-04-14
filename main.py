import sys
import mlflow
import requests

sys.path.append('..')

def main():

    API_BASE_URL = "http://localhost:8000"
    MLFLOW_URL = "http://127.0.0.1:5000"  

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.start_run()
    print("============== Ml flow logging started =======================")
    print(" ****** Calling predict API end point ******")
    endpoint = f"{API_BASE_URL}/predict/"
    response = requests.post(endpoint)
    if response.status_code == 200:
        print(F" Successfully predicted ! , open {MLFLOW_URL} to see the run results.")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    mlflow.end_run()
    print("============== Ml flow logging stopped =======================")
if __name__ == "__main__":
    main()
