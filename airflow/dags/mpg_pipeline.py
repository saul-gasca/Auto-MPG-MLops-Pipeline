from datetime import datetime
from airflow import DAG
from airflow.operators.empty import EmptyOperator
import os
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

import pandas as pd


import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ----- Paths (from Airflow's perspective, inside the worker container) -----
# The MPGDataset project is mounted into Airflow via a volume (added in the next step).
PROJECT_PATH ="/opt/airflow/mpg_project"
RAW_DATA_PATH = os.path.join(PROJECT_PATH,"datasets/raw/data_raw.csv")

HOST_PROJECT_PATH = "C:/Users/sgfar/anaconda_projects/PythonJupyterProject/MPGDataset"

MODEL_PATH = os.path.join(PROJECT_PATH, "models/model.pkl")
MIN_R2_THRESHOLD = 0.70

def validate_data():
    """Check dataset exist before tarining"""
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}.","Make sure the MPGDataset project is mounted correctly.")
    print(f"Raw dataset found at {RAW_DATA_PATH}")


def get_head_data():
    if os.path.exists(RAW_DATA_PATH):
        df = pd.read_csv(RAW_DATA_PATH)
        print(df.head(1))


def validate_metrics():
    """Load the trained model, compute test R², and fail if below threshold."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    else:
        with open(MODEL_PATH,'rb') as f:
            model = pickle.load(f)

    df = pd.read_csv(RAW_DATA_PATH)
    df_clean = df.dropna(subset=['horsepower'])
    features = ['weight','model_year','acceleration']
    X  = df_clean[features]
    y = df_clean['mpg']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test,y_pred)

    print(f"Test R2: {test_r2:.4f} | Threshold: {MIN_R2_THRESHOLD}")

    if test_r2 < MIN_R2_THRESHOLD:
        raise ValueError(
            f"Model R2 ({test_r2:.4f}) is below the acceptable threshold ({MIN_R2_THRESHOLD}). "
            "Model will NOT be promoted."
        )

    print(f"Model passed quality gate with R2 = {test_r2:.4f}")


def notify_success():
    """Log a success message when the full pipeline completes successfully."""
    print("=" * 60)
    print("MPG Pipeline completed successfully!")
    print("- Data validated")
    print("- Model trained and logged to MLflow")
    print("- Model passed the quality gate (R2 >= threshold)")
    print("- Model artifact ready at models/model.pkl")
    print("=" * 60)


with DAG(
    dag_id="mpg_pipepline",
    description="Auto MPG training pipeline orchestrated with Airflow",
    start_date=datetime(2026,7,27),
    schedule=None,
    catchup=False,
    tags=["mlops", "regression", "mpg"],

) as tag:

    start = EmptyOperator(task_id="start")

    validate_data_task = PythonOperator(task_id="validate_data",python_callable=validate_data)


    train_model_task = DockerOperator(
        task_id="train_model",
        image="mpgdataset-trainer:latest",
        auto_remove="success",
        mount_tmp_dir=False,
        mounts=[
            Mount(
                source=f'{HOST_PROJECT_PATH}/datasets',
                target="/app/datasets",
                type="bind"
            ),
            Mount(
                source=f"{HOST_PROJECT_PATH}/models",
                target="/app/models",
                type="bind",
            ),
            Mount(
                source=f"{HOST_PROJECT_PATH}/mlflow.db",
                target="/app/mlflow.db",
                type="bind",
            ),
        ],
        docker_url="unix:///var/run/docker.sock",
        network_mode="bridge",
    )

    notify_success_task = PythonOperator(
        task_id="notify_success",
        python_callable=notify_success,
    )

    validate_metrics_task = PythonOperator(
        task_id="validate_metrics",
        python_callable=validate_metrics,
    )


    end = EmptyOperator(task_id="end")

    start >> validate_data_task >>  train_model_task >> validate_metrics_task >> notify_success_task >> end
