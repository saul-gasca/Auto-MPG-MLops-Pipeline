# Auto MPG MLOps Pipeline

An end-to-end machine learning project predicting vehicle fuel efficiency (miles per gallon) using the classic **Auto MPG dataset**. This project goes beyond model training and focuses on real-world data science and MLOps best practices: exploratory data analysis, data versioning, experiment tracking, containerization, CI/CD automation, and workflow orchestration.

---

## Overview

The goal of this project is to predict a car's fuel efficiency (`mpg`) based on its physical and mechanical characteristics (weight, model year, acceleration, etc.), while applying the full lifecycle of a production-oriented ML workflow.

---

## Key Features

- **Exploratory Data Analysis (EDA):** distribution analysis, correlation matrix, outlier detection using the IQR method, and multicollinearity assessment among predictors.
- **Data Cleaning:** handling missing values with domain-informed decisions rather than blind imputation.
- **Data Versioning (DVC):** raw and processed datasets are version-controlled separately (with a Google Drive remote), ensuring full reproducibility of the pipeline.
- **Feature Selection:** multiple candidate models compared and selected based on correlation analysis and multicollinearity checks, not just raw performance.
- **Model Training & Validation:** linear regression models evaluated with K-Fold cross-validation and a held-out test set.
- **Residual Analysis:** diagnostic checks (residual distribution, residuals vs. predictions) to validate linear regression assumptions.
- **Experiment Tracking (MLflow):** logging parameters, metrics (cross-validation and test), and model artifacts across different training runs.
- **Containerization (Docker):** separate containers for training and serving, connected via Docker Compose and a shared volume.
- **Model Serving (FastAPI):** a REST API that loads the trained model and serves real-time MPG predictions.
- **CI/CD (GitHub Actions):** automated linting, Docker image builds, and publishing to Docker Hub on every push to `main`.
- **Workflow Orchestration (Apache Airflow):** DAG-based pipeline that automates data validation, model training, quality gate checks, and success notifications.

---

## Tech Stack

Python · Pandas · Scikit-learn · Matplotlib/Seaborn · DVC · MLflow · FastAPI · Docker · Docker Compose · GitHub Actions · Apache Airflow

---

## Project Structure

```
MPGDataset/
├── .github/
│   └── workflows/
│       └── ci.yml
├── airflow/
│   └── dags/
│       └── mpg_pipeline.py
├── datasets/
│   ├── raw/
│   └── preprocess/
├── notebooks/
│   └── linear_regression.ipynb
├── src/
│   ├── train.py
│   └── api.py
├── models/
├── Dockerfile.train
├── Dockerfile.api
├── docker-compose.yml
├── requirements.txt
├── .flake8
└── .gitignore
```

---

## Dataset

[Auto MPG Dataset](https://archive.ics.uci.edu/ml/datasets/auto+mpg) (UCI Machine Learning Repository)

---

## Modeling Approach

Three candidate models were trained and compared using 5-fold cross-validation and a held-out test set:

| Model | Features | Test R² | Test RMSE |
|-------|----------|---------|-----------|
| Model 1 | `weight` | 0.653 | 4.21 |
| Model 2 | `horsepower`, `model_year`, `acceleration` | 0.678 | 4.05 |
| **Model 3 (selected)** | `weight`, `model_year`, `acceleration` | **0.794** | **3.25** |

**Model 3** was selected as the final model. It achieved the best performance, the most stable cross-validation results, and — critically — all of its coefficients are logically consistent with real-world intuition: heavier cars consume more fuel, newer cars are more efficient, and cars that accelerate slower tend to be more fuel-efficient.

---

## Running the Project

### 1. Train the model and serve predictions with Docker Compose

```bash
docker compose up --build
```

This will:

- Build and run the `trainer` service, which loads the raw data, cleans it, trains the model, logs the experiment to MLflow (SQLite backend), and saves `model.pkl`.
- Build and run the `api` service, which waits for the trained model to be available and exposes it via FastAPI on port `8000`.

### 2. Test the API

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"weight\": 3200, \"model_year\": 80, \"acceleration\": 16.5}"
```

Expected response:

```json
{
  "predicted_mpg": 25.14
}
```

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

### 3. View experiment tracking (MLflow UI)

```bash
mlflow ui
```

Then open `http://localhost:5000` to compare runs, parameters, and metrics.

### 4. Pull pre-built images from Docker Hub

```bash
docker pull <your-dockerhub-username>/mpg-trainer:latest
docker pull <your-dockerhub-username>/mpg-api:latest
```

---

## CI/CD Pipeline

Every push to `main` triggers a GitHub Actions workflow that:

1. **Lints** the codebase with `flake8`.
2. **Builds** both Docker images to confirm they are error-free.
3. **Publishes** the images to Docker Hub (only on pushes to `main`).

---

## Workflow Orchestration (Apache Airflow)

The full training pipeline is orchestrated as an Airflow DAG (`airflow/dags/mpg_pipeline.py`), which chains the following tasks:

```
start → validate_data → train_model → validate_metrics → notify_success → end
```

**Task descriptions:**

- **validate_data** — ensures the raw dataset exists before training.
- **train_model** — runs the `mpgdataset-trainer` Docker image via `DockerOperator`.
- **validate_metrics** — loads the trained model, computes test R², and enforces a quality gate.
- **notify_success** — logs a success message once the pipeline completes.

### Key design decisions

- **DockerOperator over PythonOperator for training.** The training task launches the same containerized image used everywhere else, keeping training environments consistent between local runs, CI/CD, and orchestration.
- **Docker-outside-of-Docker (DooD).** Airflow's worker container has access to the host's Docker socket, allowing it to spawn sibling containers on the host without nested virtualization.
- **Quality gate.** The `validate_metrics` task acts as a guardrail — if a future retraining produces a model with `R² < 0.70`, the DAG fails and the model is not promoted, preventing silent degradation.
- **Manual trigger.** The DAG is configured with `schedule=None` for local development. In production it could be moved to a periodic schedule (`@weekly`, cron expression, etc.).

### Running the DAG

1. Ensure your Airflow instance has the Docker provider installed and the Docker socket mounted.
2. Copy `airflow/dags/mpg_pipeline.py` into your Airflow `dags/` folder.
3. Mount the MPGDataset project into the Airflow worker container so tasks can access datasets and models.
4. Trigger the DAG from the Airflow UI (`http://localhost:8080`).

---

## Data Versioning (DVC)

Raw and processed datasets are tracked with DVC, with Google Drive configured as the remote storage:

```bash
dvc pull   # download the tracked datasets
dvc push   # upload a new version after changes
```

---

## Roadmap

This project is part of a broader learning path. Planned next steps include:

- **BigQuery:** migrating the dataset from local CSVs to a cloud data warehouse.
- **Model monitoring:** tracking data and model drift over time.
- **Multi-cloud extension:** AWS SageMaker for training/serving and Terraform for Infrastructure as Code (as a separate follow-up project).

---

## Author

**Saul Gasca Farrera**  
[GitHub](https://github.com/saul-gasca)