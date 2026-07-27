Auto MPG MLOps Pipeline

An end-to-end machine learning project predicting vehicle fuel efficiency (miles per gallon) using the classic Auto MPG dataset. This project goes beyond model training, focusing on real-world data science and MLOps best practices: exploratory data analysis, data versioning, experiment tracking, containerization, and CI/CD automation.

Overview

The goal of this project is to predict a car's fuel efficiency (mpg) based on its physical and mechanical characteristics (weight, model year, acceleration, etc.), while applying the full lifecycle of a production-oriented ML workflow.

Key Features
Exploratory Data Analysis (EDA): distribution analysis, correlation matrix, outlier detection using the IQR method, and multicollinearity assessment among predictors.
Data Cleaning: handling missing values with domain-informed decisions rather than blind imputation.
Data Versioning (DVC): raw and processed datasets are version-controlled separately (with a Google Drive remote), ensuring full reproducibility of the pipeline.
Feature Selection: multiple candidate models compared and selected based on correlation analysis and multicollinearity checks, not just raw performance.
Model Training & Validation: linear regression models evaluated with K-Fold cross-validation and a held-out test set.
Residual Analysis: diagnostic checks (residual distribution, residuals vs. predictions) to validate linear regression assumptions.
Experiment Tracking (MLflow): logging parameters, metrics (cross-validation and test), and model artifacts across different training runs for comparison.
Containerization (Docker): separate containers for training and serving, connected via Docker Compose and a shared volume.
Model Serving (FastAPI): a REST API that loads the trained model and serves real-time MPG predictions.
CI/CD (GitHub Actions): automated linting, Docker image builds, and publishing to Docker Hub on every push to main.
Tech Stack

Python · Pandas · Scikit-learn · Matplotlib/Seaborn · DVC · MLflow · FastAPI · Docker · Docker Compose · GitHub Actions

## Project Structure

 <img width="452" height="383" alt="image" src="https://github.com/user-attachments/assets/7cdf5c5c-37d9-4dae-baca-0f25031250a4" />

  
Dataset

Auto MPG Dataset (UCI Machine Learning Repository)

Modeling Approach

Three candidate models were trained and compared using 5-fold cross-validation and a held-out test set:

Model	Features	Test R²	Test RMSE
Model 1	weight	0.653	4.21
Model 2	horsepower, model_year, acceleration	0.678	4.05
Model 3 (selected)	weight, model_year, acceleration	0.794	3.25

Model 3 was selected as the final model: it achieved the best performance, the most stable cross-validation results, and — critically — all of its coefficients are logically consistent with real-world intuition (heavier cars consume more fuel, newer cars are more efficient, cars that accelerate slower tend to be more fuel-efficient).

<img width="476" height="458" alt="image" src="https://github.com/user-attachments/assets/56abc996-0082-4e77-92f4-8ac3587f865f" />

Interactive API docs are available at http://127.0.0.1:8000/docs.

3. View experiment tracking (MLflow UI)
bash
mlflow ui

Then open http://localhost:5000 to compare runs, parameters, and metrics.

4. Pull pre-built images from Docker Hub
bash
docker pull <your-dockerhub-username>/mpg-trainer:latest
docker pull <your-dockerhub-username>/mpg-api:latest
CI/CD Pipeline

Every push to main triggers a GitHub Actions workflow that:

Lints the codebase with flake8.
Builds both Docker images to confirm they are error-free.
Publishes the images to Docker Hub (only on pushes to main).
Data Versioning (DVC)

Raw and processed datasets are tracked with DVC, with Google Drive configured as the remote storage:

bash
dvc pull   # download the tracked datasets
dvc push   # upload a new version after changes
Roadmap

This project is part of a broader learning path. Planned next steps include:

Apache Airflow: orchestrating the full pipeline (extract → clean → train → evaluate) as a DAG.
BigQuery: migrating the dataset from local CSVs to a cloud data warehouse.
Model monitoring: tracking data/model drift over time.
Author

Saul Gasca Farrera GitHub
