import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

RAW_DATA_PATH = "datasets/raw/data_raw.csv"
MODEL_OUTPUT_PATH = "models/model.pkl"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw Auto MPG dataset from a local CSV file."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing horsepower values."""
    df_clean = df.dropna(subset=['horsepower'])
    return df_clean


def train_model():
    print("Loading data...")
    df = load_data(RAW_DATA_PATH)

    print("Cleaning data...")
    df_clean = clean_data(df)

    # Features and target for Model 3 (the winning model)
    features = ['weight', 'model_year', 'acceleration']
    X = df_clean[features]
    y = df_clean['mpg']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("mpg-regression")

    with mlflow.start_run(run_name="model_3_weight_year_acceleration"):
        model = LinearRegression()

        # Cross-validation on train set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

        # Final training on the full train set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        test_r2 = r2_score(y_test, y_pred)
        test_rmse = root_mean_squared_error(y_test, y_pred)

        # Logging to MLflow
        mlflow.log_param("features", ", ".join(features))
        mlflow.log_metric("cv_r2_mean", cv_scores.mean())
        mlflow.log_metric("cv_r2_std", cv_scores.std())
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.sklearn.log_model(model, "model")

        print(f"CV R2 mean: {cv_scores.mean():.4f} | CV R2 std: {cv_scores.std():.4f}")
        print(f"Test R2: {test_r2:.4f} | Test RMSE: {test_rmse:.4f}")

    # Save the model as a .pkl file for the API container
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train_model()