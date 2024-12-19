import pytest
import pandas as pd
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.pipeline import Pipeline# type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
from sklearn.linear_model import LinearRegression, Ridge, Lasso # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from project.task11.main import (
    models,
    param_grids,
)  # Import your models and param grids from the main module


@pytest.fixture
def load_data():
    """Load the California housing dataset and return as DataFrame."""
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["MedianHouseValue"] = data.target
    return df


@pytest.fixture
def processed_data(load_data):
    """Process the dataset: handle missing values, split into train/test."""
    df = load_data
    # Impute missing values
    df.iloc[:, :-1] = SimpleImputer(strategy="mean").fit_transform(df.iloc[:, :-1])

    # Features and target
    X = df.drop("MedianHouseValue", axis=1)
    y = df["MedianHouseValue"]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def test_data_loading(load_data):
    """Test if the dataset is loaded correctly."""
    df = load_data
    assert not df.empty, "The dataset should not be empty."
    assert (
        "MedianHouseValue" in df.columns
    ), "Target column 'MedianHouseValue' should exist."


def test_data_processing(processed_data):
    """Test if the data is processed correctly."""
    X_train, X_test, y_train, y_test = processed_data
    assert (
        X_train.shape[1] == X_test.shape[1]
    ), "Feature dimensions must match for train and test sets."
    assert len(y_train) > 0, "Training labels should not be empty."
    assert len(y_test) > 0, "Test labels should not be empty."


def test_pipeline_training(processed_data):
    """Test if models train and predict correctly."""
    X_train, X_test, y_train, y_test = processed_data

    # Train and evaluate each model
    for model_name, model in models.items():
        # Create pipeline
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("regressor", model),
            ]
        )

        # GridSearchCV
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids.get(model_name, {}),
            scoring="neg_mean_squared_error",
            cv=5,
            error_score="raise",
        )

        # Train and predict
        grid.fit(X_train, y_train)
        y_pred = grid.best_estimator_.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Assert results
        assert mse > 0, f"MSE for {model_name} should be positive."
        assert len(y_pred) == len(
            y_test
        ), "Prediction size must match the test label size."
