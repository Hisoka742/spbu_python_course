import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Import custom implementation
from main import CustomLinearRegression

@pytest.fixture
def california_data():
    """Load California housing data."""
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["MedianHouseValue"] = data.target
    return df

@pytest.fixture
def processed_data(california_data):
    """Process data: scale features and split into train/test."""
    df = california_data.copy()
    X = df.drop("MedianHouseValue", axis=1)
    y = df["MedianHouseValue"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

def test_data_loading(california_data):
    """Test if data is loaded correctly."""
    assert not california_data.empty, "Dataframe should not be empty"
    assert "MedianHouseValue" in california_data.columns, "Target column is missing"

def test_data_scaling(processed_data):
    """Test if data scaling produces the correct shape."""
    X_train, X_test, y_train, y_test = processed_data
    assert X_train.shape[1] == X_test.shape[1], "Feature dimensions must match"
    assert len(y_train) > 0, "Training target should not be empty"

def test_linear_regression(processed_data):
    """Test training and prediction with Linear Regression."""
    X_train, X_test, y_train, y_test = processed_data
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    assert mse > 0, "MSE should be positive"
    assert len(y_pred) == len(y_test), "Prediction length must match test target length"

def test_random_forest_regression(processed_data):
    """Test training and prediction with Random Forest."""
    X_train, X_test, y_train, y_test = processed_data
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    assert mse > 0, "MSE should be positive"
    assert len(y_pred) == len(y_test), "Prediction length must match test target length"

def test_custom_linear_regression(processed_data):
    """Test custom linear regression implementation."""
    X_train, X_test, y_train, y_test = processed_data
    custom_model = CustomLinearRegression(learning_rate=0.01, iterations=100)
    custom_model.fit(X_train, y_train.values)
    y_pred = custom_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    assert custom_model.weights is not None, "Weights should be initialized after training"
    assert mse > 0, "MSE should be positive"
    assert len(y_pred) == len(y_test), "Prediction length must match test target length"
