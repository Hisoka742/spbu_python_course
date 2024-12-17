import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from project.task11.main import CustomLinearRegression


@pytest.fixture
def mock_data():
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def test_initialization():
    model = CustomLinearRegression(alpha=0.1, learning_rate=0.01, iterations=1000)
    assert model.alpha == 0.1
    assert model.learning_rate == 0.01
    assert model.iterations == 1000


def test_fit_predict(mock_data):
    X_train_scaled, X_test_scaled, y_train, _ = mock_data
    model = CustomLinearRegression(alpha=0.1, learning_rate=0.01, iterations=1000)
    model.fit(X_train_scaled, y_train)
    assert model.weights is not None  # Check if weights are initialized and modified


def test_predict(mock_data):
    _, X_test_scaled, _, _ = mock_data
    model = CustomLinearRegression(alpha=0.1, learning_rate=0.01, iterations=1000)
    predictions = model.predict(X_test_scaled)
    assert predictions is not None
    assert len(predictions) == len(X_test_scaled)


def test_mse(mock_data):
    _, X_test_scaled, _, y_test = mock_data
    model = CustomLinearRegression(alpha=0.1, learning_rate=0.01, iterations=1000)
    predictions = model.predict(X_test_scaled)
    calculated_mse = model.mse(y_test, predictions)
    assert calculated_mse >= 0  # MSE should be non-negative
