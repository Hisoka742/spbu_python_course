import pytest
from sklearn.datasets import load_iris  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
import numpy as np

# Assuming SimpleKNN is your custom k-NN implementation
from project.task_10.main import SimpleKNN


@pytest.fixture
def load_iris_data():
    return load_iris(return_X_y=True)


@pytest.fixture
def scale_data(load_iris_data):
    X, y = load_iris_data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def test_data_loading(load_iris_data):
    X, y = load_iris_data
    assert X.shape == (150, 4)  # 150 samples and 4 features
    assert y.shape == (150,)  # 150 target values


def test_data_scaling(scale_data):
    X_scaled, y = scale_data
    # Check that scaling has been done correctly
    assert np.isclose(
        np.mean(X_scaled, axis=0), 0
    ).all()  # mean should be approximately 0
    assert np.isclose(np.std(X_scaled, axis=0), 1).all()  # std should be 1


def test_knn_prediction(scale_data):
    X_scaled, y = scale_data
    knn = SimpleKNN(k=5)
    knn.fit(X_scaled, y)
    predictions = knn.predict(X_scaled)
    assert len(predictions) == 150  # Ensure we have a prediction for each instance
    assert set(predictions) <= set(y)  # Predictions should be among the known classes


# Example to test correct handling of a new parameter
def test_knn_variable_k(scale_data):
    X_scaled, y = scale_data
    for k in [1, 3, 5, 10]:
        knn = SimpleKNN(k=k)
        knn.fit(X_scaled, y)
        predictions = knn.predict(X_scaled)
        assert len(predictions) == 150  # Consistency check for different k values
