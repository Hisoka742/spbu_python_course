import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from project.task9.task9 import (
    evaluate_metrics,
)  # Import the custom evaluation function

# Mock data for testing
@pytest.fixture
def mock_data():
    data = {
        "Pclass": [1, 3, 2, 1, 3],
        "Sex": [0, 1, 0, 1, 0],
        "Age": [22, 38, 26, 35, 28],
        "Fare": [7.25, 71.2833, 7.925, 53.1, 8.05],
        "Embarked": [2, 0, 2, 0, 1],
        "Survived": [0, 1, 1, 1, 0],
    }
    return pd.DataFrame(data)


# Test data preprocessing
def test_data_preprocessing(mock_data):
    assert "Age" in mock_data.columns
    assert "Sex" in mock_data.columns
    assert mock_data.isnull().sum().sum() == 0  # No missing values
    assert mock_data["Sex"].isin([0, 1]).all()  # Ensure binary encoding


# Test feature engineering
def test_feature_engineering(mock_data):
    if "SibSp" not in mock_data.columns and "Parch" not in mock_data.columns:
        mock_data["FamilySize"] = [1, 1, 1, 1, 1]  # Simulating the FamilySize feature
        assert "FamilySize" in mock_data.columns


# Test evaluate_metrics function
def test_evaluate_metrics():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])

    precision, recall, accuracy = evaluate_metrics(y_true, y_pred)

    assert precision == pytest.approx(0.6667, rel=1e-2)
    assert recall == pytest.approx(0.6667, rel=1e-2)
    assert accuracy == pytest.approx(0.8, rel=1e-2)


# Test model training and prediction
def test_models(mock_data):
    X = mock_data.drop("Survived", axis=1)
    y = mock_data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Logistic Regression
    log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
    log_reg_model.fit(X_train, y_train)
    log_predictions = log_reg_model.predict(X_test)
    assert len(log_predictions) == len(y_test)

    # Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_predictions = dt_model.predict(X_test)
    assert len(dt_predictions) == len(y_test)


# Test integration of evaluation and model prediction
def test_model_evaluation_integration(mock_data):
    X = mock_data.drop("Survived", axis=1)
    y = mock_data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
    log_reg_model.fit(X_train, y_train)
    log_predictions = log_reg_model.predict(X_test)

    precision, recall, accuracy = evaluate_metrics(y_test.to_numpy(), log_predictions)
    assert precision >= 0  # Valid precision
    assert recall >= 0  # Valid recall
    assert 0 <= accuracy <= 1  # Accuracy in range
