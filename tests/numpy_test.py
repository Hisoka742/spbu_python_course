import pytest
import numpy as np
from project.Numpy.main import (
    load_iris_data,
    dataset_memory_size,
    normalize_data,
    categorize_feature,
    split_data,
    train_and_evaluate_classifier,
    reduce_and_visualize,
)

# Test data loading
def test_load_iris_data():
    features, labels = load_iris_data()
    assert features.shape == (150, 4)
    assert labels.shape == (150,)


# Test memory size calculation
def test_dataset_memory_size():
    features, labels = load_iris_data()
    feature_size, label_size = dataset_memory_size(features, labels)
    assert feature_size > 0
    assert label_size > 0


# Test data normalization
def test_normalize_data():
    features, _ = load_iris_data()
    normalized_features = normalize_data(features)
    assert normalized_features.min() == 0
    assert normalized_features.max() == 1


# Test feature categorization
def test_categorize_feature():
    features, _ = load_iris_data()
    categories = categorize_feature(features, column_index=0)
    assert set(categories) == {"small", "medium", "big"}


# Test data splitting
def test_split_data():
    features, labels = load_iris_data()
    train_features, test_features, train_labels, test_labels = split_data(
        features, labels
    )
    assert len(train_features) == 120
    assert len(test_features) == 30


# Test classifier training and evaluation
def test_train_and_evaluate_classifier():
    features, labels = load_iris_data()
    train_features, test_features, train_labels, test_labels = split_data(
        features, labels
    )
    accuracy = train_and_evaluate_classifier(
        train_features, test_features, train_labels, test_labels
    )
    assert 0 <= accuracy <= 1  # Check if accuracy is a valid probability


# Test dimensionality reduction and visualization
def test_reduce_and_visualize():
    features, labels = load_iris_data()
    reduced_features, _ = reduce_and_visualize(features, labels)
    assert reduced_features.shape == (150, 2)
