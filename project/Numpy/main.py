import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import requests


def load_iris_data():
    """
    Downloads and loads the Iris dataset.

    Returns:
        tuple: A tuple containing:
            - features (numpy.ndarray): A 2D array of shape (150, 4), where each row contains the
              four numeric feature values of an iris flower.
            - labels (numpy.ndarray): A 1D array of shape (150,), containing the species names of
              each flower as strings.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    response = requests.get(url)
    data = response.text.strip().split("\n")
    data = [line.split(",") for line in data if line]
    features = np.array([list(map(float, line[:4])) for line in data])
    labels = np.array([line[4] for line in data])
    return features, labels


def dataset_memory_size(features, labels):
    """
    Calculates and returns the memory size of the features and labels arrays.

    Args:
        features (numpy.ndarray): 2D array with numeric feature values of the dataset.
        labels (numpy.ndarray): 1D array with species names.

    Returns:
        tuple: A tuple containing:
            - feature_size (int): Memory size of the features array in bytes.
            - label_size (int): Memory size of the labels array in bytes.
    """
    feature_size = features.nbytes
    label_size = labels.nbytes
    return feature_size, label_size


def normalize_data(features):
    """
    Normalizes the features of the dataset to be in the range [0, 1].

    Args:
        features (numpy.ndarray): 2D array with numeric feature values.

    Returns:
        numpy.ndarray: The normalized features array with values in range [0, 1].
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)


def categorize_feature(features, column_index=0):
    """
    Categorizes a specific feature column into 'small', 'medium', and 'big' categories based on quartiles.

    Args:
        features (numpy.ndarray): 2D array with numeric feature values.
        column_index (int): Index of the feature column to categorize.

    Returns:
        numpy.ndarray: A 1D array with 'small', 'medium', or 'big' categories based on the specified feature.
    """
    feature_column = features[:, column_index]
    categories = np.empty(feature_column.shape, dtype=object)
    small, medium, big = np.quantile(feature_column, [0.25, 0.75])
    categories[feature_column < small] = "small"
    categories[(feature_column >= small) & (feature_column <= medium)] = "medium"
    categories[feature_column > medium] = "big"
    return categories


def split_data(features, labels, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Args:
        features (numpy.ndarray): 2D array with numeric feature values.
        labels (numpy.ndarray): 1D array with species names.
        test_size (float): Fraction of the dataset to include in the test split.
        random_state (int): Seed for the random number generator.

    Returns:
        tuple: A tuple containing:
            - train_features (numpy.ndarray): Training set features.
            - test_features (numpy.ndarray): Testing set features.
            - train_labels (numpy.ndarray): Training set labels.
            - test_labels (numpy.ndarray): Testing set labels.
    """
    return train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )


def train_and_evaluate_classifier(
    train_features, test_features, train_labels, test_labels
):
    """
    Trains a Support Vector Classifier (SVC) and evaluates it on the test set.

    Args:
        train_features (numpy.ndarray): Training set features.
        test_features (numpy.ndarray): Testing set features.
        train_labels (numpy.ndarray): Training set labels.
        test_labels (numpy.ndarray): Testing set labels.

    Returns:
        float: The accuracy of the classifier on the test set, in the range [0, 1].
    """
    classifier = SVC()
    classifier.fit(train_features, train_labels)
    accuracy = classifier.score(test_features, test_labels)
    return accuracy


def reduce_and_visualize(features, labels, n_components=2):
    """
    Reduces the dimensionality of the features using PCA and prepares data for visualization.

    Args:
        features (numpy.ndarray): 2D array with numeric feature values.
        labels (numpy.ndarray): 1D array with species names.
        n_components (int): The number of components to reduce the feature set to.

    Returns:
        tuple: A tuple containing:
            - reduced_features (numpy.ndarray): 2D array of shape (n_samples, n_components), the reduced feature set.
            - labels (numpy.ndarray): Original labels for the dataset, retained for color-coding in visualization.
    """
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features, labels
