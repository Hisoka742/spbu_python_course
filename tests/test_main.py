import pytest
import pandas as pd
from main import (
    load_and_concat_data,
    preprocess_data,
    get_class_counts,
    get_age_by_class_and_gender,
    filter_k_survivors,
    max_relatives_with_survivor,
    compare_fare_by_cabin,
)


@pytest.fixture
def sample_data():
    """Provide a sample Titanic dataset."""
    data = {
        "PassengerId": [1, 2, 3, 4, 5],
        "Survived": [1, 0, 1, 0, 1],
        "Pclass": [1, 2, 3, 1, 3],
        "Name": [
            "Kelley, Mr. James",
            "Karl, Miss. Anna",
            "King, Mrs. Jane",
            "Smith, Mr. John",
            "Keane, Mr. Mike",
        ],
        "Sex": ["male", "female", "female", "male", "male"],
        "Age": [22, 38, None, 35, None],
        "SibSp": [1, 1, 0, 0, 0],
        "Parch": [0, 0, 0, 0, 1],
        "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "373450"],
        "Fare": [7.25, 71.2833, 8.05, 53.1, None],
        "Cabin": [None, "C85", None, "B28", None],
        "Embarked": ["S", "C", "S", "S", "Q"],
    }
    df = pd.DataFrame(data)
    return preprocess_data(df)


def test_class_counts(sample_data):
    """Test passenger count by class."""
    class_counts = get_class_counts(sample_data)
    assert class_counts[3] == 2  # Class 3 has 2 passengers


def test_age_by_class_and_gender(sample_data):
    """Test age statistics by class and gender."""
    youngest, oldest, age_difference = get_age_by_class_and_gender(sample_data)
    assert youngest[1] < oldest[1]
    assert age_difference > 0


def test_filter_k_survivors(sample_data):
    """Test filtering survivors with last names starting with 'K'."""
    k_survivors = filter_k_survivors(sample_data)
    assert (
        k_survivors.iloc[0]["Name"] == "Keane, Mr. Mike"
    )  # Keane should be first due to highest fare among Ks


def test_max_relatives_with_survivor(sample_data):
    """Test the maximum number of relatives with a survivor."""
    max_relatives = max_relatives_with_survivor(sample_data)
    assert max_relatives == 1  # Highest FamilySize among survivors is 1


def test_compare_fare_by_cabin(sample_data):
    """Test the comparison of average fares for passengers with and without cabin information."""
    avg_fare_with_cabin, avg_fare_without_cabin, fare_ratio = compare_fare_by_cabin(
        sample_data
    )
    assert (
        avg_fare_with_cabin > avg_fare_without_cabin
    )  # Average fare should be higher for those with cabin
    assert fare_ratio > 1
