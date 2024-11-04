import pandas as pd


def load_and_concat_data(train_path, test_path):
    """Load and concatenate Titanic train and test datasets."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train["dataset_type"] = "train"
    test["dataset_type"] = "test"
    data = pd.concat([train, test], ignore_index=True)
    return data


def preprocess_data(data):
    """Preprocess the Titanic dataset."""
    # Convert categorical fields using pandas built-in functionality
    data["Survived"] = data["Survived"].astype("category")
    data["Pclass"] = data["Pclass"].astype("category")
    data["Sex"] = data["Sex"].astype("category")

    # Extract last name for filtering purposes
    data["LastName"] = data["Name"].apply(lambda name: name.split(",")[0].strip())

    # Feature engineering
    data["Title"] = data["Name"].apply(
        lambda name: name.split(",")[1].split(".")[0].strip()
    )
    data["FamilySize"] = data["SibSp"] + data["Parch"]
    data["CabinBool"] = data["Cabin"].notnull().astype(int)

    # Handle missing values for Age by calculating the median age for each title group without numpy
    title_medians = {}
    for title in data["Title"].unique():
        ages = [
            age for age in data.loc[data["Title"] == title, "Age"] if pd.notnull(age)
        ]
        median_age = sorted(ages)[len(ages) // 2] if ages else None
        title_medians[title] = median_age

    data["Age"] = data.apply(
        lambda row: title_medians[row["Title"]]
        if pd.isnull(row["Age"])
        else row["Age"],
        axis=1,
    )

    # Fill missing fare values with the median of all available fare values
    fares = [fare for fare in data["Fare"] if pd.notnull(fare)]
    median_fare = sorted(fares)[len(fares) // 2]
    data["Fare"].fillna(median_fare, inplace=True)

    # Drop irrelevant columns
    data.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)
    return data


def get_class_counts(data):
    """Return the count of passengers in each class."""
    return data["Pclass"].value_counts()


def get_age_by_class_and_gender(data):
    """Calculate the average age by class and gender."""
    age_grouped = data.groupby(["Pclass", "Sex"])["Age"].mean()
    youngest = age_grouped.idxmin(), age_grouped.min()
    oldest = age_grouped.idxmax(), age_grouped.max()
    age_difference = oldest[1] - youngest[1]
    return youngest, oldest, age_difference


def filter_k_survivors(data):
    """Return a sorted list of survivors with last names starting with 'K'."""
    k_survivors = data[(data["Survived"] == 1) & (data["LastName"].str.startswith("K"))]
    # Include "Name" from the original DataFrame
    return k_survivors[["Name", "Fare"]].sort_values(by="Fare", ascending=False)


def max_relatives_with_survivor(data):
    """Return the maximum number of relatives with any survivor."""
    data["Relatives"] = data["SibSp"] + data["Parch"]
    return data[data["Survived"] == 1]["Relatives"].max()


def compare_fare_by_cabin(data):
    """Calculate and compare the average fare for passengers with and without cabin information."""
    with_cabin = data[data["CabinBool"] == 1]["Fare"]
    without_cabin = data[data["CabinBool"] == 0]["Fare"]

    avg_fare_with_cabin = sum(with_cabin) / len(with_cabin)
    avg_fare_without_cabin = sum(without_cabin) / len(without_cabin)
    fare_ratio = (
        avg_fare_with_cabin / avg_fare_without_cabin
        if avg_fare_without_cabin != 0
        else None
    )
    return avg_fare_with_cabin, avg_fare_without_cabin, fare_ratio
