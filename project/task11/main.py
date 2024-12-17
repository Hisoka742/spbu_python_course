# Data handling and computations
import pandas as pd  # Essential library for data manipulation and analysis.
import numpy as np  # Fundamental package for scientific computing with Python.

# Data visualization
import matplotlib.pyplot as plt  # Basic plotting library in Python.
import seaborn as sns  # Advanced visualization library based on matplotlib.

# Machine learning tools
from sklearn.datasets import (
    fetch_california_housing,
)  # Dataset utility from scikit-learn.
from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures,
)  # Preprocessing utilities for scaling and polynomial feature generation.
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)  # Utilities for splitting data and hyperparameter tuning.
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
)  # Linear model algorithms.
from sklearn.ensemble import RandomForestRegressor  # Ensemble method for regression.
from sklearn.metrics import mean_squared_error  # Performance metric.

# Statistical modeling
from scipy.stats import (
    gaussian_kde,
)  # Kernel density estimation for statistical analysis.

# Performance measurement
import time  # To measure execution time.

# AutoML tools
from h2o.automl import H2OAutoML  # Automated Machine Learning framework H2O.

# Load California housing data
california_housing = fetch_california_housing()
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
df["MedianHouseValue"] = california_housing.target

# Display the first few rows of the dataset
display(df.head())

# Show a quick description of the dataset
display(df.describe())

# Information about data types and missing values
display(df.info())
# Load the data
california_housing = fetch_california_housing()
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
df["MedianHouseValue"] = california_housing.target

# Initial Data Analysis
print("First few rows of the dataset:")
display(df.head())

print("\nData Types and Missing Values:")
display(df.info())

print("\nStatistical Summary of Numeric Features:")
display(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values in Each Column:")
display(missing_values)

# Data Cleaning
# As of the typical output, California Housing data from sklearn does not contain missing values,
# but here's how you might handle them if there were any.
# Example: Fill missing values with the median of that column
# df = df.fillna(df.median())

# Detect and handle outliers (simple example using IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

print("\nData after handling outliers:")
display(df.describe())

# Additional cleaning steps can be included here based on further analysis.
# Load the dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["MedianHouseValue"] = data.target

# Print basic stats to check data integrity
print(df.describe())  # This will show stats including min, max, mean, etc.

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Visualizing the distribution of each feature and the target variable
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle("Distribution of Features and Target Variable")
for i, col in enumerate(df.columns):
    sns.histplot(df[col], kde=True, ax=axes[i // 3, i % 3])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Standard scatter plots
features = ["MedInc", "AveRooms", "AveOccup", "Latitude", "Longitude"]
fig, axes = plt.subplots(3, 2, figsize=(15, 15))  # Adjusted for 5 features
fig.delaxes(axes[2, 1])  # Remove the unused subplot
fig.suptitle("Relationships between Selected Features and Median House Value")
for i, feature in enumerate(features):
    if df[feature].notnull().all() and df["MedianHouseValue"].notnull().all():
        sns.scatterplot(x=df[feature], y=df["MedianHouseValue"], ax=axes[i // 2, i % 2])
        display(f"Plotting {feature} vs MedianHouseValue")  # Debugging statement

plt.show()
# Adding a contour plot for feature 'MedInc' vs 'MedianHouseValue'
plt.figure(figsize=(8, 6))
x = df["MedInc"]
y = df["MedianHouseValue"]
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
plt.tricontourf(x, y, z, levels=14, cmap="viridis")
plt.colorbar(label="Density")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Contour Plot of Median Income vs. Median House Value")

plt.show()
# Load the dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["MedianHouseValue"] = data.target

# Create new interaction features
df["RoomsPerHousehold"] = df["AveRooms"] / df["HouseAge"]
df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"]

# Generate Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[["MedInc", "HouseAge"]])
df_poly = pd.DataFrame(
    poly_features, columns=poly.get_feature_names_out(["MedInc", "HouseAge"])
)

# Drop the original columns and join the polynomial features
df = df.drop(["MedInc", "HouseAge"], axis=1).join(df_poly)

# Scaling features
scaler = StandardScaler()
features = df.drop("MedianHouseValue", axis=1).columns
df_scaled = pd.DataFrame(
    scaler.fit_transform(df.drop("MedianHouseValue", axis=1)), columns=features
)
df_scaled["MedianHouseValue"] = df["MedianHouseValue"]

# Display the head of the transformed DataFrame
display(df_scaled.head())
# Load and split the data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the models and parameters for GridSearchCV
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "RandomForestRegressor": RandomForestRegressor(),
}

params = {
    "LinearRegression": {},  # Linear Regression typically doesn’t need hyperparameter tuning
    "Ridge": {"alpha": [0.1, 1, 10]},
    "Lasso": {"alpha": [0.1, 1, 10]},
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 200],
        "max_features": ["auto", "sqrt"],
    },
}

# Perform grid search with cross-validation
for model_name, model in models.items():
    clf = GridSearchCV(
        model, params[model_name], cv=5, scoring="neg_mean_squared_error"
    )
    clf.fit(X_train_scaled, y_train)
    print(f"Best parameters for {model_name}: {clf.best_params_}")
    y_pred = clf.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error for {model_name}: {mse}")
# Load and prepare the data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1),
    "Lasso": Lasso(alpha=0.1),
    "RandomForestRegressor": RandomForestRegressor(
        n_estimators=100, max_features="sqrt"
    ),
}

# Dictionary to store results
results = []

# Measure performance of each model
for name, model in models.items():
    start_time = time.time()  # Start time for training
    model.fit(X_train_scaled, y_train)  # Train model
    training_time = time.time() - start_time  # Calculate training time
    y_pred = model.predict(X_test_scaled)  # Predict using the model
    mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error

    # Store results in the list
    results.append({"Model": name, "MSE": mse, "Training Time": training_time})

# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(results)

# Plot Mean Squared Error for each model
plt.figure(figsize=(10, 6))
plt.bar(results_df["Model"], results_df["MSE"], color="orange")
plt.xlabel("Model")
plt.ylabel("Mean Squared Error")
plt.title("Comparison of Model Performance (MSE: Lower is Better)")
plt.show()

# Plot Training Time for each model
plt.figure(figsize=(10, 6))
plt.bar(results_df["Model"], results_df["Training Time"], color="blue")
plt.xlabel("Model")
plt.ylabel("Training Time (seconds)")
plt.title("Comparison of Model Training Time")
plt.show()


class CustomLinearRegression:
    def __init__(self, alpha=0, learning_rate=0.01, iterations=1000):
        self.alpha = alpha  # Regularization strength
        self.learning_rate = learning_rate  # Step size for gradient descent
        self.iterations = iterations  # Number of iterations for gradient descent
        self.weights = None  # Weights for the regression model

    def fit(self, X, y):
        # Adding bias term as column of ones
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        m, n = X_b.shape

        # Initialize weights randomly
        self.weights = np.random.randn(n, 1)

        # Gradient descent optimization
        for i in range(self.iterations):
            predictions = X_b.dot(self.weights)
            errors = predictions - y.reshape(-1, 1)
            gradients = (
                2 / m * X_b.T.dot(errors)
                + self.alpha * np.r_[np.zeros([1, 1]), self.weights[1:]]
            )
            self.weights -= self.learning_rate * gradients

    def predict(self, X):
        # Predict using learned weights
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.weights)

    def mse(self, y_true, y_pred):
        # Calculate mean squared error
        return np.mean((y_true - y_pred.flatten()) ** 2)


data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = CustomLinearRegression(alpha=0.1, learning_rate=0.01, iterations=1000)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
mse = model.mse(y_test, predictions)

print(f"Custom Model MSE: {mse}")

from tpot import TPOTRegressor
import h2o
from h2o.automl import H2OAutoML

h2o.init()

# Load data and convert to H2O Frame
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
hf = h2o.H2OFrame(df)

# Split the data
train, test = hf.split_frame(ratios=[0.8], seed=42)

# Run H2O AutoML
aml = H2OAutoML(max_models=10, seed=1, max_runtime_secs=300)
aml.train(x=data.feature_names, y="target", training_frame=train)

# Leaderboard
lb = aml.leaderboard
display(lb.head())

# Predict on test data
preds = aml.leader.predict(test)
from tpot import TPOTRegressor

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Instantiate and train TPOT
tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)

# Show the final model
print(tpot.fitted_pipeline_)

# Predict and evaluate the model
tpot_pred = tpot.predict(X_test)
tpot_mse = mean_squared_error(y_test, tpot_pred)
print(f"TPOT MSE: {tpot_mse}")
