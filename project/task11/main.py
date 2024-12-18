# 1. Load and analyze data
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["MedianHouseValue"] = data.target

print("Dataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# 2. Data Cleaning
# Check and handle missing values
missing_values = df.isnull().sum()
print("\nMissing Values in Each Column:")
print(missing_values)

# Handle outliers using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# 3. Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
df.hist(bins=20, figsize=(15, 10))
plt.show()

# Scatter plots for key features
features = ["MedInc", "AveRooms", "AveOccup"]
for feature in features:
    sns.scatterplot(x=df[feature], y=df["MedianHouseValue"])
    plt.title(f"{feature} vs MedianHouseValue")
    plt.show()

# 4. Feature Engineering
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

df["RoomsPerHousehold"] = df["AveRooms"] / df["HouseAge"]
df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"]

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[["MedInc", "HouseAge"]])
df_poly = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(["MedInc", "HouseAge"]))
df = df.drop(["MedInc", "HouseAge"], axis=1).join(df_poly)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop("MedianHouseValue", axis=1))

# 5. Model Training with Hyperparameter Tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X = df_scaled
y = df["MedianHouseValue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "RandomForestRegressor": RandomForestRegressor(),
}

params = {
    "Ridge": {"alpha": [0.1, 1, 10]},
    "Lasso": {"alpha": [0.1, 1, 10]},
    "RandomForestRegressor": {"n_estimators": [50, 100], "max_features": ["auto", "sqrt"]},
}

for model_name, model in models.items():
    grid = GridSearchCV(model, params.get(model_name, {}), cv=5, scoring="neg_mean_squared_error")
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{model_name} MSE: {mse}, Best Params: {grid.best_params_}")

# 6. Custom Model Implementation
class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.random.randn(X_b.shape[1], 1)
        for _ in range(self.iterations):
            gradient = 2 / X_b.shape[0] * X_b.T.dot(X_b.dot(self.weights) - y.reshape(-1, 1))
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.weights)

# Custom model testing
custom_model = CustomLinearRegression(learning_rate=0.01, iterations=1000)
custom_model.fit(X_train, y_train)
y_pred_custom = custom_model.predict(X_test)
custom_mse = mean_squared_error(y_test, y_pred_custom)
print(f"Custom Model MSE: {custom_mse}")

# 7. AutoML Comparison
from h2o.automl import H2OAutoML
import h2o

h2o.init()
h2o_df = h2o.H2OFrame(df)
train, test = h2o_df.split_frame(ratios=[0.8], seed=42)
aml = H2OAutoML(max_models=10, max_runtime_secs=300)
aml.train(x=list(df.columns[:-1]), y="MedianHouseValue", training_frame=train)
print(aml.leaderboard)
